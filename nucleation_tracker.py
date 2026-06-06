"""
nucleation_tracker.py  —  Along-blade attachment propagation analysis

Physical context
----------------
The BLADE electrode is an inverted-V (^) geometry: two blade edges meet near
the top with a minimum separation of ~2.4 mm at the apex.  Each new discharge
nucleates near the apex and the two arc-attachment points then propagate DOWN
their respective blade edges, away from the apex into the widening gap.

As the attachment points move further from the apex:
  - The inter-electrode separation at the attachment sites grows (the gap widens)
  - The arc must bridge an ever-larger gap, which eventually leads to extinction
    and re-nucleation at the apex for the next event

This script projects the measured attachment point positions (tip_a, tip_b from
blade_run2.py) onto the actual blade-line directions (from blade_lines.npy) to
compute physically meaningful along-blade distances and propagation speeds.

Key outputs
-----------
  nucleation_positions.csv     — per-event: along-blade distances, propagation
                                  velocities, inter-attachment separation,
                                  asymmetry (left minus right along-blade distance)
  nucleation_along_blade.png   — 4-panel summary:
                                   [A] along-blade position drift per event (nucleation)
                                   [B] along-blade position vs time (kymograph)
                                   [C] propagation velocity distribution
                                   [D] asymmetry (left − right along-blade) per event
  nucleation_separation.png    — inter-attachment separation vs along-blade progress;
                                  tests consistency with V-geometry and shows how
                                  far the arc travels before extinction
  nucleation_trajectory.png    — per-event along-blade trajectory (one line per event)

Reads:
  measurements_all_events.csv   (output of event_velocity.py)
  blade_lines.npy               (napari-annotated electrode edges)

Usage:
    python nucleation_tracker.py [measurements_all_events.csv] [output_dir]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import platform

# =============================================================================
# CONFIG
# =============================================================================

FPS       = 200_000    # frames per second
PX_PER_MM = 9.167      # px / mm

if platform.system() == 'Windows':
    BLADE_DIR = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE'
else:
    BLADE_DIR = Path('/sessions/epic-kind-sagan/mnt/BLADE')

BLADE_LINES_PATH = BLADE_DIR / 'blade_lines.npy'

# Minimum complete-arc frames in an event for it to appear in any analysis
MIN_COMPLETE_FRAMES = 3

# Minimum frames to fit a propagation velocity (OLS along blade vs time)
MIN_VEL_FRAMES = 5

# Sliding-window half-width (in complete-arc frames, not real frames) for
# instantaneous propagation velocity.  Larger = smoother; smaller = better
# time resolution.  N=5 is appropriate given typical 10–50 complete-arc
# frames per event.
LOCAL_VEL_N = 5

# =============================================================================
# BLADE GEOMETRY HELPERS
# =============================================================================

def load_blade_lines(path):
    """
    Load blade_lines.npy.  Shape (2, 2, 2): [blade_A, blade_B] × [pt0, pt1] × [row, col].
    pt0 = apex end (near top of image, minimum electrode separation).
    pt1 = far end  (lower in image, larger electrode gap).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"blade_lines.npy not found at {path}.\n"
            f"Re-run the napari blade-line annotation step.")
    bl = np.load(path).astype(float)
    assert bl.shape == (2, 2, 2), f"Expected shape (2,2,2), got {bl.shape}"
    print(f"Blade lines loaded:")
    for i, name in enumerate(('A (right)', 'B (left)')):
        r0, c0 = bl[i, 0]
        r1, c1 = bl[i, 1]
        length_mm = np.linalg.norm([r1 - r0, c1 - c0]) / PX_PER_MM
        print(f"  Blade {name}: ({r0:.1f},{c0:.1f})→({r1:.1f},{c1:.1f}) px  "
              f"length={length_mm:.1f} mm")
    apex_sep_px = np.linalg.norm(bl[0, 0] - bl[1, 0])
    print(f"  Apex separation: {apex_sep_px / PX_PER_MM:.2f} mm")
    return bl


def perp_dist_to_blade(row_px, col_px, blade):
    """Perpendicular distance from pixel (row_px, col_px) to the infinite line through blade."""
    p  = np.array([row_px, col_px])
    p0 = blade[0]
    p1 = blade[1]
    v  = p1 - p0
    v_len = np.linalg.norm(v)
    if v_len < 1e-9:
        return np.linalg.norm(p - p0)
    t    = np.dot(p - p0, v) / np.dot(v, v)
    proj = p0 + t * v
    return float(np.linalg.norm(p - proj))


def along_blade_mm(row_px, col_px, blade):
    """
    Signed distance from the apex (blade[0]) along the blade direction, in mm.
    Positive = moving toward blade[1] (away from apex, deeper into the gap).
    Negative = above the apex (should not happen for real arc attachment).
    """
    p  = np.array([row_px, col_px])
    p0 = blade[0]
    p1 = blade[1]
    v  = p1 - p0
    v_len = np.linalg.norm(v)
    if v_len < 1e-9:
        return 0.0
    t = np.dot(p - p0, v) / np.dot(v, v)
    return float(t * v_len / PX_PER_MM)   # convert px → mm


def gap_at_distance(d_A_mm, d_B_mm, blade_lines):
    """
    Euclidean distance between two attachment points that lie d_A_mm and d_B_mm
    along blade A and blade B respectively, in mm.
    """
    # Position of each attachment in pixel space
    def pos_on_blade(d_mm, blade):
        p0 = blade[0]
        p1 = blade[1]
        v  = p1 - p0
        v_len = np.linalg.norm(v)
        return p0 + (d_mm * PX_PER_MM / v_len) * v

    pA = pos_on_blade(d_A_mm, blade_lines[0])
    pB = pos_on_blade(d_B_mm, blade_lines[1])
    return float(np.linalg.norm(pA - pB) / PX_PER_MM)

# =============================================================================
# LOAD AND PROJECT
# =============================================================================

def load(csv_path):
    df = pd.read_csv(csv_path)
    needed = {'tip_a_row', 'tip_a_col', 'tip_b_row', 'tip_b_col',
              'arc_complete', 'arc_detected', 'event_id', 'frame', 'file'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}\n"
            f"Re-run blade_run2.py (tip_a/b) and event_velocity.py (event_id).")
    for col in ['tip_a_row', 'tip_a_col', 'tip_b_row', 'tip_b_col', 'frame']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['arc_complete'] = (pd.to_numeric(df['arc_complete'], errors='coerce')
                          .fillna(0).astype(bool))
    print(f"Loaded {len(df):,} rows | "
          f"{df['arc_complete'].sum():,} complete-arc frames | "
          f"{df['event_id'].nunique():,} events")
    return df


def project_tips(df, blade_lines):
    """
    For each complete-arc frame, assign tip_a and tip_b to their nearest
    blade and compute the along-blade distance (mm from apex) for each.

    Assignment is by perpendicular distance to blade line — this is robust
    even if tip_a/tip_b labels swap between frames.

    Adds columns:
      blade_A_dist_mm  — along-blade distance of the tip assigned to blade A
      blade_B_dist_mm  — along-blade distance of the tip assigned to blade B
      gap_mm           — Euclidean distance between the two attachment points
      asymmetry_mm     — blade_A_dist_mm − blade_B_dist_mm  (+ = A ahead of B)
    """
    complete = df[df['arc_complete'] & (df['event_id'] >= 0)].copy()
    complete = complete.dropna(subset=['tip_a_row', 'tip_a_col',
                                       'tip_b_row', 'tip_b_col'])

    blA = blade_lines[0]   # blade A
    blB = blade_lines[1]   # blade B

    dA_list, dB_list, gap_list = [], [], []

    for _, row in complete.iterrows():
        ta = (row['tip_a_row'], row['tip_a_col'])
        tb = (row['tip_b_row'], row['tip_b_col'])

        # Perpendicular distance from each tip to each blade
        da_to_A = perp_dist_to_blade(*ta, blA)
        da_to_B = perp_dist_to_blade(*ta, blB)
        db_to_A = perp_dist_to_blade(*tb, blA)
        db_to_B = perp_dist_to_blade(*tb, blB)

        # Assign: tip closer to blade A → blade A attachment, etc.
        if da_to_A + db_to_B <= da_to_B + db_to_A:
            # tip_a → blade A,  tip_b → blade B
            d_A = along_blade_mm(*ta, blA)
            d_B = along_blade_mm(*tb, blB)
        else:
            # tip_a → blade B,  tip_b → blade A  (labels swapped)
            d_A = along_blade_mm(*tb, blA)
            d_B = along_blade_mm(*ta, blB)

        gap = gap_at_distance(max(d_A, 0.0), max(d_B, 0.0), blade_lines)
        dA_list.append(d_A)
        dB_list.append(d_B)
        gap_list.append(gap)

    complete = complete.copy()
    complete['blade_A_dist_mm'] = dA_list
    complete['blade_B_dist_mm'] = dB_list
    complete['gap_mm']          = gap_list
    complete['asymmetry_mm']    = (np.array(dA_list) - np.array(dB_list))

    print(f"Projected {len(complete):,} complete-arc frames onto blade lines")
    return complete

# =============================================================================
# PER-EVENT METRICS
# =============================================================================

def sliding_vel(dist_arr, frame_arr, half=LOCAL_VEL_N):
    """
    Per-frame along-blade velocity (mm/s) via centred sliding-window OLS.
    Returns array same length as dist_arr, NaN at boundaries with < 3 points.
    """
    n   = len(dist_arr)
    vel = np.full(n, np.nan)
    for k in range(n):
        lo = max(0, k - half)
        hi = min(n, k + half + 1)
        if hi - lo < 3:
            continue
        slope, _ = np.polyfit(frame_arr[lo:hi].astype(float),
                              dist_arr[lo:hi], 1)
        vel[k] = slope * FPS   # px/frame → mm/s  (dist already in mm)
    return vel


def compute_per_event(proj_df):
    """
    Per-event summary: nucleation position (along-blade at first frame),
    mean propagation velocity, max along-blade extent, asymmetry at nucleation.
    """
    records = []
    for eid, grp in proj_df.groupby('event_id'):
        grp = grp.sort_values('frame')
        n   = len(grp)
        if n < MIN_COMPLETE_FRAMES:
            continue

        frames = grp['frame'].values
        dA     = grp['blade_A_dist_mm'].values
        dB     = grp['blade_B_dist_mm'].values
        gap    = grp['gap_mm'].values

        # Nucleation = first frame of event (closest to apex)
        nuc_A  = float(dA[0])
        nuc_B  = float(dB[0])

        # Propagation velocity along each blade (OLS over event)
        vel_A_ms = np.nan
        vel_B_ms = np.nan
        if n >= MIN_VEL_FRAMES:
            f = frames.astype(float)
            sl_A, _ = np.polyfit(f, dA, 1)
            sl_B, _ = np.polyfit(f, dB, 1)
            vel_A_ms = sl_A * FPS / 1000   # mm/s → m/s
            vel_B_ms = sl_B * FPS / 1000

        records.append({
            'event_id'       : int(eid),
            'file'           : grp['file'].iloc[0],
            'frame_start'    : int(frames[0]),
            'n_complete'     : n,
            'nuc_A_mm'       : round(nuc_A, 3),
            'nuc_B_mm'       : round(nuc_B, 3),
            'nuc_gap_mm'     : round(float(gap[0]), 3),
            'nuc_asym_mm'    : round(nuc_A - nuc_B, 3),  # + = A ahead at nucleation
            'max_A_mm'       : round(float(np.nanmax(dA)), 3),
            'max_B_mm'       : round(float(np.nanmax(dB)), 3),
            'max_gap_mm'     : round(float(np.nanmax(gap)), 3),
            'prop_vel_A_ms'  : round(vel_A_ms, 4) if not np.isnan(vel_A_ms) else np.nan,
            'prop_vel_B_ms'  : round(vel_B_ms, 4) if not np.isnan(vel_B_ms) else np.nan,
            'asym_vel_ms'    : (round(vel_A_ms - vel_B_ms, 4)
                                if not (np.isnan(vel_A_ms) or np.isnan(vel_B_ms))
                                else np.nan),
        })

    return pd.DataFrame(records)

# =============================================================================
# FIGURES
# =============================================================================

def plot_summary(ev, proj_df, out_dir):
    """
    3-panel summary  (nucleation_summary.png):
      [0] Propagation velocity distribution — both blades, all runs overlaid.
          Fill = blade A; step outline = blade B.  Global bins for cross-run
          comparison.  Units: m/s (values ~1–8 m/s for a gliding arc).
      [1] Nucleation position from apex — boxplot per run, A vs B side-by-side.
          Shows where each event first attaches and run-to-run reproducibility.
      [2] Nucleation asymmetry (A − B along-blade) — histogram per run.
          Positive = blade A attachment starts further from apex.  A systematic
          non-zero shift reveals a geometric or flow bias.
    """
    files      = list(ev['file'].unique())
    colors     = ['#1565C0', '#E65100', '#2E7D32']
    run_labels = [f'Run {i+1}' for i in range(len(files))]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── [0] Velocity distribution ─────────────────────────────────────────────
    ax = axes[0]
    all_v = pd.concat([ev['prop_vel_A_ms'].dropna(),
                       ev['prop_vel_B_ms'].dropna()]).values
    v_lo  = max(0.0, np.percentile(all_v, 1))
    v_hi  = np.percentile(all_v, 99)
    bins  = np.linspace(v_lo, v_hi, 30)

    for fname, col, lbl in zip(files, colors, run_labels):
        sub = ev[ev['file'] == fname]
        vA  = sub['prop_vel_A_ms'].dropna()
        vB  = sub['prop_vel_B_ms'].dropna()
        ax.hist(vA, bins=bins, color=col, alpha=0.55, label=lbl)
        ax.hist(vB, bins=bins, color=col, alpha=0.0,
                histtype='step', lw=1.8)

    ax.set_xlabel('Propagation velocity (m/s)')
    ax.set_ylabel('Event count')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── [1] Nucleation position boxplot (A vs B per run) ─────────────────────
    ax = axes[1]
    n   = len(files)
    x_A = np.arange(n) * 2.2
    x_B = x_A + 0.8
    bp_kw = dict(widths=0.65, patch_artist=True,
                 showmeans=True, meanline=True,
                 meanprops={'color': 'white', 'lw': 2},
                 medianprops={'linewidth': 0},
                 whiskerprops={'lw': 1.2},
                 capprops={'lw': 1.2},
                 flierprops={'marker': '.', 'ms': 3, 'alpha': 0.5})

    data_A = [ev[ev['file'] == f]['nuc_A_mm'].dropna().values for f in files]
    data_B = [ev[ev['file'] == f]['nuc_B_mm'].dropna().values for f in files]

    bpA = ax.boxplot(data_A, positions=x_A, **bp_kw)
    bpB = ax.boxplot(data_B, positions=x_B, **bp_kw)
    for patch, col in zip(bpA['boxes'], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
        bpA['whiskers'][colors.index(col)*2].set_color(col)
        bpA['whiskers'][colors.index(col)*2+1].set_color(col)
        bpA['caps'][colors.index(col)*2].set_color(col)
        bpA['caps'][colors.index(col)*2+1].set_color(col)
    for patch in bpB['boxes']:
        patch.set_facecolor('#aaaaaa'); patch.set_alpha(0.55)

    ax.set_xticks(x_A + 0.4)
    ax.set_xticklabels(run_labels)
    ax.set_ylabel('Along-blade distance from apex (mm)')
    ax.grid(alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── [2] Asymmetry distribution ────────────────────────────────────────────
    ax = axes[2]
    all_asym = ev['nuc_asym_mm'].dropna().values
    a_lo = np.percentile(all_asym, 1); a_hi = np.percentile(all_asym, 99)
    abins = np.linspace(a_lo, a_hi, 25)

    for fname, col, lbl in zip(files, colors, run_labels):
        sub = ev[ev['file'] == fname]['nuc_asym_mm'].dropna()
        ax.hist(sub, bins=abins, color=col, alpha=0.55, label=lbl)

    ax.axvline(0, color='#333', lw=1.2, ls='--')
    ax.set_xlabel('Nucleation asymmetry  A - B  (mm)')
    ax.set_ylabel('Event count')
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'nucleation_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_separation(proj_df, out_dir):
    """
    Single panel  (nucleation_separation.png): inter-attachment gap (mm) vs
    mean along-blade distance, all runs overlaid with different colours.

    The gap should widen monotonically if the arc attachment follows the V
    geometry.  Deviations reveal arc bowing, asymmetric attachment, or
    measurement noise.  Running medians are overlaid per run to show the
    central trend without being overwhelmed by point scatter.
    """
    files  = list(proj_df['file'].unique())
    colors = ['#1565C0', '#E65100', '#2E7D32']

    fig, ax = plt.subplots(figsize=(6, 5))

    for fname, col in zip(files, colors):
        sub    = proj_df[proj_df['file'] == fname].copy()
        d_mean = (sub['blade_A_dist_mm'] + sub['blade_B_dist_mm']) / 2
        lbl    = f"Run {files.index(fname)+1}"

        # Raw points (thin, low alpha)
        ax.scatter(d_mean, sub['gap_mm'], color=col, s=1.5,
                   alpha=0.18, rasterized=True)

        # Running median for the trend line
        sort_idx   = d_mean.argsort()
        d_sorted   = d_mean.iloc[sort_idx].values
        gap_sorted = sub['gap_mm'].iloc[sort_idx].values
        win = max(30, len(d_sorted) // 40)
        med_x, med_y = [], []
        for k in range(0, len(d_sorted) - win, win // 2):
            med_x.append(np.mean(d_sorted[k:k+win]))
            med_y.append(np.mean(gap_sorted[k:k+win]))
        if med_x:
            ax.plot(med_x, med_y, color=col, lw=2.0, label=lbl)

    ax.set_xlabel('Mean along-blade distance (mm)')
    ax.set_ylabel('Inter-attachment gap (mm)')
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'nucleation_separation.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_trajectories(proj_df, ev, out_dir):
    """
    Per-event along-blade trajectories  (nucleation_trajectory.png).

    Layout: 1 row × N_runs panels.  Each panel shows ONE run with BOTH
    blades overlaid:
      solid  line  = blade A (right)
      dashed line  = blade B (left)

    Colour encodes propagation velocity using a SINGLE shared plasma scale
    computed across all runs, so colours are directly comparable across panels.
    A single figure-level colourbar avoids the overlap problem that arises when
    each panel carries its own colourbar.

    y-axis starts at 0 (apex = 0 mm) so it is clear that events nucleate
    2–4 mm from the apex, not at the apex itself.
    """
    from matplotlib.lines import Line2D

    files  = list(proj_df['file'].unique())
    n_runs = len(files)
    cmap   = plt.get_cmap('plasma')

    # Global velocity scale: 0 → 97th percentile across all runs + both blades
    all_v = pd.concat([ev['prop_vel_A_ms'].dropna(),
                       ev['prop_vel_B_ms'].dropna()]).values
    v_max = float(np.percentile(all_v[all_v > 0], 97)) if (all_v > 0).any() else 1.0
    norm  = plt.Normalize(0, v_max)

    fig, axes = plt.subplots(1, n_runs, figsize=(5.5 * n_runs, 5),
                             sharey=True)
    if n_runs == 1:
        axes = [axes]

    for col_i, (ax, fname) in enumerate(zip(axes, files)):
        sub_proj = proj_df[proj_df['file'] == fname]
        sub_ev   = ev[ev['file'] == fname]
        vel_map  = sub_ev.set_index('event_id')

        for eid, grp in sub_proj.groupby('event_id'):
            grp     = grp.sort_values('frame')
            time_ms = (grp['frame'].values - grp['frame'].values[0]) / FPS * 1000

            row = vel_map.loc[eid] if eid in vel_map.index else None
            vA  = (row['prop_vel_A_ms'] if row is not None
                   and not np.isnan(row['prop_vel_A_ms']) else np.nan)
            vB  = (row['prop_vel_B_ms'] if row is not None
                   and not np.isnan(row['prop_vel_B_ms']) else np.nan)
            cA  = cmap(norm(vA)) if not np.isnan(vA) else '#999999'
            cB  = cmap(norm(vB)) if not np.isnan(vB) else '#999999'

            ax.plot(time_ms, grp['blade_A_dist_mm'].values,
                    color=cA, lw=0.8, alpha=0.7)
            ax.plot(time_ms, grp['blade_B_dist_mm'].values,
                    color=cB, lw=0.8, alpha=0.5, ls='--')

        ax.set_xlabel('Time within event (ms)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Along-blade distance from apex (mm)')

    # One shared colourbar for the whole figure
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label('Propagation velocity (m/s)')

    # A vs B legend
    legend_handles = [
        Line2D([0], [0], color='gray', lw=1.5, ls='-',  label='Blade A (right)'),
        Line2D([0], [0], color='gray', lw=1.5, ls='--', label='Blade B (left)'),
    ]
    axes[-1].legend(handles=legend_handles, fontsize=8, frameon=False,
                    loc='lower right')

    # tight_layout + figure-level colorbar: leave 8% on the right for the cbar
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    out = out_dir / 'nucleation_trajectory.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(ev):
    files = list(ev['file'].unique())
    print(f"\n{'─'*60}")
    print(f"  Along-blade propagation summary")
    print(f"{'─'*60}")
    for fname in files:
        sub = ev[ev['file'] == fname]
        lbl = f"Run {files.index(fname)+1}"
        vA   = sub['prop_vel_A_ms'].dropna()   # m/s
        vB   = sub['prop_vel_B_ms'].dropna()   # m/s
        asym = sub['nuc_asym_mm'].dropna()
        print(f"\n  {lbl}  ({fname})")
        print(f"    Events analysed              : {len(sub)}")
        print(f"    Nucleation pos A  (mean)   : {sub['nuc_A_mm'].mean():.2f} mm from apex")
        print(f"    Nucleation pos B  (mean)   : {sub['nuc_B_mm'].mean():.2f} mm from apex")
        print(f"    Nucleation asym   (mean+/-std): {asym.mean():.2f} +/- {asym.std():.2f} mm")
        print(f"    Nucleation gap    (mean)   : {sub['nuc_gap_mm'].mean():.2f} mm")
        print(f"    Max extent A      (mean)   : {sub['max_A_mm'].mean():.2f} mm")
        print(f"    Max extent B      (mean)   : {sub['max_B_mm'].mean():.2f} mm")
        print(f"    Max gap at ext.   (mean)   : {sub['max_gap_mm'].mean():.2f} mm")
        if len(vA):
            print(f"    Prop. vel blade A (mean)      : {vA.mean():.3f} m/s  "
                  f"(range {vA.min():.3f}-{vA.max():.3f})")
        if len(vB):
            print(f"    Prop. vel blade B (mean)      : {vB.mean():.3f} m/s  "
                  f"(range {vB.min():.3f}-{vB.max():.3f})")
    print(f"{'-'*60}\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    script_dir = Path(__file__).parent.resolve()
    csv_path   = (Path(sys.argv[1]) if len(sys.argv) > 1
                  else script_dir / 'measurements_all_events.csv')
    out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        print("Run event_velocity.py first to generate measurements_all_events.csv")
        sys.exit(1)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Nucleation tracker (along-blade propagation)")
    print(f"  CSV    : {csv_path}")
    print(f"  Output : {out_dir}")
    print(f"{'='*55}\n")

    blade_lines = load_blade_lines(BLADE_LINES_PATH)
    df          = load(csv_path)
    proj_df     = project_tips(df, blade_lines)

    if len(proj_df) == 0:
        print("[ERROR] No complete-arc frames with valid tip_a/tip_b data.")
        sys.exit(1)

    ev = compute_per_event(proj_df)
    print(f"  Per-event records: {len(ev)}")

    ev.to_csv(out_dir / 'nucleation_positions.csv', index=False)
    print(f"Saved: {out_dir / 'nucleation_positions.csv'}")

    print_summary(ev)
    plot_trajectories(proj_df, ev, out_dir)   # nucleation_trajectory.png
    plot_summary(ev, proj_df, out_dir)        # nucleation_summary.png
    plot_separation(proj_df, out_dir)         # nucleation_separation.png

    print("Done.\n")


if __name__ == '__main__':
    main()
