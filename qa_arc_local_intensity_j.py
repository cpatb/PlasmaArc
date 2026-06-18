#!/usr/bin/env python3
"""
qa_arc_local_intensity_j.py  —  LOCAL intensity around the arc, to choose a
detection threshold that keeps the arc but rejects the surrounding plume.

For a handful of arc frames per recording, this:
  1. locates the arc spine (largest bright connected component -> skeleton),
  2. measures every pixel's distance from that spine,
  3. plots brightness vs distance-from-arc (the local fall-off), and
  4. compares two pixel populations directly:
        ARC      = pixels on/next to the spine     (distance <= ~1.5 px)
        SURROUND = far-field plume pixels          (distance 14-22 px)

The threshold you want sits ABOVE the SURROUND brightness (so plume is removed)
and BELOW the ARC brightness (so the arc is not eroded). The script draws that
window: surround ~99th percentile (must exceed) vs arc ~10th percentile (must
stay under). No clean gap => a single global threshold can't separate them.

Usage:
    python qa_arc_local_intensity_j.py --all "<...>/J Electrode"
    python qa_arc_local_intensity_j.py "<...>/HS_0.4mm_1500mL_(CO2)_trig_results"
    python qa_arc_local_intensity_j.py --all "<...>/J Electrode" --seed 180 --n-arc-frames 15
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, label as ndi_label
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAXDIST = 22            # px outward from the spine to profile
ARC_BAND = 1.5          # distance (px) counted as "arc"
SUR_LO, SUR_HI = 14, 22 # far-field band (px) = surrounding plume, clear of the arc halo


def sample_stack(results_dir, n_sample):
    tiffs = sorted((Path(results_dir) / 'tiff').glob('*.tiff'))
    if not tiffs:
        return None
    handles = [tifffile.TiffFile(str(tp)) for tp in tiffs]
    counts = [len(h.pages) for h in handles]
    starts = np.cumsum([0] + counts)
    total = sum(counts)
    want = np.linspace(0, total - 1, min(n_sample, total), dtype=int)
    out = []
    for gi in want:
        c = int(np.searchsorted(starts, gi, 'right') - 1)
        out.append(handles[c].pages[int(gi - starts[c])].asarray())
    for h in handles:
        h.close()
    return np.stack(out, 0)


def arc_spine(frame, seed, min_px=12):
    """Largest bright connected component -> 1px skeleton (the arc spine)."""
    b = frame > seed
    if b.sum() < min_px:
        return None
    lbl, n = ndi_label(b)
    if n == 0:
        return None
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    mask = lbl == sizes.argmax()
    if mask.sum() < min_px:
        return None
    sk = skeletonize(mask)
    return sk if sk.any() else None


def gather(stack, idx, seed):
    d_all, v_all, used = [], [], []
    for fi in idx:
        fr = stack[fi]
        sk = arc_spine(fr, seed)
        if sk is None:
            continue
        dist = distance_transform_edt(~sk)
        m = dist <= MAXDIST
        d_all.append(dist[m].ravel())
        v_all.append(fr[m].ravel().astype(float))
        used.append(fi)
    if not d_all:
        return None
    return np.concatenate(d_all), np.concatenate(v_all), used


def plot_recording(dist, val, name, out_path):
    edges = np.arange(0, MAXDIST + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med, q25, q75 = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = val[(dist >= lo) & (dist < hi)]
        if len(v):
            med.append(np.median(v)); q25.append(np.percentile(v, 25)); q75.append(np.percentile(v, 75))
        else:
            med.append(np.nan); q25.append(np.nan); q75.append(np.nan)
    med, q25, q75 = map(np.array, (med, q25, q75))

    arc = val[dist <= ARC_BAND]
    sur = val[(dist >= SUR_LO) & (dist <= SUR_HI)]
    sur_hi = np.percentile(sur, 99)     # threshold must EXCEED this to drop plume
    arc_lo = np.percentile(arc, 10)     # threshold must stay UNDER this to keep arc
    clean = sur_hi < arc_lo
    suggested = 0.5 * (sur_hi + arc_lo) if clean else None

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    ax.fill_between(centers, q25, q75, color='#9ecae1', alpha=0.6, label='middle 50% of pixels')
    ax.plot(centers, med, color='#08519c', lw=2, label='median brightness')
    ax.axvspan(0, ARC_BAND, color='#fdae6b', alpha=0.3, label='arc (<=%.1f px)' % ARC_BAND)
    ax.axvspan(SUR_LO, SUR_HI, color='#a1d99b', alpha=0.3, label='plume (%d-%d px)' % (SUR_LO, SUR_HI))
    ax.set_xlabel('distance from arc spine (pixels)')
    ax.set_ylabel('pixel brightness (0-255)')
    ax.set_title('how brightness falls off away from the arc')
    ax.set_xlim(0, MAXDIST); ax.set_ylim(0, 255)
    ax.grid(True, color='#e6e6e6'); ax.set_axisbelow(True)
    ax.legend(fontsize=8)

    ax = axs[1]
    ax.hist(sur, bins=64, range=(0, 255), color='#74c476', alpha=0.75, label='surrounding plume pixels', density=True)
    ax.hist(arc, bins=64, range=(0, 255), color='#fd8d3c', alpha=0.75, label='arc pixels', density=True)
    ax.axvline(sur_hi, color='#238b45', lw=1.4, ls='--', label='plume 99th pct = %d' % round(sur_hi))
    ax.axvline(arc_lo, color='#d94701', lw=1.4, ls='--', label='arc 10th pct = %d' % round(arc_lo))
    if clean:
        ax.axvspan(sur_hi, arc_lo, color='#fee391', alpha=0.6)
        ax.set_title('clean gap - use threshold ~ %d' % round(suggested))
    else:
        ax.set_title('overlap - faint arc edge blends into bright plume')
    ax.set_xlabel('pixel brightness (0-255)')
    ax.set_ylabel('relative frequency')
    ax.set_xlim(0, 255)
    ax.grid(True, color='#e6e6e6'); ax.set_axisbelow(True)
    ax.legend(fontsize=8)

    fig.suptitle(name, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return dict(sur_hi=sur_hi, arc_lo=arc_lo, clean=clean, suggested=suggested)


def pick_arc_frames(stack, k, bright):
    n_bright = (stack > bright).sum(axis=(1, 2))
    return np.argsort(n_bright)[::-1][:k]


def main():
    ap = argparse.ArgumentParser(description='Local arc intensity / threshold window (J TIFFs)')
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--all', default=None)
    ap.add_argument('--n-sample', type=int, default=4000)
    ap.add_argument('--n-arc-frames', type=int, default=15)
    ap.add_argument('--bright', type=int, default=150, help='DN for ranking arc frames')
    ap.add_argument('--seed', type=int, default=180, help='DN seed to locate the arc spine')
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    if args.all:
        dirs += sorted(Path(args.all).glob('*_results'))
    dirs = [d for d in dict.fromkeys(dirs) if (d / 'tiff').exists()]
    if not dirs:
        raise SystemExit('no *_results dirs with a tiff/ subfolder found')

    for d in dirs:
        name = d.name.replace('_results', '')
        stack = sample_stack(d, args.n_sample)
        if stack is None:
            print('  %s: no tiffs; skip' % name); continue
        idx = pick_arc_frames(stack, args.n_arc_frames, args.bright)
        got = gather(stack, idx, args.seed)
        if got is None:
            print('  %s: no arc spine at seed=%d (try lowering --seed)' % (name, args.seed)); continue
        dist, val, used = got
        r = plot_recording(dist, val, name, d / 'arc_local_intensity.png')
        msg = ('clean gap -> threshold ~%d' % round(r['suggested'])) if r['clean'] else 'OVERLAP'
        print('%s : plume99=%d  arc10=%d  %s  (%d frames)'
              % (name, round(r['sur_hi']), round(r['arc_lo']), msg, len(used)))
        # Persist the data-driven threshold so run_j_co2.py picks it up per
        # recording (read_threshold_txt looks in tiff/threshold_suggested.txt).
        # Only write when a clean arc/plume gap exists; on OVERLAP a single
        # global threshold cannot separate them and we leave it to the fallback.
        if r['clean'] and r['suggested'] is not None:
            tdir = d / 'tiff'
            if tdir.exists():
                (tdir / 'threshold_suggested.txt').write_text(
                    'threshold = %d\n'
                    '# data-driven: midpoint of plume-99th-pct (%d) and arc-10th-pct (%d)\n'
                    '# from qa_arc_local_intensity_j.py over %d arc frames\n'
                    % (round(r['suggested']), round(r['sur_hi']),
                       round(r['arc_lo']), len(used)))
                print('    wrote %s' % (tdir / 'threshold_suggested.txt'))


if __name__ == '__main__':
    main()
