"""
temporal_summary_xrecording.py — aggregate temporal_velocity_profiles results
across every recording and across multiple group sizes.

Reads the RESULT line emitted by temporal_velocity_profiles.py (run with
multiple --n-events values per recording) and produces:

  temporal_summary_xrecording.csv     — one row per (recording, n_events)
  temporal_summary_xrecording.{pdf,png,txt}
      grouped-bar plot of delta_median vs recording, one set of bars per
      group size, plus a per-recording sigma_sys line for context.

Pass the RESULT lines via stdin or via --paste-file. Each line must match the
canonical format::

    RESULT recording=<stem>  n_events=<k>  n_qualifying=<n>  med_first=<v>
           med_last=<v>  delta_med=<v>  sigma_sys=<v>  KS_D=<v>  KS_p=<v>
           n_obs_first=<n>  n_obs_last=<n>
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figstyle


_KV_RE = re.compile(r'([A-Za-z_]+)=([\-+0-9.eE]+|[^\s]+)')


def parse_lines(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if 'RESULT' not in line:
            continue
        d = {}
        for k, v in _KV_RE.findall(line):
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        if 'recording' in d:
            rows.append(d)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--paste-file', type=str, default=None,
                    help='File containing the RESULT lines (default: stdin).')
    ap.add_argument('--out-dir', type=str, required=True,
                    help='Where to write the summary CSV + figure.')
    args = ap.parse_args()

    text = (Path(args.paste_file).read_text() if args.paste_file
            else sys.stdin.read())
    df = parse_lines(text)
    if df.empty:
        print('[ERROR] no RESULT lines parsed.')
        sys.exit(1)

    for c in ('n_events', 'n_qualifying', 'med_first', 'med_last', 'delta_med',
              'sigma_sys', 'KS_D', 'KS_p', 'n_obs_first', 'n_obs_last'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['n_events'] = df['n_events'].astype(int)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'temporal_summary_xrecording.csv', index=False)
    print(df.to_string(index=False))

    # Plot
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(10, 4.8))

    recs = sorted(df['recording'].unique(), key=lambda s: s)
    Ns   = sorted(df['n_events'].unique())
    short_recs = [r.replace('.cine', '')
                   .replace('HS_', '')
                   .replace('(CO2)_trig', '')
                   .replace('_(2)', '_2')
                   .strip('_')
                  for r in recs]
    x       = np.arange(len(recs), dtype=float)
    width   = 0.8 / max(len(Ns), 1)
    colors  = {3: '#00204d', 5: '#5d7588', 10: '#fde725'}

    for i, n in enumerate(Ns):
        sub = df[df['n_events'] == n].set_index('recording').reindex(recs)
        offset = (i - (len(Ns) - 1) / 2.0) * width
        ax.bar(x + offset, sub['delta_med'],
                width=width * 0.95, color=colors.get(int(n), '#979c91'),
                edgecolor='black', linewidth=0.5,
                label=f'first/last n={int(n)}')

    # sigma_sys horizontal context: median of the three sigmas per recording.
    for j, r in enumerate(recs):
        s = float(df.loc[df['recording'] == r, 'sigma_sys'].median())
        ax.plot([j - 0.45, j + 0.45], [+s, +s], color='#979c91', lw=1.2)
        ax.plot([j - 0.45, j + 0.45], [-s, -s], color='#979c91', lw=1.2)

    ax.axhline(0.0, color='#000000', lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(short_recs, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Δ median local velocity  (last − first)   (m/s)')
    ax.set_xlabel('Recording')
    ax.legend(frameon=False, fontsize=9, loc='upper right',
               title='Group size', title_fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Temporal drift in arc-tip propagation velocity, '
                  'cross-recording summary', fontsize=10)
    # Annotate the grey context band
    ax.text(0.01, 0.98, 'Grey bars: ±σ_sys per recording '
            '(cancels for within-recording first-vs-last; shown for scale only)',
            transform=ax.transAxes, fontsize=8, color='#555',
            va='top', ha='left')
    fig.tight_layout()
    figstyle.save(fig, str(out / 'temporal_summary_xrecording'),
                  caption=(
        'Per-recording delta = median(last n events) - median(first n events) '
        'of advancing-front local velocity. Three group sizes (n=3,5,10). '
        'Grey horizontal segments mark +/-sigma_sys per recording (median of '
        'the three sigma_sys values), shown for visual scale only -- the '
        'systematic CANCELS for an internal first-vs-last comparison.\n\n'
        'Reading: a real heating/conditioning drift would show a consistent '
        'sign across group sizes within a recording, and ideally a consistent '
        'sign across recordings. Sign flips between n=3 and n=10 indicate '
        'the result is dominated by which specific events sit at the very '
        'ends (start-of-recording transients in particular), not a drift.\n\n'
        'KS p-values: see CSV column KS_p. Large n_obs (~hundreds) gives the '
        'KS test high power, so small p does not equate to a physically '
        'meaningful drift -- judge by delta_med vs sigma_sys, not by p alone.'))
    figstyle.verify(str(out / 'temporal_summary_xrecording'))
    print(f"\nWrote: {out / 'temporal_summary_xrecording'}.pdf / .png / .txt")


if __name__ == '__main__':
    main()
