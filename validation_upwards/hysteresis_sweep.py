"""PRE-REGISTRATION 2 sweep (OVERNIGHT_LOG.md): hysteresis detector, T_low = a*N, T_high = b*N.

N (noise ceiling) = 99.99th pct of pixels in the darkest 20 % of 2000 frames spread over the record.
Smear ceiling S = 99th pct of pixels 10..60 px to the RIGHT of a saturated (>= 4078) pixel in the
same row, in sampled frames containing saturation, excluding pixels above T_high of that (a,b)
(computed once with the raw frame: excluding pixels > 2*N is the conservative variant used).
Writes <out>/<stem>_hyst.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cine_frames as cf  # noqa: E402
from analyze_blade import binarize  # noqa: E402

A_GRID = [1.0, 1.3, 1.6, 2.0, 3.0]
B_GRID = [3, 5, 8, 12]
N_FRAMES = 3000
NOISE_SAMPLE, DARK_FRAC, NOISE_Q = 2000, 0.20, 99.99
SAT_DN = 4078
CLOSE_R, MIN_PX, MAX_PX = 3, 22, 10**9     # area ceiling disabled here; the flood cut is validated separately


def noise_and_smear(src):
    idx = np.linspace(0, src.n - 1, NOISE_SAMPLE).astype(int)
    fr = np.stack([src.frame(int(i)) for i in idx])
    fmax = fr.reshape(len(fr), -1).max(1)
    dark = fr[fmax <= np.quantile(fmax, DARK_FRAC)]
    N = float(np.percentile(dark, NOISE_Q))
    vals = []
    for f in fr[fmax >= SAT_DN]:
        rr, cc = np.nonzero(f >= SAT_DN)
        for r in np.unique(rr):
            c_last = cc[rr == r].max()
            seg = f[r, c_last + 10:c_last + 61].astype(float)
            vals.append(seg[seg <= 2 * N] if N > 0 else seg)
    S = float(np.percentile(np.concatenate(vals), 99)) if vals and sum(len(v) for v in vals) else float("nan")
    return N, S, int((fmax >= SAT_DN).sum()), float(np.median(fmax)), float(np.quantile(fmax, DARK_FRAC))


def main(cine, out):
    src = cf.open_cine(cine)
    N, S, n_sat, fmax_med, dark_cut = noise_and_smear(src)
    res = dict(cine=cine, N=N, smear_S=S, n_sat_frames_of_2000=n_sat, fmax_median=fmax_med,
               dark_frame_max_cut=dark_cut, n=N_FRAMES, grid=[])
    fr = src.block(0, N_FRAMES)
    for b in B_GRID:
        for a in A_GRID:
            lo, hi = a * N, b * N
            if lo >= hi:
                continue
            det = np.zeros(N_FRAMES, bool)
            L = np.full(N_FRAMES, np.nan)
            W = np.full(N_FRAMES, np.nan)
            top = np.full(N_FRAMES, np.nan)
            area = np.full(N_FRAMES, np.nan)
            for i in range(N_FRAMES):
                m = binarize(fr[i], CLOSE_R, dark_guard=hi, bright_thresh=hi, min_arc_px=MIN_PX,
                             max_arc_px=MAX_PX, low_thresh=lo)
                if not m.any():
                    continue
                det[i] = True
                sk = skeletonize(m)
                L[i] = sk.sum()
                area[i] = m.sum()
                W[i] = area[i] / max(L[i], 1)
                top[i] = np.nonzero(m.any(1))[0][0]
            d = np.abs(np.diff(top))
            d = d[np.isfinite(d)]
            row = dict(a=a, b=b, T_low=lo, T_high=hi, det_frac=float(det.mean()),
                       len_med=float(np.nanmedian(L)) if det.any() else None,
                       width_med=float(np.nanmedian(W)) if det.any() else None,
                       area_p50=float(np.nanmedian(area)) if det.any() else None,
                       area_p99=float(np.nanpercentile(area, 99)) if det.any() else None,
                       top_speed_mean=float(d.mean()) if d.size else None,
                       top_speed_p99=float(np.percentile(d, 99)) if d.size else None)
            res["grid"].append(row)
            print(Path(cine).stem, row, flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, Path(cine).stem + "_hyst.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
