"""V-T1 threshold sweep (pre-registered in E:/BladeReactor_Upwards_analysis/OVERNIGHT_LOG.md).

For one cine: A = median per-frame max over arc frames; for each f, binarize every frame of
[0, N) at T = f*A and record geometry-free observables. Writes <out>/<stem>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cine_frames as cf  # noqa: E402
from analyze_blade import binarize  # noqa: E402

F_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
N_FRAMES = 10_000
CLOSE_R, MIN_PX, MAX_PX = 3, 22, 3000      # detector held fixed (pre-registration)
ARC_FRAME_FACTOR = 10                      # arc frame: max > 10 x background p99


A_QUANTILE = 0.90   # amendment 1: A = 90th pct of per-frame max (see OVERNIGHT_LOG)
BLOCK = 1000


def brightness_scale(src, n):
    fmax = np.concatenate([src.block(a, min(BLOCK, n - a)).reshape(-1, src.H * src.W).max(1)
                           for a in range(0, n, BLOCK)]).astype(float)
    return float(np.quantile(fmax, A_QUANTILE)), fmax


def main(cine, out):
    src = cf.open_cine(cine)
    A, fmax = brightness_scale(src, N_FRAMES)
    res = dict(cine=cine, A=A, A_quantile=A_QUANTILE, fmax_pct=np.percentile(fmax, [10, 50, 90, 99]).tolist(),
               n=N_FRAMES, sweep=[])
    for f in F_GRID:
        T = f * A
        det = np.zeros(N_FRAMES, bool)
        L = np.full(N_FRAMES, np.nan)
        Wd = np.full(N_FRAMES, np.nan)
        top = np.full(N_FRAMES, np.nan)
        for i in range(N_FRAMES):
            b = binarize(src.frame(i), CLOSE_R, dark_guard=T, bright_thresh=T, min_arc_px=MIN_PX, max_arc_px=MAX_PX)
            if not b.any():
                continue
            det[i] = True
            sk = skeletonize(b)
            L[i] = sk.sum()
            Wd[i] = b.sum() / max(sk.sum(), 1)
            top[i] = np.nonzero(b.any(1))[0][0]
        d = np.abs(np.diff(top))
        d = d[np.isfinite(d)]
        res["sweep"].append(dict(f=f, T=T, det_frac=float(det.mean()), len_med=float(np.nanmedian(L)) if det.any() else None,
                                 width_med=float(np.nanmedian(Wd)) if det.any() else None,
                                 top_speed_med=float(np.median(d)) if d.size else None,
                                 top_speed_mean=float(np.mean(d)) if d.size else None,
                                 top_speed_p90=float(np.percentile(d, 90)) if d.size else None,
                                 area_p99=float(np.nanpercentile(L * Wd, 99)) if det.any() else None,
                                 n_reject_area=None))
        print(Path(cine).stem, f, round(T, 1), res["sweep"][-1], flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / (Path(cine).stem + ".json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
