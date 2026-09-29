"""Per-file px/mm from the 2.4 mm minimum electrode gap (folder name), measured from arc attachment.

Method (blades are not visible; only the arc's attachment is):
 1. frames sampled in WINDOWS x WIN frames across the record; binarize at the validated T.
 2. keep clean arches: skeleton with exactly 2 endpoints, both endpoints below (larger row than)
    the mask's top row by >= ARCH_MIN_RISE_PX, feet on opposite sides of the frame's arc midline.
 3. foot = the mask pixel that is most extreme along the skeleton's outward direction at the
    endpoint (within FOOT_SEARCH_PX of it): the skeleton itself stops ~half a width short.
 4. RANSAC line (skimage LineModelND) through left feet and through right feet separately;
    opening angle between the lines (check against the 45 deg folder label).
 5. blade tip = the inlier foot with the lowest TIP_QUANTILE projection toward the apex,
    i.e. the lowest attachment the arc ever reaches on that blade (re-strikes at the gap).
 6. gap_px = |tip_L - tip_R|; px_per_mm = gap_px / 2.4.
Writes <out>/<stem>_gap.json and a QA png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from skimage.measure import LineModelND, ransac
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cine_frames as cf  # noqa: E402
from analyze_blade import binarize, skeleton_endpoints  # noqa: E402

GAP_MM = 2.4
WINDOWS, WIN = 5, 2000
CLOSE_R, MIN_PX, MAX_PX = 3, 22, 10**9
ARCH_MIN_RISE_PX = 5
FOOT_SEARCH_PX = 8
RANSAC_RESID_PX = 2.0
TIP_QUANTILE = 0.01
SEED = 20260928
EDGE_MIN_HITS_FRAC = 0.002   # edge-cloud pixel kept if hit by >= 0.2 % of clean arches (and >= 3)


def outward_dir(sk, e, n_back=5):
    """unit vector from the skeleton pixel n_back steps inward to the endpoint e."""
    pts = np.argwhere(sk)
    d = np.hypot(*(pts - e).T)
    near = pts[(d > 0) & (d <= n_back + 0.5)]
    if not len(near):
        return None
    far = near[np.argmax(np.hypot(*(near - e).T))]
    u = e - far
    n = np.hypot(*u)
    return u / n if n > 0 else None


def foot_point(mask, sk, e):
    u = outward_dir(sk, e)
    if u is None:
        return e.astype(float)
    pts = np.argwhere(mask)
    d = np.hypot(*(pts - e).T)
    cand = pts[d <= FOOT_SEARCH_PX]
    proj = (cand - e) @ u
    return cand[np.argmax(proj)].astype(float)


def main(cine, out, T, T_low=None, background=None, low_frac=None):
    src = cf.open_cine(cine, background=background)
    starts = np.linspace(0, src.n - WIN, WINDOWS).astype(int)
    feetL, feetR, epL, epR = [], [], [], []
    n_det = n_arch = 0
    for a in starts:
        blk = src.block(int(a), WIN)
        for f in blk:
            b = binarize(f, CLOSE_R, dark_guard=T, bright_thresh=T, min_arc_px=MIN_PX, max_arc_px=MAX_PX,
                         low_thresh=T_low, low_frac=low_frac)
            if not b.any():
                continue
            n_det += 1
            sk = skeletonize(b)
            e = skeleton_endpoints(sk)
            if len(e) != 2:
                continue
            top = np.nonzero(b.any(1))[0][0]
            if min(e[:, 0]) < top + ARCH_MIN_RISE_PX:
                continue
            e = e[np.argsort(e[:, 1])]
            n_arch += 1
            feetL.append(foot_point(b, sk, e[0]))
            feetR.append(foot_point(b, sk, e[1]))
            epL.append(e[0]); epR.append(e[1])
    L, R = np.array(feetL), np.array(feetR)
    res = dict(cine=cine, T=T, T_low=T_low, low_frac=low_frac, background=background, frames_scanned=int(WINDOWS * WIN), detected=n_det, clean_arches=n_arch)
    if n_arch < 50:
        res["status"] = "FAIL: fewer than 50 clean arches"
        Path(out, Path(cine).stem + "_gap.json").write_text(json.dumps(res, indent=1))
        return
    rng = np.random.default_rng(SEED)
    fits = {}
    for name, P in (("L", L), ("R", R)):
        model, inl = ransac(P, LineModelND, min_samples=2, residual_threshold=RANSAC_RESID_PX,
                            max_trials=2000, rng=rng)
        o, d = model.params
        d = d if d[0] > 0 else -d               # direction pointing DOWN (toward the apex, larger row)
        s = (P[inl] - o) @ d
        s_tip = np.quantile(s, 1 - TIP_QUANTILE)  # lowest attachment reached along the blade
        tip = o + s_tip * d
        fits[name] = dict(origin=o.tolist(), dir=d.tolist(), inliers=int(inl.sum()), n=int(len(P)),
                          tip=tip.tolist(), tip_q_spread_px=float(np.quantile(s, 1 - TIP_QUANTILE / 2) -
                                                                  np.quantile(s, 1 - 2 * TIP_QUANTILE)))
    dL, dR = np.array(fits["L"]["dir"]), np.array(fits["R"]["dir"])
    full_angle = float(np.degrees(np.arccos(np.clip(dL @ dR, -1, 1))))
    tipL, tipR = np.array(fits["L"]["tip"]), np.array(fits["R"]["tip"])
    gap_px = float(np.hypot(*(tipL - tipR)))
    res.update(fits=fits, full_opening_angle_deg=full_angle,
               half_angle_from_vertical_L=float(np.degrees(np.arctan2(abs(dL[1]), dL[0]))),
               half_angle_from_vertical_R=float(np.degrees(np.arctan2(abs(dR[1]), dR[0]))),
               gap_px=gap_px, px_per_mm=gap_px / GAP_MM, tip_row_max=float(max(tipL[0], tipR[0])),
               status="OK")
    # electrode-edge clouds for analyze_blade --electrode-edges: skeleton endpoints (the quantity
    # analyze_blade tests), pixels hit often enough to be an attachment site, not a stray foot
    thr = max(3, int(np.ceil(EDGE_MIN_HITS_FRAC * n_arch)))
    clouds = {}
    for nm, E in (("A", np.array(epL)), ("B", np.array(epR))):
        u, c = np.unique(E.astype(int), axis=0, return_counts=True)
        clouds[nm] = u[c >= thr]
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out, Path(cine).stem + "_edges.npz"), A=clouds["A"].astype(float), B=clouds["B"].astype(float),
             A_tip=tipL, B_tip=tipR)
    res.update(edge_min_hits=thr, edge_A_px=int(len(clouds["A"])), edge_B_px=int(len(clouds["B"])))
    Path(out, Path(cine).stem + "_gap.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(clouds["A"][:, 1], clouds["A"][:, 0], s=4, c="b", marker="s")
    ax.scatter(clouds["B"][:, 1], clouds["B"][:, 0], s=4, c="m", marker="s")
    ax.scatter(L[:, 1], L[:, 0], s=1, c="c", alpha=0.3)
    ax.scatter(R[:, 1], R[:, 0], s=1, c="orange", alpha=0.3)
    for nm, c in (("L", "c"), ("R", "orange")):
        o, d = np.array(fits[nm]["origin"]), np.array(fits[nm]["dir"])
        tt = np.linspace(-80, 80, 2)
        ax.plot(o[1] + tt * d[1], o[0] + tt * d[0], c=c, lw=1)
        ax.plot(fits[nm]["tip"][1], fits[nm]["tip"][0], "r+", ms=12)
    ax.set_xlim(40, 200); ax.set_ylim(256, 120); ax.set_aspect("equal")
    ax.set_title(f"gap {gap_px:.1f} px  angle {full_angle:.1f}")
    fig.savefig(Path(out, Path(cine).stem + "_gap.png"), dpi=90)
    print(json.dumps({k: res[k] for k in ("clean_arches", "gap_px", "px_per_mm", "full_opening_angle_deg")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else None,
         sys.argv[5] if len(sys.argv) > 5 else None)
