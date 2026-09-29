"""Print the hysteresis sweep grids and the pre-registered V-H1 / V-H2 / V-H4 verdicts."""
import glob
import json
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else r"E:/BladeReactor_Upwards_analysis/validation/hysteresis_sweep")
TOL = 0.10          # V-H1: < 10 % change between adjacent grid points
SPEED_P99_MAX = 10  # V-H4: px/frame
SMEAR_MARGIN = 1.2  # V-H2


def rel(a, b):
    if a is None or b is None or a == 0:
        return float("inf")
    return abs(b - a) / abs(a)


rows = {}
for f in sorted(glob.glob(str(D / "*_hyst.json"))):
    d = json.load(open(f))
    name = Path(f).stem.replace("BLADE_ADCB_", "").replace("_06mmNozzle_trial1_hyst", "")
    print(f"\n{name}  N={d['N']:.0f}  smear S={d['smear_S']:.0f}  sat frames {d['n_sat_frames_of_2000']}/2000  "
          f"fmax median {d['fmax_median']:.0f}")
    g = {(r["a"], r["b"]): r for r in d["grid"]}
    A = sorted({k[0] for k in g}); B = sorted({k[1] for k in g})
    print("   b\\a " + "".join(f"{a:>22}" for a in A))
    for b in B:
        cells = []
        for a in A:
            r = g.get((a, b))
            cells.append(" " * 22 if r is None else
                         f"{r['det_frac']:.2f}/{r['len_med']}/{r['top_speed_mean'] and round(r['top_speed_mean'], 1)}/{r['top_speed_p99']}".rjust(22))
        print(f"  {b:>3} " + "".join(cells))
    ok = {}
    for (a, b), r in g.items():
        checks = []
        ia, ib = A.index(a), B.index(b)
        for nb in ([(A[ia - 1], b)] if ia > 0 else []) + ([(A[ia + 1], b)] if ia + 1 < len(A) else []) + \
                  ([(a, B[ib - 1])] if ib > 0 else []) + ([(a, B[ib + 1])] if ib + 1 < len(B) else []):
            if nb in g:
                checks.append(rel(r["len_med"], g[nb]["len_med"]) < TOL and
                              rel(r["top_speed_mean"], g[nb]["top_speed_mean"]) < TOL)
        vh1 = bool(checks) and all(checks)
        vh2 = r["T_low"] > SMEAR_MARGIN * d["smear_S"] if d["smear_S"] == d["smear_S"] else True
        vh4 = r["top_speed_p99"] is not None and r["top_speed_p99"] < SPEED_P99_MAX
        ok[(a, b)] = (vh1, vh2, vh4)
    rows[name] = ok
    print("   pass (VH1,VH2,VH4): " + "  ".join(f"{k}:{''.join('Y' if x else 'n' for x in v)}" for k, v in sorted(ok.items())))

if rows:
    keys = set.intersection(*[set(v) for v in rows.values()])
    print("\nfiles passing ALL of VH1,VH2,VH4 per (a,b):")
    for k in sorted(keys):
        n = sum(all(rows[f][k]) for f in rows)
        print(f"  {k}: {n}/{len(rows)}")
