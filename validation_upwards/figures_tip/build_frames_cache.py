"""Per-frame table of the quotable leading-tip quantities for every analysed BLADE-upwards recording (one parquet),
so the distribution / profile / stationarity figures read one file instead of 113 CSVs.

Rows: complete, non-flash frames only (the arc_filter.valid_frames criterion used by aggregate_upwards.py).
Columns: run, gas, Re, trial, frame, t_frac (frame / n_frames), h_mm (leading-tip height above the gap, hand scale),
L_mm (skeleton arc length), v_ms (local axial tip velocity from event_velocity.py; NaN off the advancing front),
speed_ms (2-D local tip speed).

Checks printed and asserted: frames per run == n_frames of the recording; per-run medians reproduce the aggregate
table (arc length, tip height p90) to 1e-9.

usage: python build_frames_cache.py
"""
import json

import numpy as np
import pandas as pd

import _common as C


def main():
    F = C.per_run()
    rows, n_in = [], 0
    for _, r in F.iterrows():
        rd = next((C.ROOT / "runs_annotated").rglob(f"{r['stem']}/run_metadata.json")).parent
        m = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
        pxmm, H, gap_row = float(m["calibration"]["px_per_mm"]), int(m["H"]), float(m["calibration"]["gap_row"])
        if not np.isclose(pxmm, r["px_per_mm"], rtol=0, atol=1e-12):
            raise SystemExit(f"{r['stem']}: px/mm {pxmm} != aggregate {r['px_per_mm']}")
        d = pd.read_csv(rd / "measurements_all_events.csv",
                        usecols=["frame", "arc_complete", "is_flash", "arc_length_px", "arc_tip_row", "local_vel_ms",
                                 "local_speed_ms"])
        if len(d) != int(m["n_frames"]):
            raise SystemExit(f"{r['stem']}: {len(d)} rows != n_frames {m['n_frames']}")
        n_in += len(d)
        c = d[(d["arc_complete"] == 1) & (d["is_flash"] == 0)]
        h = (c["arc_tip_row"] - ((H - 1) - gap_row)) / pxmm
        L = c["arc_length_px"] / pxmm
        # anchor: the aggregate table's medians, computed by aggregate_upwards.py from measurements.csv
        if not (np.isclose(np.median(L), r["arc_len_mm_med"], rtol=0, atol=1e-9)
                and np.isclose(np.percentile(h, 90), r["tip_height_above_gap_mm_p90"], rtol=0, atol=1e-9)):
            raise SystemExit(f"{r['stem']}: per-frame medians do not reproduce the aggregate table")
        rows.append(pd.DataFrame({"run": r["stem"], "gas": r["gas"], "Re": int(r["Re"]), "trial": int(r["trial"]),
                                  "frame": c["frame"].to_numpy(), "t_frac": c["frame"].to_numpy() / int(m["n_frames"]),
                                  "h_mm": h.to_numpy(), "L_mm": L.to_numpy(), "v_ms": c["local_vel_ms"].to_numpy(),
                                  "speed_ms": c["local_speed_ms"].to_numpy()}))
    D = pd.concat(rows, ignore_index=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(C.CACHE, index=False)
    print(f"{len(F)} runs, {n_in} frames read, {len(D)} complete non-flash frames kept, "
          f"{int(D['v_ms'].notna().sum())} with a local velocity; aggregate medians reproduced on every run")
    print(f"wrote {C.CACHE}")


if __name__ == "__main__":
    main()
