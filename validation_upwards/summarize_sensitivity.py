"""Detector-sensitivity table: every arm vs the nominal arm, same recording, same first 20 000 frames,
same calibration/background.  Relative change (arm - nominal) / nominal for each headline metric.

usage: summarize_sensitivity.py <sensitivity_root> <out_csv>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def metrics(d: Path):
    a = json.loads((d / "ARM_DONE.json").read_text())
    df = pd.read_csv(d / "measurements.csv", usecols=["arc_complete", "arc_length_px", "arc_tip_row"])
    c = df[df["arc_complete"] == 1]
    nm = json.loads((Path(a["nominal"]) / "run_metadata.json").read_text())
    gap_from_bottom = (nm["H"] - 1) - nm["calibration"]["gap_row"]   # arc_tip_row ('up') is from the bottom edge
    r = dict(frac_complete=float((df["arc_complete"] == 1).mean()),
             arc_len_px_med=float(np.median(c["arc_length_px"])) if len(c) else np.nan,
             tip_height_px_med=float(np.median(c["arc_tip_row"] - gap_from_bottom)) if len(c) else np.nan)
    ev = pd.read_csv(d / "event_summary.csv") if (d / "event_summary.csv").exists() else pd.DataFrame()
    r.update(n_events=len(ev), event_vel_med=float(np.nanmedian(ev["vel_ms"])) if len(ev) else np.nan,
             local_vel_med=float(np.nanmedian(ev["local_vel_median"])) if len(ev) and "local_vel_median" in ev else np.nan)
    rs = pd.read_csv(d / "re_strikes_qa.csv") if (d / "re_strikes_qa.csv").exists() else pd.DataFrame()
    cyc = rs["cycle_frames"][rs["cycle_frames"] > 0] if len(rs) else pd.Series(dtype=float)
    r.update(n_restrikes=len(rs), frac_modeB=float((rs["mode"] == "B").mean()) if len(rs) else np.nan,
             cycle_frames_mean=float(cyc.mean()) if len(cyc) else np.nan)
    r.update(T_low=a["T_low"], T_high=a["T_high"], low_frac=a["low_frac"])
    return r


rows = []
for done in ROOT.rglob("ARM_DONE.json"):
    d = done.parent
    arm = d.relative_to(ROOT).parts[0]
    rows.append(dict(arm=arm, stem=d.name, gas=d.relative_to(ROOT).parts[1], **metrics(d)))
A = pd.DataFrame(rows)
if A.empty:
    sys.exit("no finished arms")
keys = ["frac_complete", "arc_len_px_med", "tip_height_px_med", "n_events", "event_vel_med", "local_vel_med",
        "n_restrikes", "frac_modeB", "cycle_frames_mean"]
nom = A[A["arm"] == "nominal"].set_index("stem")
out = []
for _, r in A[A["arm"] != "nominal"].iterrows():
    if r["stem"] not in nom.index:
        continue
    n = nom.loc[r["stem"]]
    row = dict(stem=r["stem"], gas=r["gas"], arm=r["arm"])
    for k in keys:
        row[k + "_nom"] = n[k]
        row[k + "_arm"] = r[k]
        row[k + "_rel"] = (r[k] - n[k]) / n[k] if n[k] not in (0, np.nan) and np.isfinite(n[k]) and n[k] != 0 else np.nan
    out.append(row)
S = pd.DataFrame(out)
S.to_csv(OUT, index=False)
A.to_csv(Path(OUT).with_name("sensitivity_raw.csv"), index=False)
show = ["stem", "arm"] + [k + "_rel" for k in keys]
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(S[show].round(3).to_string(index=False))
    print("\nmax |relative change| over arms, per metric and file:")
    print(S.groupby("stem")[[k + "_rel" for k in keys]].agg(lambda x: np.nanmax(np.abs(x))).round(3).to_string())
