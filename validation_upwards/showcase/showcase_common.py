"""Shared inputs for the single-recording showcase (videos + per-recording kinematics plots).

One recording, read exactly as the annotated pipeline read it (run_blade_upwards.py --annotations):
background-subtracted, defect-corrected 12-bit DN, the run's own T_low/T_high/low_frac, close radius, speckle
floor, hand-annotated electrode edges and attachment tolerance. Every threshold and scale is read from the run's
run_metadata.json / chunk_params.json; nothing is typed here except the recording name and the output folder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "validation_upwards" / "figures_tip"))

STEM = "BLADE_ADCB_Argon_Re8594_06mmNozzle_trial1"
CINE = Path(r"E:\BladeReactor_Upwards\Argon\45degree_2.4mmGap\Re8594") / f"{STEM}.cine"
RUN = Path(r"E:\BladeReactor_Upwards_analysis\runs_annotated\Argon\45degree_2.4mmGap\Re8594") / STEM
THR = Path(r"E:\BladeReactor_Upwards_analysis\tip_restrike_check\threshold")
OUT = CINE.parent / f"{STEM}_showcase"           # author: everything in a folder next to the recording

WIN = 2400            # source frames in the clip (12 ms at 200 kfps)
STRIDE = 2            # every 2nd source frame -> 1200 video frames
FPS_OUT = 60          # 1200 / 60 = 20 s of video
SEARCH_STEP = 100     # window-start grid for the selection rule


def meta():
    m = json.loads((RUN / "run_metadata.json").read_text(encoding="utf-8"))
    if m.get("status") != "DONE":
        raise SystemExit(f"run status {m.get('status')}")
    return m


def mode_valley_mm():
    """The Mode A/B valley of the re-strike height, as printed by tip_modes.py (not typed)."""
    for line in (THR / "tip_modes.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("valley value used:"):
            return float(line.split(":")[1].split()[0])
    raise SystemExit("valley not found in tip_modes.txt")


def t_tip_mm():
    return float((THR / "T_TIP.txt").read_text().strip())


def frames_table(m):
    """Every frame of the recording with the leading-tip quantities of the figure suite (build_frames_cache.py):
    h_mm = leading-tip height above the hand gap row, L_mm = skeleton arc length, v_ms = local axial tip velocity.
    `valid` = complete and not flash, the frames the suite uses."""
    d = pd.read_csv(RUN / "measurements_all_events.csv")
    if len(d) != int(m["n_frames"]):
        raise SystemExit(f"{len(d)} rows != n_frames {m['n_frames']}")
    H, pxmm, gap_row = int(m["H"]), float(m["calibration"]["px_per_mm"]), float(m["calibration"]["gap_row"])
    d["valid"] = (d["arc_complete"] == 1) & (d["is_flash"] == 0)
    d["h_mm"] = np.where(d["valid"], (d["arc_tip_row"] - ((H - 1) - gap_row)) / pxmm, np.nan)
    d["L_mm"] = np.where(d["valid"], d["arc_length_px"] / pxmm, np.nan)
    d["v_ms"] = np.where(d["valid"], d["local_vel_ms"], np.nan)
    d["t_ms"] = d["frame"] / float(m["fps"]) * 1e3
    return d


def restrikes():
    """Kept leading-tip re-strikes of this recording (fall >= T_TIP), with Mode from the valley split."""
    E = pd.read_csv(THR / "tip_restrike_events_kept.csv")
    n_all = int((E["run"] == STEM).sum())
    E = E[(E["run"] == STEM) & E["kept"]].copy()
    E["mode"] = np.where(E["tip_after_mm"] > mode_valley_mm(), "B", "A")
    print(f"tip re-strikes of this run: {n_all}, kept (fall >= {t_tip_mm()} mm) {len(E)} "
          f"(A {int((E['mode'] == 'A').sum())}, B {int((E['mode'] == 'B').sum())})")
    return E.reset_index(drop=True)


def pick_window(d, E):
    """Selection rule (stated, not tuned to a picture): among WIN-frame windows starting on a SEARCH_STEP grid,
    the one with the most kept re-strikes of the RARER mode, then the most kept re-strikes in total, then the
    earliest. So the clip shows both re-strike modes as often as the recording allows."""
    best = None
    for s in range(0, len(d) - WIN + 1, SEARCH_STEP):
        e = E[(E["frame"] >= s) & (E["frame"] < s + WIN)]
        nA, nB = int((e["mode"] == "A").sum()), int((e["mode"] == "B").sum())
        key = (min(nA, nB), nA + nB, -s)
        if best is None or key > best[0]:
            best = (key, s, nA, nB)
    _, s, nA, nB = best
    print(f"window: frames [{s}, {s + WIN}), kept re-strikes A {nA} / B {nB}")
    return s
