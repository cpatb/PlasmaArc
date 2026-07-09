"""
regen_hand_calib.py — regenerate ALL J-electrode outputs + figures on the
hand-authoritative calibration. Pure-Python runner (no PowerShell execution-policy
issues). Calibration was already injected into j_geometry.json + run_metadata.json
by apply_hand_calibration.py (backups in each <results>/__hand_calib_backup__/).

WHY --from-stage merge (not analyze): px/mm enters only as a final scale
(vel_ms = slope*FPS/px_per_mm/1000); detection is calibration-free. Restarting at
merge reuses existing per-chunk detections and reproduces a full re-detect exactly,
skipping the hours-long convert+detect (esp. the 873k-frame 1.2 mm file).

    python regen_hand_calib.py            # run everything
    python regen_hand_calib.py --dry-run  # just print the commands
"""
import subprocess, sys, argparse
from pathlib import Path

REPO = Path(r"C:\Users\compl\Github\Plasma Arc")
ROOT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
PY = sys.executable  # same interpreter that launched this script

# recording -> original bright_thresh (not stored in j_geometry.json)
REC = [
    ("HS_0.4mm_1000mL_(CO2)_trig.cine",                 160),
    ("HS_0.4mm_1500mL_(CO2)_trig.cine",                 220),
    ("HS_0.4mm_1500mL_(CO2)_trig_repeat.cine",          235),
    ("HS_0.8mm_1500mL_(CO2)_trig(2).cine",              160),
    ("HS_0.8mm_1500mL_(CO2)_trig_repeat_weird(2).cine", 210),
    ("HS_1.2mm_1500mL_(CO2).cine",                      175),
]

def cmds():
    out = []
    # 1. per-recording: regenerate CSVs + figures from existing detections
    for cine, bt in REC:
        out.append([PY, "run_j_co2.py", "--cine", cine,
                    "--from-stage", "merge", "--bright-thresh", str(bt)])
    # 2. per-recording uncertainty columns
    out.append([PY, "recompute_uncertainty.py", "--root", str(ROOT)])
    # 3. cross-recording comparison figures (repeat sentinel still honored)
    out.append([PY, str(ROOT / "Reynolds_comparison" / "reynolds_replot.py")])
    out.append([PY, "mode_ratio_vs_re.py",  "--root", str(ROOT)])
    out.append([PY, "asymmetry_vs_time.py", "--root", str(ROOT)])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for c in cmds():
        print("\n>>>", " ".join(f'"{x}"' if " " in x else x for x in c), flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(c, cwd=str(REPO))
        if r.returncode != 0:
            sys.exit(f"[FAILED rc={r.returncode}] {' '.join(c)}")
    print("\nDONE." if not args.dry_run else "\n(dry run — nothing executed)")

if __name__ == "__main__":
    main()
