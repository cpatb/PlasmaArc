"""Post-stage for one finished run: apply the validated flash cut and re-run the CSV-only stages with the
PHYSICAL gap row (qa_restrike otherwise anchors the apex on a circular p5 of the tip trajectory).

usage: post_stage.py <run_dir> --eff-width W [--arc-width A]
- measurements.csv is kept as measurements_raw.csv (first call only) and rewritten with is_flash from
  arc_filter.add_flash_flag(eff_width=W, arc_width=A) computed on the raw columns;
- event_velocity.py and qa_restrike.py are re-run into the same run dir;
- POST.json records the cut, the gap row and N before/after.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import arc_filter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--eff-width", type=float, required=True)
    ap.add_argument("--arc-width", type=float, default=0.0)
    a = ap.parse_args()
    d = Path(a.run_dir)
    m = json.loads((d / "run_metadata.json").read_text())
    raw = d / "measurements_raw.csv"
    if not raw.exists():
        (d / "measurements.csv").replace(raw)
    df = pd.read_csv(raw)
    n0 = len(df)
    df = df.drop(columns=["is_flash"])
    df = arc_filter.add_flash_flag(df, eff_width=a.eff_width, arc_width=a.arc_width or None)
    df["is_flash"] = df["is_flash"].astype(int)
    df = df.drop(columns=["eff_width"])
    assert len(df) == n0 == m["n_frames"]
    df.to_csv(d / "measurements.csv", index=False)
    cal = m["calibration"]
    gap_row = int(round(0.5 * (cal["fits"]["L"]["tip"][0] + cal["fits"]["R"]["tip"][0])))
    pxmm, fps = cal["px_per_mm"], m["fps"]
    comp = df["arc_complete"] == 1
    post = dict(eff_width_cut=a.eff_width, arc_width_cut=a.arc_width, gap_row_raw=gap_row, n_frames=n0,
                complete=int(comp.sum()), complete_flash=int((comp & (df["is_flash"] == 1)).sum()))
    log = d / "post_stage.log"
    py = [sys.executable, "-X", "utf8"]
    with open(log, "w", encoding="utf-8") as f:
        for cmd in ([HERE / "event_velocity.py", d / "measurements.csv", d, "--px-per-mm", f"{pxmm:.5f}",
                     "--fps", f"{fps:.0f}", "--local-vel-n", "10"],
                    [HERE / "qa_restrike.py", "--csv", d / "measurements_all_events.csv",
                     "--arc-direction", "up", "--frame-height", str(m["H"]), "--gap-row", str(gap_row),
                     "--px-per-mm", f"{pxmm:.5f}", "--fps", f"{fps:.0f}", "--method", "abs",
                     "--threshold", "0.10", "--ext-prominence-frac", "0.10", "--out", d / "qa_restrike.png"]):
            f.write(">>> " + " ".join(map(str, cmd)) + "\n")
            f.flush()
            r = subprocess.run(py + [str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
            if r.returncode:
                post["status"] = f"FAIL {Path(str(cmd[0])).name} rc={r.returncode}"
                (d / "POST.json").write_text(json.dumps(post, indent=1))
                sys.exit(r.returncode)
    post["status"] = "DONE"
    (d / "POST.json").write_text(json.dumps(post, indent=1))
    print(d.name, post)


if __name__ == "__main__":
    main()
