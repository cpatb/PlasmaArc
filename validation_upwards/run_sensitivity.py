"""Detector sensitivity arms (DECISION 23:48 + AMENDMENT 4, OVERNIGHT_LOG).

For each nominal run dir (finished by run_blade_upwards.py), re-run analyze_blade on the first
N_FRAMES frames with ONE detector knob changed, REUSING the nominal background, noise/smear floor and
calibration (edge clouds, px/mm) so the arm isolates the detector. The nominal is re-analysed on the
same frames from its own merged CSV (no re-detection needed: rows [0, N_FRAMES)). Both then go through
event_velocity + qa_restrike with the physical gap row, into <arm_dir>.

usage: run_sensitivity.py <list of nominal run dirs file> [--workers 6] [--files-parallel 2]
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_blade_upwards as rb  # noqa: E402  (constants only; nothing is modified)

N_FRAMES = 20_000
ARMS = {  # name: (low_scale, high_scale, low_frac)
    "nominal": (1.0, 1.0, rb.LOW_FRAC),
    "frac0.03": (1.0, 1.0, 0.03),
    "frac0.10": (1.0, 1.0, 0.10),
    "floor1.5": (1.5, 1.0, rb.LOW_FRAC),
    "seed0.5": (1.0, 0.5, rb.LOW_FRAC),
}
OUT = rb.OUT_ROOT / "sensitivity"


def run(cmd, log):
    with open(log, "a", encoding="utf-8") as f:
        f.write(">>> " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([sys.executable, "-X", "utf8"] + [str(c) for c in cmd], stdout=f,
                           stderr=subprocess.STDOUT, cwd=str(HERE))
    if r.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} rc={r.returncode} ({log})")


def one_arm(nom: Path, arm: str, workers: int):
    m = json.loads((nom / "run_metadata.json").read_text())
    cine = Path(m["cine"])
    od = OUT / arm / nom.relative_to(rb.OUT_ROOT / "runs")
    od.mkdir(parents=True, exist_ok=True)
    if (od / "ARM_DONE.json").exists():
        return
    ls, hs, fr = ARMS[arm]
    det, cal = m["detector"], m["calibration"]
    floor = max(det["N"], det["smear_S"])
    T_low, T_high = rb.LOW_K * floor * ls, rb.HIGH_K * rb.LOW_K * floor * hs
    pxmm, fps = cal["px_per_mm"], m["fps"]
    edges = nom / "calibration" / (nom.name + "_edges.npz")
    bgp = nom / "background_b10.npy"
    log = od / "arm.log"
    mcsv = od / "measurements.csv"
    if arm == "nominal":
        src = nom / ("measurements_raw.csv" if (nom / "measurements_raw.csv").exists() else "measurements.csv")
        df = pd.read_csv(src, nrows=N_FRAMES)
    else:
        starts = list(range(0, N_FRAMES, rb.CHUNK))

        def chunk(k):
            cd = od / "chunks" / f"c{k:02d}"
            if (cd / "measurements.csv").exists():
                return
            run([HERE / "analyze_blade.py", cine, "--out-dir", cd, "--fps", f"{fps:.0f}",
                 "--frame-start", starts[k], "--frame-count", min(rb.CHUNK, N_FRAMES - starts[k]),
                 "--px-per-mm", f"{pxmm:.5f}", "--background", bgp, "--low-thresh", f"{T_low:.3f}",
                 "--low-frac", fr, "--bright-thresh", f"{T_high:.3f}", "--dark-guard", f"{T_high:.3f}",
                 "--min-arc-px", rb.MIN_ARC_PX, "--max-arc-px", 10**9, "--close-radius", rb.CLOSE_R,
                 "--flash-eff-width", 0, "--flash-arc-width", 0, "--electrode-edges", edges,
                 "--elec-tol-px", rb.ELEC_TOL_PX, "--arc-direction", "up", "--qa-interval", 5000], log)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(chunk, range(len(starts))))
        parts = []
        for k, a in enumerate(starts):
            d = pd.read_csv(od / "chunks" / f"c{k:02d}" / "measurements.csv")
            d["frame"] += a
            d["time_s"] = d["frame"] / fps
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        df["file"] = cine.stem
    assert len(df) == N_FRAMES, (arm, nom, len(df))
    df.to_csv(mcsv, index=False)
    gap_row = int(round(0.5 * (cal["fits"]["L"]["tip"][0] + cal["fits"]["R"]["tip"][0])))
    run([HERE / "event_velocity.py", mcsv, od, "--px-per-mm", f"{pxmm:.5f}", "--fps", f"{fps:.0f}",
         "--local-vel-n", 10], log)
    run([HERE / "qa_restrike.py", "--csv", od / "measurements_all_events.csv", "--arc-direction", "up",
         "--frame-height", m["H"], "--gap-row", gap_row, "--px-per-mm", f"{pxmm:.5f}", "--fps", f"{fps:.0f}",
         "--method", "abs", "--threshold", "0.10", "--ext-prominence-frac", "0.10",
         "--out", od / "qa_restrike.png"], log)
    (od / "ARM_DONE.json").write_text(json.dumps(dict(arm=arm, nominal=str(nom), T_low=T_low, T_high=T_high,
                                                      low_frac=fr, n_frames=N_FRAMES, gap_row=gap_row), indent=1))
    print(f"[arm done] {arm} {nom.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list_file")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--files-parallel", type=int, default=2)
    a = ap.parse_args()
    noms = [Path(x.strip()) for x in Path(a.list_file).read_text().splitlines() if x.strip()]
    jobs = [(n, arm) for n in noms for arm in ARMS]

    def job(t):
        try:
            one_arm(t[0], t[1], a.workers)
        except Exception as e:
            print(f"[arm FAIL] {t[1]} {t[0].name}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=a.files_parallel) as ex:
        list(ex.map(job, jobs))
    print("arms finished", flush=True)


if __name__ == "__main__":
    main()
