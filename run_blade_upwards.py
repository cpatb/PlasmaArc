#!/usr/bin/env python3
"""
run_blade_upwards.py -- BLADE pipeline straight from .cine (no TIFF), for the upward
45-degree ADCB campaign on E:\\BladeReactor_Upwards (a NEW configuration: nothing calibrated
on the older rigs is reused; see E:\\BladeReactor_Upwards_analysis\\OVERNIGHT_LOG.md).

Per recording (every number written to <out>/run_metadata.json):
  0. cine_frames: header frame count, fps; fixed defect map from 1200 frames spread over the
     record. A defect map that is not the known 128-px lattice STOPS the file.
  1. cine_frames.measure_background (AMENDMENT 3): per-pixel temporal 10th percentile B10 over
     2000 frames (subtracted from every frame), residual noise ceiling N on arc-free pixels, and
     readout-smear ceiling S right of saturated pixels.  Hysteresis thresholds on the residual:
        T_low  = LOW_K  * max(N, S)          (just above both measured artefacts, V-H2)
        T_high = HIGH_K * T_low              (seed: unambiguously arc)
     multiplied by --low-scale / --high-scale in the sensitivity arms.
  2. validation_upwards/gap_calibration: RANSAC blade loci from arc attachment, blade tips,
     gap_px -> px/mm = gap_px / 2.4 mm (folder name), and the electrode-edge clouds used by
     analyze_blade's attachment test. Calibration failure STOPS the file.
  3. analyze_blade.py on 10 000-frame windows in parallel; inherited px filters OFF
     (max_arc_px, flash widths, gap row: tuned on another rig; the flash cut is applied later
     from the recorded widths); merged to one CSV with global frames, N checked == header.
  4. event_velocity, qa_restrike, fourier_analysis (band opened to FREQ_MAX_HZ),
     build_velocity_map, visualize_blade.

Usage:
    python run_blade_upwards.py --list
    python run_blade_upwards.py --match Argon_Re2250_06mmNozzle_trial1
    python run_blade_upwards.py --all --files-parallel 2 --workers 7
    python run_blade_upwards.py --list-file sweep_files.txt --low-scale 0.8   (sensitivity arm)
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import cine_frames as cf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "validation_upwards"))
import gap_calibration  # noqa: E402

DATA_ROOT = Path(r"E:\BladeReactor_Upwards")
OUT_ROOT = Path(r"E:\BladeReactor_Upwards_analysis")

GAP_MM = 2.4                    # folder name "45degree_2.4mmGap": minimum electrode gap (trusted)
ARC_DIRECTION = "up"            # upward rig; pycine orientation has electrodes at the bottom
CHUNK = 10_000                  # frames per analysis window
LOW_K, HIGH_K = 1.2, 4.0        # DECISION 23:48 / AMENDMENT 3 (OVERNIGHT_LOG)
LOW_FRAC = 0.05                 # AMENDMENT 4: low threshold >= 5 % of the frame peak (Argon glow, V-H3)
MIN_ARC_PX = 22                 # speckle floor (px^2); tiny next to any arc mask (~100-900 px^2)
ELEC_TOL_PX = 4                 # attachment tolerance to the edge clouds (validated visually)
CLOSE_R = 3
FREQ_MAX_HZ = 20_000            # fourier: do not discard the kHz band (default 500 Hz)
LOCAL_VEL_N = 10
GAP_PX_PLAUSIBLE = (12.0, 45.0)  # outside this the calibration is reported as failed
ANGLE_TOL_DEG = 5.0             # full opening vs 2 x folder angle: flag beyond this
KNOWN_LATTICE = (list(range(1, 122, 8)), list(range(11, 124, 16)))
# independent cross-check only (hand-marked on Argon Re2250 t1 by the collaborator's LeadArc tool)
ALEX_GAP_PX = 33.1119129345809


def parse_condition(p: Path):
    s = p.stem
    rel = p.relative_to(DATA_ROOT).parts
    re_m = re.search(r"Re(\d+)", s)
    tr = re.search(r"trial(\d+)", s)
    ang = re.search(r"(\d+)degree", str(p))
    gap = re.search(r"([\d.]+)mmGap", str(p))
    return dict(gas=rel[0], Re=int(re_m.group(1)) if re_m else None,
                trial=int(tr.group(1)) if tr else None,
                angle_deg=int(ang.group(1)) if ang else None,
                gap_mm_folder=float(gap.group(1)) if gap else None)


def out_dir_for(p: Path, tag: str = "") -> Path:
    base = OUT_ROOT / ("runs" if not tag else f"sens_{tag}") / p.relative_to(DATA_ROOT).parent
    return base / p.stem


def run(cmd, log):
    t = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n>>> " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        r = subprocess.run([sys.executable, "-X", "utf8"] + [str(c) for c in cmd],
                           stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {Path(str(cmd[0])).name} (see {log})")
    return time.time() - t


def process(p: Path, workers: int, force=False, low_scale=1.0, high_scale=1.0, tag="", figures=False,
            low_frac=LOW_FRAC):
    od = out_dir_for(p, tag)
    od.mkdir(parents=True, exist_ok=True)
    log = od / "pipeline.log"
    done = od / "DONE.json"
    if done.exists() and not force:
        print(f"[skip] {p.name} (DONE)", flush=True)
        return json.loads(done.read_text())
    t0 = time.time()
    meta = dict(cine=str(p), out=str(od), tag=tag, **parse_condition(p),
                started=time.strftime("%Y-%m-%d %H:%M:%S"),
                git=subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                                   text=True).stdout.strip(),
                git_dirty=bool(subprocess.run(["git", "status", "--porcelain"], cwd=HERE,
                                              capture_output=True, text=True).stdout.strip()),
                low_scale=low_scale, high_scale=high_scale, low_frac=low_frac)
    mpath = od / "run_metadata.json"

    def save():
        mpath.write_text(json.dumps(meta, indent=1, default=float))

    # 0. source + defects
    src = cf.open_cine(str(p))
    rows, cols = src.defects.lattice_signature()
    meta.update(n_frames=src.n, fps=src.fps, W=src.W, H=src.H, shutter_ns=src.shutter_ns,
                real_bpp=src.real_bpp, camera_serial=src.serial,
                defects=dict(n=src.defects.n, values=src.defects.values,
                             frames_sampled=src.defects.n_frames_sampled,
                             known_lattice=(rows, cols) == KNOWN_LATTICE and src.defects.n == 128))
    save()
    if not meta["defects"]["known_lattice"]:
        meta["status"] = "STOPPED: defect map is not the known 128-px lattice"
        save()
        raise RuntimeError(meta["status"])

    # 1. background, noise, smear -> thresholds
    bgp = od / "background_b10.npy"
    bg, N, info = cf.measure_background(src)
    np.save(bgp, bg)
    floor = max(N, info["smear_S"])
    T_low = LOW_K * floor * low_scale
    T_high = HIGH_K * LOW_K * floor * high_scale
    if T_high <= T_low:
        raise RuntimeError(f"T_high {T_high} <= T_low {T_low}")
    meta["detector"] = dict(kind="hysteresis on background-subtracted 12-bit DN", N=N, **info,
                            T_low=T_low, T_high=T_high, LOW_K=LOW_K, HIGH_K=HIGH_K,
                            close_r=CLOSE_R, min_arc_px=MIN_ARC_PX)
    save()

    # 2. calibration + edge clouds (same detector)
    cal_dir = od / "calibration"
    gap_calibration.main(str(p), str(cal_dir), T_high, T_low, str(bgp), low_frac)
    cal = json.loads((cal_dir / (p.stem + "_gap.json")).read_text())
    meta["calibration"] = cal
    if cal.get("status") != "OK" or not (GAP_PX_PLAUSIBLE[0] <= cal["gap_px"] <= GAP_PX_PLAUSIBLE[1]):
        meta["status"] = f"STOPPED: calibration {cal.get('status')} gap_px {cal.get('gap_px')}"
        save()
        raise RuntimeError(meta["status"])
    pxmm = cal["px_per_mm"]
    edges = cal_dir / (p.stem + "_edges.npz")
    meta["calibration"]["alex_gap_px_crosscheck"] = ALEX_GAP_PX
    # the folder label (45 deg per blade from vertical = 90 deg full opening) is an independent check on
    # the blade loci; flagged (not stopped) because a two-band blade can tilt a RANSAC line by a few deg
    meta["calibration"]["angle_flag"] = abs(cal["full_opening_angle_deg"] - 2 * (meta["angle_deg"] or 45)) > ANGLE_TOL_DEG
    save()

    # 3. analyze windows in parallel
    starts = list(range(0, src.n, CHUNK))
    chunk_dirs = [od / "chunks" / f"c{k:02d}" for k in range(len(starts))]

    import hashlib
    stamp = dict(git=meta["git"], T_low=round(T_low, 6), T_high=round(T_high, 6), low_frac=low_frac,
                 px_per_mm=round(pxmm, 6), edges_md5=hashlib.md5(edges.read_bytes()).hexdigest(),
                 background_md5=hashlib.md5(bgp.read_bytes()).hexdigest())

    def one(k):
        a = starts[k]
        cnt = min(CHUNK, src.n - a)
        cd = chunk_dirs[k]
        sp = cd / "chunk_params.json"
        if (cd / "measurements.csv").exists() and not force and sp.exists() and json.loads(sp.read_text()) == stamp:
            return 0.0      # reuse ONLY a chunk made with identical code, thresholds and calibration
        (cd / "measurements.csv").unlink(missing_ok=True)
        cd.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(stamp))
        return run([HERE / "analyze_blade.py", p, "--out-dir", cd, "--fps", f"{src.fps:.0f}",
                    "--frame-start", a, "--frame-count", cnt, "--px-per-mm", f"{pxmm:.5f}",
                    "--background", bgp, "--low-thresh", f"{T_low:.3f}", "--low-frac", low_frac,
                    "--bright-thresh", f"{T_high:.3f}", "--dark-guard", f"{T_high:.3f}",
                    "--min-arc-px", MIN_ARC_PX, "--max-arc-px", 10**9, "--close-radius", CLOSE_R,
                    "--flash-eff-width", 0, "--flash-arc-width", 0,
                    "--electrode-edges", edges, "--elec-tol-px", ELEC_TOL_PX,
                    "--arc-direction", ARC_DIRECTION, "--qa-interval", 2500],
                   od / f"analyze_c{k:02d}.log")

    ta = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(starts))))
    meta["analyze_s"] = time.time() - ta

    import pandas as pd
    dfs = []
    for k, cd in enumerate(chunk_dirs):
        df = pd.read_csv(cd / "measurements.csv")
        expect = min(CHUNK, src.n - starts[k])
        if len(df) != expect:
            raise RuntimeError(f"chunk {k}: {len(df)} rows, expected {expect}")
        df["file"] = p.stem
        df["frame"] = df["frame"] + starts[k]
        df["time_s"] = df["frame"] / src.fps
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) != src.n or not np.array_equal(merged["frame"].to_numpy(), np.arange(src.n)):
        raise RuntimeError(f"merged frames {len(merged)} != header {src.n} or not contiguous")
    mcsv = od / "measurements.csv"
    for stale in ("measurements_raw.csv", "POST.json"):
        (od / stale).unlink(missing_ok=True)      # a new merged CSV invalidates any earlier post-stage
    merged.to_csv(mcsv, index=False)
    meta["frames"] = dict(total=int(len(merged)), arc_detected=int((merged["arc_detected"] == 1).sum()),
                          arc_complete=int((merged["arc_complete"] == 1).sum()),
                          flash_flagged=int((merged["is_flash"] == 1).sum()))
    save()

    # 4. downstream stages
    fps = f"{src.fps:.0f}"
    run([HERE / "event_velocity.py", mcsv, od, "--px-per-mm", f"{pxmm:.5f}", "--fps", fps,
         "--local-vel-n", LOCAL_VEL_N], log)
    ev = od / "measurements_all_events.csv"
    if ev.exists():
        run([HERE / "qa_restrike.py", "--csv", ev, "--cine", p, "--background", bgp,
             "--arc-direction", ARC_DIRECTION, "--px-per-mm", f"{pxmm:.5f}", "--fps", fps,
             "--thresh", f"{T_high:.3f}", "--low-thresh", f"{T_low:.3f}", "--low-frac", low_frac,
             "--dark-guard", f"{T_high:.3f}", "--method", "abs", "--threshold", "0.10", "--ext-prominence-frac", "0.10",
             "--frame-height", src.H, "--gap-row", int(round(cal["gap_row"])),
             "--out", od / "qa_restrike.png"], log)
        run([HERE / "fourier_analysis.py", ev, od, "--px-per-mm", f"{pxmm:.5f}", "--fps", fps,
             "--freq-max", FREQ_MAX_HZ], log)
    if figures and ev.exists():
        # figure-only stages (build_velocity_map re-reads and re-binarises the whole cine on one
        # thread, ~15 min/file): off in the batch, run afterwards for chosen files
        run([HERE / "build_velocity_map.py", "--csv", ev, "--cine", p, "--background", bgp,
             "--out-dir", od, "--px-per-mm", f"{pxmm:.5f}", "--fps", fps,
             "--thresh", f"{T_high:.3f}", "--low-thresh", f"{T_low:.3f}", "--low-frac", low_frac,
             "--dark-guard", f"{T_high:.3f}", "--min-arc-px", MIN_ARC_PX, "--max-arc-px", 10**9], log)
    if figures:
        run([HERE / "visualize_blade.py", ev if ev.exists() else mcsv, od, "--px-per-mm", f"{pxmm:.5f}",
             "--arc-direction", ARC_DIRECTION], log)

    meta["elapsed_s"] = time.time() - t0
    meta["status"] = "DONE"
    save()
    done.write_text(json.dumps(meta, indent=1, default=float))
    print(f"[done] {p.name}  {meta['elapsed_s'] / 60:.1f} min  gap {cal['gap_px']:.1f}px "
          f"px/mm {pxmm:.2f}  T {T_low:.0f}/{T_high:.0f}  complete {meta['frames']['arc_complete']}/{src.n}",
          flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--match", default=None, help="substring of the cine stem")
    ap.add_argument("--list-file", default=None, help="text file of cine paths")
    ap.add_argument("--workers", type=int, default=7, help="analysis windows in parallel per file")
    ap.add_argument("--files-parallel", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--figures", action="store_true", help="also run build_velocity_map + visualize_blade")
    ap.add_argument("--low-scale", type=float, default=1.0)
    ap.add_argument("--low-frac", type=float, default=LOW_FRAC)
    ap.add_argument("--high-scale", type=float, default=1.0)
    a = ap.parse_args()
    # trial-major, then Re, then gas: every condition's trial 1 before any trial 2, and the gases
    # interleaved, so a partial batch covers every gas early
    files = sorted(DATA_ROOT.rglob("*.cine"),
                   key=lambda f: (parse_condition(f)["trial"] or 0, parse_condition(f)["Re"] or 0, parse_condition(f)["gas"]))
    if a.list_file:
        files = [Path(x.strip()) for x in Path(a.list_file).read_text().splitlines() if x.strip()]
    if a.match:
        files = [f for f in files if a.match in f.stem]
    if a.list or not (a.all or a.match or a.list_file):
        for f in files:
            print(f)
        print(len(files), "files")
        return
    tag = ("" if (a.low_scale, a.high_scale, a.low_frac) == (1.0, 1.0, LOW_FRAC)
           else f"lo{a.low_scale:g}_hi{a.high_scale:g}_fr{a.low_frac:g}")
    fails = []

    def job(f):
        try:
            process(f, a.workers, force=a.force, low_scale=a.low_scale, high_scale=a.high_scale, tag=tag,
                    figures=a.figures, low_frac=a.low_frac)
        except Exception as e:  # recorded, the batch continues
            print(f"[FAIL] {f.name}: {e}", flush=True)
            fails.append((str(f), str(e)))

    with ThreadPoolExecutor(max_workers=a.files_parallel) as ex:
        list(ex.map(job, files))
    print(f"finished: {len(files) - len(fails)} ok, {len(fails)} failed", flush=True)
    for f, e in fails:
        print("  FAIL", f, e)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
