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
import os
import re
import shutil
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
import rig_annotation as ra  # noqa: E402

DATA_ROOT = Path(r"E:\BladeReactor_Upwards")
OUT_ROOT = Path(r"E:\BladeReactor_Upwards_analysis")
ANNOT_DIR = HERE / "rig_annotations"   # rig_annotate.py (napari) saves one JSON per annotated video here

GAP_MM = 2.4                    # folder name "45degree_2.4mmGap": minimum electrode gap (trusted)
ARC_DIRECTION = "up"            # upward rig; pycine orientation has electrodes at the bottom
CHUNK_MAX = 10_000              # frames per analysis window at most; a run uses ceil(n / workers) so every
                                # worker gets one window (only arc_tip_vel_mm_s, a diagnostic-only column,
                                # is NaN at window edges, so the split moves no downstream number)
LOW_K, HIGH_K = 1.2, 4.0        # DECISION 23:48 / AMENDMENT 3 (OVERNIGHT_LOG)
LOW_FRAC = 0.05                 # AMENDMENT 4: low threshold >= 5 % of the frame peak (Argon glow, V-H3)
MIN_ARC_PX = 22                 # speckle floor (px^2); tiny next to any arc mask (~100-900 px^2)
ELEC_TOL_PX = 4                 # attachment tolerance to the edge clouds (validated visually)
# Against HAND-drawn metal edges the tolerance must cover the skeleton's inset: a skeleton end stops inside the
# arc body, ~half the arc width (median 4.4 px) short of the metal. Measured 2026-09-29 on Air Re2250 t1: of the
# frames the automatic-cloud run called complete, the larger end-to-hand-line distance has percentiles
# 50/75/90/99 = 2.9/4.2/5.5/7.1 px, so 4 px rejected 32 %. 7 px covers the 99th percentile and stays well under
# half the ~22 px gap. Author's choice (2026-09-29).
HAND_ELEC_TOL_PX = 7
CLOSE_R = 3
FREQ_MAX_HZ = 20_000            # fourier: do not discard the kHz band (default 500 Hz)
LOCAL_VEL_N = 10
NPOINT_N = 10                   # the 10-segment whole-arc reconstruction (the J suite's N_HEADLINE)
NPOINT_METHOD = "skeleton route foot A -> foot B, resample_arc"   # change => every run and chunk is redone
GAP_PX_PLAUSIBLE = (12.0, 45.0)  # outside this the calibration is reported as failed
ANGLE_TOL_DEG = 5.0             # full opening vs 2 x folder angle: flag beyond this
KNOWN_LATTICE = (list(range(1, 122, 8)), list(range(11, 124, 16)))


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


def out_dir_for(p: Path, tag: str = "", annotated: bool = False) -> Path:
    runs = "runs_annotated" if annotated else "runs"
    base = OUT_ROOT / (runs if not tag else f"sens_{'annot_' if annotated else ''}{tag}") / p.relative_to(DATA_ROOT).parent
    return base / p.stem


MEMBER_GAP_FLAG_FRAC = 0.10   # flag (not drop) a member whose gap is > 10 % from the members' median


SETUPS_FILE = ANNOT_DIR / "setups.json"
AUTO_VS_HAND_FLAG_PX = ELEC_TOL_PX  # flag a recording whose automatic attachment ends miss the hand cutoffs by more


def _setups():
    """{"<gas>/<angle folder>": {"boundary": [Re, trial], "before": name, "after": name}, ...}: a condition whose
    image moved mid-campaign is split into two setups at a (Re, trial) boundary (inclusive on the 'before' side).
    Recordings are ordered by (Re, trial)."""
    if not SETUPS_FILE.exists():
        return {}
    return {k: v for k, v in json.loads(SETUPS_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def condition_of(path) -> tuple | None:
    """(gas folder, angle/gap folder[, setup]) of a recording under DATA_ROOT, e.g. ('Air', '45degree_2.4mmGap').
    A condition listed in setups.json gets a third element naming its setup."""
    try:
        parts = Path(path).resolve().relative_to(DATA_ROOT.resolve()).parts
    except ValueError:
        return None
    if len(parts) < 3:
        return None
    key = tuple(parts[:2])
    split = _setups().get("/".join(key))
    if split is None:
        return key
    re_m, tr = re.search(r"Re(\d+)", Path(path).stem), re.search(r"trial(\d+)", Path(path).stem)
    if not (re_m and tr):
        return key + ("UNASSIGNED (no Re/trial in name)",)
    before = (int(re_m.group(1)), int(tr.group(1))) <= tuple(split["boundary"])
    return key + (split["before"] if before else split["after"],)


def find_annotations(p: Path):
    """Every rig_annotate JSON in ANNOT_DIR drawn on a recording of the SAME condition (gas + angle folder) as
    p. The setup is fixed within a condition, so all of them are averaged and used for every recording in it.
    Returns (paths, problems)."""
    key = condition_of(p)
    hits, problems = [], []
    for j in sorted(ANNOT_DIR.glob("rig_*.json")):
        try:
            cines = json.loads(j.read_text(encoding="utf-8")).get("cines", [])
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"{j.name} unreadable: {e}")
            continue
        conds = {condition_of(c) for c in cines}
        if key in conds:
            if len(conds) > 1:
                problems.append(f"{j.name} was drawn on recordings from {len(conds)} conditions {sorted(map(str, conds))}")
            hits.append(j)
    if not hits:
        problems.append(f"no hand annotation for condition {key} in {ANNOT_DIR}")
    return hits, problems


def annotation_md5(paths) -> str:
    """One hash over all member files: adding, removing or re-saving any member changes it."""
    import hashlib
    h = hashlib.md5()
    for q in sorted(paths):
        h.update(Path(q).name.encode())
        h.update(Path(q).read_bytes())
    return h.hexdigest()


def annotation_calibration(p: Path, H: int, W: int):
    """Hand-annotated gap and electrode edges for one recording: the AVERAGE of every annotation of its
    condition (rig_annotation.average_straight), BLADE edges as straight lines from the gap cutoff to the image
    border. Returns (cal, problems, flags): `problems` non-empty means the file must STOP. Everything is
    recomputed from the drawn lines, never read from stored numbers."""
    paths, problems = find_annotations(p)
    if problems:
        return None, problems, []
    anns, flags = [], []
    gap_folder = parse_condition(p)["gap_mm_folder"]
    for q in paths:
        try:
            a = ra.load(q)
        except (ValueError, KeyError) as e:
            problems.append(f"{q.name} is incomplete or unreadable: {e}")
            continue
        if not (a.get("electrode_A") and a.get("electrode_B") and a.get("gap_line")):
            problems.append(f"{q.name} needs both electrode edges and a gap line")
        if list(a.get("image_shape", [])) != [H, W]:
            problems.append(f"{q.name} drawn on a {a.get('image_shape')} image, recording is {[H, W]}")
        if gap_folder is not None and not np.isclose(a["gap_mm"], gap_folder, rtol=0, atol=1e-9):
            problems.append(f"{q.name} gap_mm {a['gap_mm']} != folder {gap_folder} mm")
        try:
            sb_m = ra.straight_blades(a, (H, W))
        except ValueError as e:
            problems.append(f"{q.name}: {e}")
            continue
        for k, off in zip("AB", sb_m["gap_end_offsets_px"]):
            if off > ELEC_TOL_PX:
                flags.append(f"{q.name}: gap-line end {off:.1f} px off the blade-{k} line (> {ELEC_TOL_PX} px)")
        anns.append(a)
    if problems:
        return None, problems, flags
    avg = ra.average_straight(anns, (H, W))
    d = ra.geometry(avg)
    sb = ra.straight_blades(avg, (H, W))
    mem = avg["members"]
    med = float(np.median(mem["gap_px"]))
    for q, g in zip(paths, mem["gap_px"]):
        if abs(g - med) > MEMBER_GAP_FLAG_FRAC * med:
            flags.append(f"{q.name}: gap {g:.2f} px is {100 * (g / med - 1):+.0f} % from the members' median "
                         f"{med:.2f} px (kept in the average)")
    if not (GAP_PX_PLAUSIBLE[0] <= d["gap_px"] <= GAP_PX_PLAUSIBLE[1]):
        problems.append(f"averaged gap {d['gap_px']:.1f} px outside {GAP_PX_PLAUSIBLE}")
    cal = dict(source="hand annotation", edge_model="straight lines from the gap cutoff to the image border",
               averaging="mean gap-line ends and mean blade directions over every annotation of the condition",
               condition=list(condition_of(p)), annotations=[str(q) for q in paths],
               annotation_md5=annotation_md5(paths), members=mem,
               px_per_mm_sem=mem["gap_px_sem"] / avg["gap_mm"], averaged_annotation=avg,
               gap_source=d["gap_source"], gap_mm=avg["gap_mm"],
               gap_px=d["gap_px"], px_per_mm=d["px_per_mm"], gap_ends=d["gap_ends"], gap_row=d["gap_row"],
               cutoff_sep_px=sb["cutoff_sep_px"], full_opening_angle_deg=sb["opening_deg"],
               # the attachment test's two point sets ARE the extended edges; they are closest at the cutoffs:
               cloud_sep_min_px=sb["cutoff_sep_px"], gap_ransac_tips_px=float("nan"),
               flags=flags, status="OK" if not problems else "FAIL: " + "; ".join(problems))
    return cal, problems, flags


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


def up_to_date(p: Path, tag="", annotated=False):
    """The DONE record of p if it needs no re-run (same annotations, tolerance and 10-segment outputs), else None."""
    done = out_dir_for(p, tag, annotated) / "DONE.json"
    if not done.exists():
        return None
    prev = json.loads(done.read_text())
    if prev.get("status") == "NO_GLIDING":
        return prev
    if prev.get("npoint_n") != NPOINT_N or prev.get("npoint_method") != NPOINT_METHOD:
        return None
    if not annotated:
        return prev
    a_paths, a_prob = find_annotations(p)
    cur_md5 = annotation_md5(a_paths) if not a_prob else None
    same_tol = prev.get("elec_tol_px", ELEC_TOL_PX) == HAND_ELEC_TOL_PX
    return prev if same_tol and prev.get("calibration", {}).get("annotation_md5") == cur_md5 else None


def local_copy(src: Path, cache: Path) -> Path:
    """Byte-for-byte copy of a cine onto the local disk (sequential read of the USB drive, then fast random
    access). Written to a .part name and renamed only after the size check, so a half copy is never read."""
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return dst
    part = cache / (src.name + ".part")
    shutil.copyfile(src, part)
    if part.stat().st_size != src.stat().st_size:
        raise RuntimeError(f"copy of {src.name} is {part.stat().st_size} bytes, source {src.stat().st_size}")
    os.replace(part, dst)
    return dst


def process(p: Path, workers: int, force=False, low_scale=1.0, high_scale=1.0, tag="", figures=False,
            low_frac=LOW_FRAC, annotated=False, read_path: Path | None = None):
    """p names the recording (condition, output dir, annotations); read_path, if given, is a byte-identical
    local copy of it that every stage READS (the USB drive seeks when 7+ workers read one cine at once)."""
    rp = Path(read_path) if read_path is not None else p
    od = out_dir_for(p, tag, annotated)
    od.mkdir(parents=True, exist_ok=True)
    log = od / "pipeline.log"
    done = od / "DONE.json"
    if not force:
        prev = up_to_date(p, tag, annotated)
        if prev is not None:
            print(f"[skip] {p.name} (DONE)", flush=True)
            return prev
        if done.exists():
            print(f"[redo] {p.name}: its hand annotations, tolerance or 10-segment outputs are out of date",
                  flush=True)
    t0 = time.time()
    meta = dict(cine=str(p), read_from=str(rp), out=str(od), tag=tag, **parse_condition(p),
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
    src = cf.open_cine(str(rp))
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
    if annotated:           # fail before the expensive stages, not after them
        _, problems, _ = annotation_calibration(p, src.H, src.W)
        if problems:
            meta["status"] = "STOPPED: annotation " + "; ".join(problems)
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
    gap_calibration.main(str(rp), str(cal_dir), T_high, T_low, str(bgp), low_frac)
    cal = json.loads((cal_dir / (p.stem + "_gap.json")).read_text())
    meta["calibration"] = cal
    meta["regime"] = dict(frames_scanned=cal.get("frames_scanned"), detected=cal.get("detected"),
                          clean_arches=cal.get("clean_arches"),
                          arch_frac_of_frames=(cal.get("clean_arches") or 0) / max(cal.get("frames_scanned") or 1, 1))
    if str(cal.get("status", "")).startswith("FAIL: fewer than 50 clean arches"):
        # a REGIME result, not an error: the discharge sits as spots and hardly ever forms an arch across
        # the gap, so there is nothing to calibrate from and gliding-arc observables are undefined
        meta["status"] = "NO_GLIDING"
        meta["elapsed_s"] = time.time() - t0
        save()
        done.write_text(json.dumps(meta, indent=1, default=float))
        print(f"[done] {p.name}  NO_GLIDING  detected {cal.get('detected')}  clean arches {cal.get('clean_arches')}"
              f" / {cal.get('frames_scanned')} frames", flush=True)
        return meta
    if annotated:
        # gap and electrode edges from the HAND annotation; the automatic calibration above is kept only for
        # the regime count (NO_GLIDING) and for comparison, so its geometry failures do not stop the file
        hcal, problems, flags = annotation_calibration(p, src.H, src.W)
        meta["calibration_auto"] = cal
        if problems:
            meta["calibration"] = hcal
            meta["status"] = "STOPPED: annotation " + "; ".join(problems)
            save()
            raise RuntimeError(meta["status"])
        hcal["clean_arches"] = cal.get("clean_arches")
        cal = hcal
        meta["calibration"] = cal
        edges = ra.edges_npz(cal["averaged_annotation"], cal_dir / (p.stem + "_edges_annotated.npz"),
                             shape=(src.H, src.W))
        # independent per-recording check: the automatic arc-attachment gap ends of THIS video against the hand
        # cutoffs of its setup (a video in the wrong setup, or a camera move, shows here as a multi-px miss)
        auto_ends = meta["calibration_auto"].get("gap_ends")
        if auto_ends:
            a_s = sorted(auto_ends, key=lambda e: e[1])
            h_s = sorted(cal["gap_ends"], key=lambda e: e[1])
            miss = [float(np.hypot(*(np.subtract(x, y)))) for x, y in zip(a_s, h_s)]
            cal["auto_vs_hand_end_px"] = miss
            if max(miss) > AUTO_VS_HAND_FLAG_PX:
                flags.append(f"automatic attachment ends miss the hand cutoffs by {miss[0]:.1f} / {miss[1]:.1f} px "
                             f"(> {AUTO_VS_HAND_FLAG_PX} px): wrong setup or the image moved?")
        cal["flags"] = flags
        for fl in flags:
            print(f"[flag] {p.name}: {fl}", flush=True)
    else:
        if cal.get("status") != "OK" or not (GAP_PX_PLAUSIBLE[0] <= cal["gap_px"] <= GAP_PX_PLAUSIBLE[1]):
            meta["status"] = f"STOPPED: calibration {cal.get('status')} gap_px {cal.get('gap_px')}"
            save()
            raise RuntimeError(meta["status"])
        edges = cal_dir / (p.stem + "_edges.npz")
    pxmm = cal["px_per_mm"]
    # the folder label (45 deg per blade from vertical = 90 deg full opening) is an independent check on
    # the blade loci; flagged (not stopped) because a two-band blade can tilt a RANSAC line by a few deg
    meta["calibration"]["angle_flag"] = abs(cal["full_opening_angle_deg"] - 2 * (meta["angle_deg"] or 45)) > ANGLE_TOL_DEG
    save()

    # 3. analyze windows in parallel
    chunk = min(CHUNK_MAX, -(-src.n // max(workers, 1)))
    starts = list(range(0, src.n, chunk))
    meta["chunk_frames"] = chunk
    chunk_dirs = [od / "chunks" / f"c{k:02d}" for k in range(len(starts))]

    import hashlib
    elec_tol = HAND_ELEC_TOL_PX if annotated else ELEC_TOL_PX
    meta["elec_tol_px"] = elec_tol
    stamp = dict(git=meta["git"], T_low=round(T_low, 6), T_high=round(T_high, 6), low_frac=low_frac,
                 elec_tol_px=elec_tol, npoint_n=NPOINT_N, npoint_method=NPOINT_METHOD,
                 px_per_mm=round(pxmm, 6), edges_md5=hashlib.md5(edges.read_bytes()).hexdigest(),
                 background_md5=hashlib.md5(bgp.read_bytes()).hexdigest())

    def one(k):
        a = starts[k]
        cnt = min(chunk, src.n - a)
        cd = chunk_dirs[k]
        sp = cd / "chunk_params.json"
        kstamp = dict(stamp, frame_start=a, frame_count=cnt)
        if (cd / "measurements.csv").exists() and not force and sp.exists() and json.loads(sp.read_text()) == kstamp:
            return 0.0      # reuse ONLY a chunk made with identical code, thresholds and calibration
        (cd / "measurements.csv").unlink(missing_ok=True)
        cd.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(kstamp))
        return run([HERE / "analyze_blade.py", rp, "--out-dir", cd, "--fps", f"{src.fps:.0f}",
                    "--npoint-n", NPOINT_N, "--no-figures",
                    "--frame-start", a, "--frame-count", cnt, "--px-per-mm", f"{pxmm:.5f}",
                    "--background", bgp, "--low-thresh", f"{T_low:.3f}", "--low-frac", low_frac,
                    "--bright-thresh", f"{T_high:.3f}", "--dark-guard", f"{T_high:.3f}",
                    "--min-arc-px", MIN_ARC_PX, "--max-arc-px", 10**9, "--close-radius", CLOSE_R,
                    "--flash-eff-width", 0, "--flash-arc-width", 0,
                    "--electrode-edges", edges, "--elec-tol-px", elec_tol,
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
        expect = min(chunk, src.n - starts[k])
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
    fr_np, pt_np = [], []
    for k, cd in enumerate(chunk_dirs):
        z = np.load(cd / "npoint_xy.npz")
        if int(z["n"]) != NPOINT_N or int(z["H"]) != src.H:
            raise RuntimeError(f"chunk {k}: npoint_xy.npz n={int(z['n'])} H={int(z['H'])}")
        fr_np.append(z["frames"] + starts[k])
        pt_np.append(z["pts"])
    fr_np = np.concatenate(fr_np)
    if not np.all(np.diff(fr_np) > 0):
        raise RuntimeError("merged npoint frames not strictly increasing")
    det_frames = merged.loc[merged["arc_detected"] == 1, "frame"].to_numpy()
    if not np.isin(fr_np, det_frames).all():
        raise RuntimeError("npoint frames include a frame the CSV does not call detected")
    np.savez(od / "npoint_xy.npz", n=NPOINT_N, H=src.H, frames=fr_np, pts=np.concatenate(pt_np))
    meta["npoint_frames"] = dict(with_points=int(len(fr_np)), detected=int(len(det_frames)))
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
        run([HERE / "qa_restrike.py", "--csv", ev, "--cine", rp, "--background", bgp,
             "--arc-direction", ARC_DIRECTION, "--px-per-mm", f"{pxmm:.5f}", "--fps", fps,
             "--thresh", f"{T_high:.3f}", "--low-thresh", f"{T_low:.3f}", "--low-frac", low_frac,
             "--dark-guard", f"{T_high:.3f}", "--method", "abs", "--threshold", "0.10", "--ext-prominence-frac", "0.10",
             "--frame-height", src.H, "--gap-row", int(round(cal["gap_row"])),
             "--out", od / "qa_restrike.png"], log)
        run([HERE / "fourier_analysis.py", ev, od, "--px-per-mm", f"{pxmm:.5f}", "--fps", fps,
             "--freq-max", FREQ_MAX_HZ], log)
        # 10-segment whole-arc analysis (npoint re-strikes, per-segment velocities) on this run's points
        run([HERE / "npoint_blade.py", od, "--n", NPOINT_N], log)
        meta["npoint_n"] = NPOINT_N
        meta["npoint_method"] = NPOINT_METHOD
        meta["npoint_summary"] = json.loads((od / "npoint_summary.json").read_text())
    if figures and ev.exists():
        # figure-only stages (build_velocity_map re-reads and re-binarises the whole cine on one
        # thread, ~15 min/file): off in the batch, run afterwards for chosen files
        run([HERE / "build_velocity_map.py", "--csv", ev, "--cine", rp, "--background", bgp,
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
    ap.add_argument("--annotations", action="store_true",
                    help="use the hand-drawn gap and electrode edges from rig_annotations\\ (rig_annotate.py); a "
                         "recording without one is STOPPED. Output goes to runs_annotated\\")
    ap.add_argument("--cache-dir", default=None,
                    help="copy each cine to this LOCAL directory first (next one prefetched while the current "
                         "runs) and read it from there; one recording at a time; copies deleted after use")
    ap.add_argument("--check-annotations", action="store_true",
                    help="list every recording's hand annotation (px/mm, gap line vs edges, problems); run nothing")
    a = ap.parse_args()
    # trial-major, then Re, then gas: every condition's trial 1 before any trial 2, and the gases
    # interleaved, so a partial batch covers every gas early
    files = sorted(DATA_ROOT.rglob("*.cine"),
                   key=lambda f: (parse_condition(f)["trial"] or 0, parse_condition(f)["Re"] or 0, parse_condition(f)["gas"]))
    if a.list_file:
        files = [Path(x.strip()) for x in Path(a.list_file).read_text().splitlines() if x.strip()]
    if a.match:
        files = [f for f in files if a.match in f.stem]
    if a.check_annotations:
        n_ok, seen = 0, {}
        for f in files:
            s_ = cf.open_cine(str(f))
            cal, problems, flags = annotation_calibration(f, s_.H, s_.W)
            key = condition_of(f)
            if problems:
                print(f"MISSING/BAD  {f.stem}: {'; '.join(problems)}")
                continue
            n_ok += 1
            if key not in seen:     # print each condition's average once, with its members
                seen[key] = cal
                m = cal["members"]
                print(f"\n{key[0]} / {key[1]}: AVERAGE of {m['n']} annotation(s): {cal['px_per_mm']:.3f} "
                      f"+/- {cal['px_per_mm_sem']:.3f} px/mm (sem)  gap {cal['gap_px']:.2f} px "
                      f"(member sd {m['gap_px_sd']:.2f} px)  opening {cal['full_opening_angle_deg']:.1f} deg")
                for q, g, o in zip(cal["annotations"], m["gap_px"], m["opening_deg"]):
                    print(f"    member {Path(q).name}: gap {g:.2f} px, opening {o:.1f} deg")
                for fl in flags:
                    print(f"    FLAG: {fl}")
            print(f"ok           {f.stem}  <- {key[0]} / {key[1]} average")
        print(f"\n{n_ok} of {len(files)} recordings have a usable hand annotation")
        return
    if a.list or not (a.all or a.match or a.list_file):
        for f in files:
            print(f)
        print(len(files), "files")
        return
    tag = ("" if (a.low_scale, a.high_scale, a.low_frac) == (1.0, 1.0, LOW_FRAC)
           else f"lo{a.low_scale:g}_hi{a.high_scale:g}_fr{a.low_frac:g}")
    fails = []

    def job(f, read_path=None):
        try:
            process(f, a.workers, force=a.force, low_scale=a.low_scale, high_scale=a.high_scale, tag=tag,
                    figures=a.figures, low_frac=a.low_frac, annotated=a.annotations, read_path=read_path)
        except Exception as e:  # recorded, the batch continues
            print(f"[FAIL] {f.name}: {e}", flush=True)
            fails.append((str(f), str(e)))

    if a.cache_dir:
        # --files-parallel recordings at a time, each read from a local copy. Copies are made ONE AT A TIME (a
        # lock: the USB drive is fast only when read sequentially), so one recording copies while the others
        # compute. Only recordings that will actually run are copied; each copy is deleted after its run (only
        # files local_copy created, inside the cache dir).
        import threading
        cache = Path(a.cache_dir)
        todo = [f for f in files if a.force or up_to_date(f, tag, a.annotations) is None]
        print(f"{len(files)} recordings, {len(todo)} to run (the rest are up to date)", flush=True)
        copy_lock = threading.Lock()

        def cached_job(f):
            try:
                with copy_lock:
                    lp = local_copy(f, cache)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {f.name}: local copy: {e}", flush=True)
                fails.append((str(f), f"local copy: {e}"))
                return
            try:
                job(f, lp)
            finally:
                lp.unlink(missing_ok=True)

        with ThreadPoolExecutor(max_workers=a.files_parallel) as ex:
            list(ex.map(cached_job, todo))
        print(f"finished: {len(todo) - len(fails)} ok, {len(fails)} failed", flush=True)
        for f, e in fails:
            print("  FAIL", f, e)
        sys.exit(1 if fails else 0)

    with ThreadPoolExecutor(max_workers=a.files_parallel) as ex:
        list(ex.map(job, files))
    print(f"finished: {len(files) - len(fails)} ok, {len(fails)} failed", flush=True)
    for f, e in fails:
        print("  FAIL", f, e)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
