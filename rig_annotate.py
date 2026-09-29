#!/usr/bin/env python3
"""
rig_annotate.py -- trace the electrode edges and the minimum gap of ANY rig in napari.

    python rig_annotate.py <cine or folder> [more cines/folders] [--gap-mm 2.4] [--out rig.json]

One annotation per rig/camera POSITION. Give it one cine, several, or a folder (it samples up to
--max-files cines across it). It builds reference images from the video itself -- the arc light
outlines the electrodes -- and opens napari with three layers to draw on:

  electrode A  (path)  trace the METAL edge of one electrode that faces the gap
  electrode B  (path)  the same for the other electrode
  gap          (line)  one line across the MINIMUM gap, metal edge to metal edge

Save: Shift+S at any time, and automatically when the window closes. Re-running with the same
--out reopens the saved drawing for editing. Output: <out>.json (see rig_annotation.py for the
schema) and <out>_edges.npz (the file analyze_blade --electrode-edges reads).

Reference layers (toggle in the layer list; adjust contrast with the slider). All are over --frames frames
spread across the record unless named otherwise:
  temporal std  -- (shown first) per-pixel standard deviation: the attachment bands along both blades stand out
                   cleanly even in Argon, where the glow hides everything else
  CLAHE(mean)   -- the mean after adaptive histogram equalisation (skimage equalize_adapthist)
  log(mean)     -- the mean on a log scale: blade silhouettes under the glow (clearest in Air)
  mean          -- plain per-pixel mean
  mean first N  -- mean of the first --first-n frames of each video (early arcs only)
  median  -- temporal median: electrodes silhouetted against gas glow, where there is glow
  p99     -- 99th percentile per pixel: the envelope the arc sweeps, i.e. where it attaches
  max     -- brightest value per pixel
Where the arc lights a BAND several px wide along a blade, trace the band's edge FACING THE GAP (upper/inner
side) every time -- that is the convention of the existing annotations; tracing the outer side instead moves
the gap end by the band width (~6 px).
  <file> max -- one per input cine: if these do not line up, the rig moved between files
  pipeline attachment -- if a run_blade_upwards calibration exists for the file (optional)
"""
import argparse
import datetime
import getpass
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import cine_frames as cf
import rig_annotation as ra

HERE = Path(__file__).resolve().parent


def find_cines(paths, max_files):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.cine")) if p.is_dir() else [p]
    if not files:
        sys.exit("no .cine files found in: " + ", ".join(paths))
    if len(files) > max_files:
        files = [files[i] for i in np.linspace(0, len(files) - 1, max_files).astype(int)]
    return files


REF_KEYS = ("temporal std", "CLAHE(mean)", "log(mean)", "mean", "mean first N", "median", "p99", "max")
CLAHE_CLIP = 0.02           # skimage equalize_adapthist clip_limit (its default is 0.01; 0.02 lifts faint bands)


def _norm01(im, lo=0.5, hi=99.5):
    a, b = np.percentile(im, [lo, hi])
    return np.clip((im - a) / max(b - a, 1e-9), 0, 1)


def reference_images(cine, n_frames, first_n):
    """Per-pixel statistics over n_frames spread across the record, plus the mean of the first first_n frames
    (defect-corrected 12-bit DN)."""
    from skimage import exposure
    src = cf.open_cine(str(cine))
    idx = np.linspace(0, src.n - 1, n_frames).astype(int)
    fr = np.stack([src.frame(int(i)) for i in idx]).astype(np.float32)
    first = np.stack([src.frame(i) for i in range(min(first_n, src.n))]).astype(np.float32)
    mean = fr.mean(0)
    return {"temporal std": fr.std(0), "CLAHE(mean)": exposure.equalize_adapthist(_norm01(mean), clip_limit=CLAHE_CLIP),
            "log(mean)": np.log1p(mean), "mean": mean, "mean first N": first.mean(0),
            "median": np.median(fr, 0), "p99": np.percentile(fr, 99, axis=0), "max": fr.max(0)}


def gap_mm_from_path(files):
    for f in files:
        m = re.search(r"([\d.]+)\s*mm\s*_?gap", str(f), flags=re.I)
        if m:
            return float(m.group(1))
    return None


def pipeline_overlay(cine, root):
    """Attachment clouds + gap ends from a run_blade_upwards calibration of this cine, if any."""
    if root is None:
        return None
    hits = list(Path(root).rglob(f"{Path(cine).stem}_edges.npz"))
    if not hits:
        return None
    z = np.load(hits[0])
    return np.vstack([z["A"], z["B"]]), np.vstack([z["A_tip"], z["B_tip"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".cine files and/or folders (same rig position)")
    ap.add_argument("--gap-mm", type=float, default=None,
                    help="known minimum gap in mm (default: parsed from a '<x>mmGap' folder name)")
    ap.add_argument("--out", default=None, help="output .json (default: ./rig_annotations/rig_<first cine stem>.json)")
    ap.add_argument("--max-files", type=int, default=4, help="cines sampled from a folder")
    ap.add_argument("--frames", type=int, default=1500, help="frames sampled per cine for the reference images")
    ap.add_argument("--first-n", type=int, default=2000, help="frames averaged for the 'mean first N' layer")
    ap.add_argument("--selftest", action="store_true",
                    help="build the viewer off-screen, draw a known V annotation, save, exit (for testing)")
    ap.add_argument("--overlay-root", default=r"E:\BladeReactor_Upwards_analysis\runs",
                    help="folder searched for pipeline calibrations to overlay ('' to disable)")
    a = ap.parse_args()

    files = find_cines(a.paths, a.max_files)
    gap_mm = a.gap_mm if a.gap_mm is not None else gap_mm_from_path(files)
    if gap_mm is None:
        sys.exit("could not read the gap from the folder name; pass --gap-mm")
    out = Path(a.out) if a.out else Path.cwd() / "rig_annotations" / f"rig_{files[0].stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not str(out).lower().endswith(".json"):
        out = out.with_suffix(".json")
    print(f"{len(files)} cine(s); gap = {gap_mm} mm; saving to {out}")

    refs = []
    for f in files:
        print("  reference images:", f.name, flush=True)
        refs.append(reference_images(f, a.frames, a.first_n))
    stackd = {k: np.median(np.stack([r[k] for r in refs]), 0) for k in REF_KEYS}

    prev = json.loads(out.read_text()) if out.exists() else {}
    import napari
    v = napari.Viewer(title=f"rig_annotate  |  gap {gap_mm} mm  |  Shift+S saves, closing saves", show=not a.selftest)
    for k in reversed(REF_KEYS):            # napari lists the LAST added layer on top
        im = stackd[k]
        name = f"mean first {a.first_n}" if k == "mean first N" else k
        v.add_image(im, name=name, colormap="gray", visible=(k == "temporal std"), gamma=0.5,
                    contrast_limits=[float(np.percentile(im, 1)), float(np.percentile(im, 99.9)) + 1])
    for f, r in zip(files, refs):
        v.add_image(r["max"], name=f"{f.stem} max", colormap="magma", visible=False, blending="additive", gamma=0.5)
    ov = pipeline_overlay(files[0], a.overlay_root or None)
    if ov is not None:
        v.add_points(ov[0], name="pipeline attachment", size=1.5, face_color="yellow", opacity=0.5, visible=False)
        v.add_points(ov[1], name="pipeline gap ends", size=4, face_color="red", visible=False)
    la = v.add_shapes([np.asarray(prev["electrode_A"])] if prev.get("electrode_A") else None, shape_type="path",
                      name="electrode A", edge_color="cyan", edge_width=1)
    lb = v.add_shapes([np.asarray(prev["electrode_B"])] if prev.get("electrode_B") else None, shape_type="path",
                      name="electrode B", edge_color="lime", edge_width=1)
    lg = v.add_shapes([np.asarray(prev["gap_line"])] if prev.get("gap_line") else None, shape_type="line",
                      name="gap", edge_color="red", edge_width=1)

    def collect():
        def last(layer):
            return np.asarray(layer.data[-1])[:, -2:].tolist() if len(layer.data) else None
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        return dict(cines=[str(f) for f in files], frames_per_cine=a.frames, image_shape=list(stackd["p99"].shape),
                    coordinates="raw (row, col), pycine orientation, row 0 = top of frame",
                    gap_mm=gap_mm, electrode_A=last(la), electrode_B=last(lb), gap_line=last(lg),
                    annotator=getpass.getuser(), saved=datetime.datetime.now().isoformat(timespec="seconds"),
                    tool_git=git)

    def do_save(*_):
        ann = collect()
        if not (ann["electrode_A"] or ann["electrode_B"] or ann["gap_line"]):
            msg = f"nothing drawn: {out.name} NOT written"
            print(msg, flush=True)
            return msg
        try:
            ann = ra.save(ann, out)
            d = ann["derived"]
            if ann.get("electrode_A") and ann.get("electrode_B"):
                ra.edges_npz(ann, out.with_name(out.stem + "_edges.npz"))
            msg = (f"saved {out.name}: gap {d['gap_px']:.2f} px = {d['px_per_mm']:.3f} px/mm"
                   + (f", edges' closest approach {d['gap_from_edges_px']:.2f} px, opening {d['opening_deg']:.1f} deg"
                      if "gap_from_edges_px" in d else ""))
        except ValueError as e:
            Path(out).write_text(json.dumps(dict(ann, schema=ra.SCHEMA, derived=None), indent=1))
            msg = f"saved {out.name} (incomplete: {e})"
        print(msg, flush=True)
        try:
            v.status = msg
        except Exception:
            pass
        return msg

    v.bind_key("Shift-S", do_save, overwrite=True)
    print("\nDraw: select 'electrode A' -> path tool -> click along the metal edge -> Esc;"
          " same for 'electrode B'; select 'gap' -> line tool -> drag across the minimum gap.\n"
          "Shift+S saves; closing the window saves.\n", flush=True)
    if a.selftest:
        la.add([np.array([[230.0, 100.0], [180.0, 50.0]])], shape_type="path")
        lb.add([np.array([[230.0, 124.0], [180.0, 174.0]])], shape_type="path")
        lg.add([np.array([[230.0, 100.0], [230.0, 124.0]])], shape_type="line")
        do_save()
        v.close()
        return
    napari.run()
    do_save()


if __name__ == "__main__":
    main()
