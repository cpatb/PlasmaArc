#!/usr/bin/env python3
"""
make_calib_tiff.py  —  extract a compact, well-exposed multi-frame TIFF from a
.cine for **interactive electrode calibration** (calibrate_j_electrodes.py).

This is NOT the analysis conversion. It samples frames across the recording via
direct seek (cheap even on the 114 GB file), applies the SAME percentile-based
8-bit normalisation as run_11may26.convert_chunked (so the frame looks like the
analysis frames), and writes them to:

    <J dir>/<stem>_results/calib_frames.tiff

Electrode pixel positions are independent of intensity normalisation and
bad-pixel correction, so a frame clicked here yields geometry valid for the
faithful analysis TIFFs produced later by run_j_co2.py on the workstation.

Usage:
    python make_calib_tiff.py "<...>/HS_0.4mm_1500mL_(CO2)_trig.cine"
    python make_calib_tiff.py CINE --n-frames 400 --saturated 0.35
"""
import argparse, struct
from pathlib import Path
import numpy as np
import tifffile
from pycine.file import read_header
from pycine.raw import create_raw_array


def read_frames_at(cine_path, indices, header):
    """Yield raw uint16 frames at specific 0-based indices via direct seek."""
    with open(str(cine_path), "rb") as f:
        for idx in indices:
            f.seek(header["pImage"][idx])
            ann_size = struct.unpack("I", f.read(4))[0]
            ann_bytes = ann_size - 8
            if ann_bytes > 0:
                f.seek(ann_bytes, 1)
            img_size = struct.unpack("I", f.read(4))[0]
            yield create_raw_array(f.read(img_size), header)


def main():
    ap = argparse.ArgumentParser(description="Extract calibration frames from a .cine")
    ap.add_argument("cine")
    ap.add_argument("--out", default=None,
                    help="Output TIFF (default: <stem>_results/calib_frames.tiff)")
    ap.add_argument("--n-frames", type=int, default=300,
                    help="How many frames to sample across the recording (default 300)")
    ap.add_argument("--saturated", type=float, default=0.35,
                    help="%% clipped each end for percentile normalisation (default 0.35)")
    ap.add_argument("--min-max-dn", type=int, default=200,
                    help="Keep only sampled frames whose max DN >= this (a visible arc / "
                         "lit electrodes), so you scrub through useful frames (default 200)")
    args = ap.parse_args()

    cine_path = Path(args.cine)
    if not cine_path.exists():
        raise SystemExit("not found: %s" % cine_path)
    stem = cine_path.stem
    out = Path(args.out) if args.out else cine_path.parent / (stem + "_results") / "calib_frames.tiff"
    out.parent.mkdir(parents=True, exist_ok=True)

    header = read_header(str(cine_path))
    n_total = len(header["pImage"])
    print("%s : %d frames" % (cine_path.name, n_total))

    idx = np.linspace(0, n_total - 1, args.n_frames, dtype=int).tolist()
    raw = list(read_frames_at(cine_path, idx, header))
    stack = np.stack([np.asarray(r) for r in raw], 0)        # (N,H,W) uint16
    H, W = stack.shape[1:]

    # keep frames with signal (lit electrodes / arc) so scrubbing is useful;
    # fall back to all frames if too few pass.
    maxdn = stack.reshape(len(stack), -1).max(axis=1)
    keep = stack[maxdn >= args.min_max_dn]
    if len(keep) < 20:
        keep = stack
        print("  [note] only %d frames >= %d DN; keeping all sampled frames"
              % (int((maxdn >= args.min_max_dn).sum()), args.min_max_dn))

    lo = float(np.percentile(keep, args.saturated / 2))
    hi = float(np.percentile(keep, 100.0 - args.saturated / 2))
    if hi <= lo:
        hi = lo + 1.0
    scale = 255.0 / (hi - lo)
    eight = np.clip((keep.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)

    with tifffile.TiffWriter(str(out), bigtiff=False) as tw:
        for fr in eight:
            tw.write(fr, photometric="minisblack", contiguous=True)
    print("  wrote %d frames -> %s  (%dx%d, lo=%.1f hi=%.1f)"
          % (len(eight), out, H, W, lo, hi))
    print("  calibrate with:")
    print('    python calibrate_j_electrodes.py --tiff "%s" --gap-mm 2.4 --arc-direction down'
          % out)


if __name__ == "__main__":
    main()
