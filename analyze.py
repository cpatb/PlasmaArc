"""
Plasma arc analysis: thresholds .cine frames, then quantifies
occupation area and arc length per frame.

Usage:
    python analyze.py recording.cine
    python analyze.py recording.cine --pixel-size 0.05   # mm/px
    python analyze.py recording.cine --close 4 --open 2
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from plasma_utils.image import binarize, measure, normalize_frame
from plasma_utils.io import load_cine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cine", help=".cine file")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Physical size of one pixel in mm")
    parser.add_argument("--close", type=int, default=3,
                        help="Morphological closing disk radius (px)")
    parser.add_argument("--open", type=int, default=2,
                        help="Morphological opening disk radius (px)")
    args = parser.parse_args()

    cine_path = Path(args.cine)
    out_dir = cine_path.parent / cine_path.stem
    out_dir.mkdir(exist_ok=True)

    frames, fps = load_cine(cine_path)
    px = args.pixel_size

    records = []
    for i, raw in enumerate(tqdm(frames, desc=cine_path.name)):
        gray = normalize_frame(raw)
        binary = binarize(gray, args.close, args.open)
        area_px, length_px = measure(binary)
        records.append({
            "frame": i,
            "time_s": i / fps if fps else i,
            "area_px": area_px,
            "arc_length_px": length_px,
            "area_mm2": area_px * px ** 2 if px else None,
            "arc_length_mm": length_px * px if px else None,
            "mean_intensity": float(gray.mean()),
        })

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "measurements.csv", index=False)

    # --- plots ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = df["time_s"]

    axes[0].plot(t, df["area_mm2"] if px else df["area_px"])
    axes[0].set_ylabel("Area (mm²)" if px else "Area (px)")
    axes[0].set_title(cine_path.name)

    axes[1].plot(t, df["arc_length_mm"] if px else df["arc_length_px"])
    axes[1].set_ylabel("Arc Length (mm)" if px else "Arc Length (px)")

    # Mean intensity exposes AC frequency; useful for phase-resolved analysis
    axes[2].plot(t, df["mean_intensity"])
    axes[2].set_ylabel("Mean Intensity")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=150)
    plt.close()

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
