"""Alex-format copies of the per-frame files with the Mode A/B labels of mode_split.py added (author, 2026-11-23).

For every per-frame file of the Alex package (processed_GAD_BLADE_OCT26/<video>.csv: frame, time_s, arc_present,
arc_complete, tip_y_mm, arc_length_mm, tip_velocity_vertical_m_s) this writes a file of the same name whose first
seven columns are the original text, unchanged, followed by:
  restrike               1 on the frame of a counted re-strike (1.8 mm rule), else 0
  arc_mode_fixed_2p4mm   A/B of the re-strike that started the arc on this frame (fixed 2.4 mm cut)
  arc_mode_valley        the same under the per-condition valley; blank where the condition has no valley
The arc mode is filled from a re-strike frame up to the frame before the next one, on arc_present = 1 frames only;
it is blank before the first counted re-strike of a video. Also writes restrikes_counted_modes.csv, one row per
re-strike with both labels, position, glide velocity and discharge time (from mode_split.py's events.csv).

usage: python mode_split_export.py <alex_per_frame_dir> <mode_split_out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["frame", "time_s", "arc_present", "arc_complete", "tip_y_mm", "arc_length_mm", "tip_velocity_vertical_m_s"]


def main():
    src, ms = Path(sys.argv[1]), Path(sys.argv[2])
    out = ms / "processed_GAD_BLADE_OCT26_modes"
    out.mkdir(exist_ok=True)
    F = pd.read_csv(ms / "fixed_cut_2p4mm" / "events.csv")
    V = pd.read_csv(ms / "valley_cut" / "events.csv")
    key = ["recording", "frame"]
    assert (F[key].to_numpy() == V[key].to_numpy()).all(), "the two rules must list the same re-strikes"
    E = F.rename(columns={"mode": "mode_fixed_2p4mm", "cut_mm": "cut_fixed_mm"})
    E["mode_valley"], E["cut_valley_mm"] = V["mode"], V["cut_mm"]
    E["time_s"] = np.nan
    allf = sorted(src.glob("*.csv"))
    files = [p for p in allf if not p.name.startswith(".")]      # macOS '._' resource-fork sidecars are not data
    print(f"per-frame files in {src}: {len(files)} (skipped {len(allf) - len(files)} dot-files: "
          f"{[p.name for p in allf if p.name.startswith('.')]}); re-strikes {len(E)} in {E.recording.nunique()} "
          f"recordings")
    assert set(E.recording) == {p.stem for p in files}, "recording lists differ"

    n_lines = n_rs = 0
    for p in files:
        raw = p.read_text(encoding="utf-8").splitlines()
        D = pd.read_csv(p, dtype=str, keep_default_na=False)
        assert list(D.columns) == COLS, p.name
        fr = D["frame"].astype(int).to_numpy()
        assert (fr == np.arange(len(fr))).all()
        e = E[E.recording == p.stem].sort_values("frame")
        rf = e["frame"].to_numpy()
        present = D["arc_present"].to_numpy() == "1"
        assert present[rf].all(), f"{p.stem}: a re-strike frame has no arc"
        rs = np.zeros(len(fr), int)
        rs[rf] = 1
        k = np.searchsorted(rf, fr, side="right") - 1          # index of the latest re-strike at or before the frame
        ok = (k >= 0) & present
        cols = {}
        for name, m in (("arc_mode_fixed_2p4mm", "mode_fixed_2p4mm"), ("arc_mode_valley", "mode_valley")):
            lab = e[m].fillna("").to_numpy(object)
            col = np.full(len(fr), "", object)
            col[ok] = lab[k[ok]]
            cols[name] = col
        with open(out / p.name, "w", encoding="utf-8", newline="\n") as f:
            f.write(raw[0] + ",restrike,arc_mode_fixed_2p4mm,arc_mode_valley\n")
            for i, line in enumerate(raw[1:]):
                f.write(f"{line},{rs[i]},{cols['arc_mode_fixed_2p4mm'][i]},{cols['arc_mode_valley'][i]}\n")
        # check: the original seven columns are unchanged, text for text
        back = (out / p.name).read_text(encoding="utf-8").splitlines()
        assert len(back) == len(raw)
        assert all(b.rsplit(",", 3)[0] == r for b, r in zip(back, raw)), p.name
        n_lines += len(raw) - 1
        n_rs += int(rs.sum())
        E.loc[e.index, "time_s"] = D["time_s"].astype(float).to_numpy()[rf]
    print(f"wrote {len(files)} files, {n_lines} frames, {n_rs} re-strike frames; original columns verified unchanged")
    assert n_rs == len(E)

    keep = ["recording", "gas", "Re", "trial", "frame", "time_s", "restrike_height_mm", "mode_fixed_2p4mm",
            "mode_valley", "cut_valley_mm", "old_arc_tip_height_mm", "fall_mm", "glide_velocity_m_s",
            "discharge_time_ms"]
    with open(ms / "restrikes_counted_modes.csv", "w", encoding="utf-8", newline="\n") as f:
        E[keep].to_csv(f, index=False, float_format="%.6g", lineterminator="\n")
    print(f"wrote restrikes_counted_modes.csv: {len(E)} rows")


if __name__ == "__main__":
    main()
