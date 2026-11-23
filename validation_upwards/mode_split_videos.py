"""One CSV per video with everything the mode split produced for it (author, 2026-11-23).

File = '#' header lines (video summary, both rules), then one row per frame:
  frame ... tip_velocity_vertical_m_s   Alex's per-frame columns (processed_GAD_BLADE_OCT26), text unchanged
  restrike, restrike_height_mm, old_arc_tip_height_mm, fall_mm
                                         on a counted re-strike frame (1.8 mm rule); restrike = 0 and the rest
                                         blank on every other frame
  arc_id, arc_mode_fixed_2p4mm, arc_mode_valley, arc_start_height_mm, arc_glide_velocity_m_s, arc_max_tip_mm,
  arc_max_length_mm, arc_discharge_time_ms
                                         the arc started by the latest re-strike, on every frame from that re-strike
                                         to the frame before the next (dark frames included; arc_present says whether
                                         an arc is visible). Blank before the first counted re-strike. The last arc of
                                         a video has no next re-strike, so its discharge time, glide velocity and
                                         maxima are blank.
Every value comes from mode_split.py's outputs (events.csv and runs.csv of both rules); nothing is recomputed.
Read with pandas.read_csv(path, comment="#").

usage: python mode_split_videos.py <alex_per_frame_dir> <mode_split_out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["frame", "time_s", "arc_present", "arc_complete", "tip_y_mm", "arc_length_mm", "tip_velocity_vertical_m_s"]
NEW = ["restrike", "restrike_height_mm", "old_arc_tip_height_mm", "fall_mm", "arc_id", "arc_mode_fixed_2p4mm",
       "arc_mode_valley", "arc_start_height_mm", "arc_glide_velocity_m_s", "arc_max_tip_mm", "arc_max_length_mm",
       "arc_discharge_time_ms"]
FMT = "%.6g"


def g(x, fmt="{:.4g}"):
    """header value; 'none' where the quantity does not exist (e.g. no Mode B re-strike in this video)"""
    return "none" if pd.isna(x) else fmt.format(x)


def header(stem, rf, rv, ev):
    r0 = rf
    lines = [f"video: {stem}",
             f"gas: {r0['gas']}", f"Re: {r0['Re']}", f"trial: {r0['trial']}",
             f"px_per_mm: {ev['px_per_mm']:.5g}",
             f"n_restrikes_detected: {int(r0['n_restrikes_detected'])}",
             f"mean_discharge_time_ms: {g(r0['mean_discharge_time_ms_UPPER_BOUND'])}",
             f"median_glide_velocity_m_s: {g(r0['median_glide_velocity_m_s'])}"]
    for tag, r in (("fixed", rf), ("valley", rv)):
        cut = r["cut_mm"]
        lines.append(f"{tag}_cut_mm: {g(cut) if pd.notna(cut) else 'no valley for this gas and Re'}")
        if pd.isna(cut):
            continue
        lines += [f"{tag}_n_modeA: {int(r['n_modeA'])}", f"{tag}_n_modeB: {int(r['n_modeB'])}",
                  f"{tag}_fracB_of_detected: {g(r['fracB_of_detected'])}",
                  f"{tag}_mean_discharge_time_A_ms: {g(r['modeA_mean_discharge_time_ms_UPPER_BOUND'])}",
                  f"{tag}_mean_discharge_time_B_ms: {g(r['modeB_mean_discharge_time_ms_UPPER_BOUND'])}",
                  f"{tag}_median_glide_velocity_A_m_s: {g(r['modeA_median_glide_velocity_m_s'])}",
                  f"{tag}_median_glide_velocity_B_m_s: {g(r['modeB_median_glide_velocity_m_s'])}",
                  f"{tag}_median_restrike_height_A_mm: {g(r['modeA_median_height_mm'])}",
                  f"{tag}_median_restrike_height_B_mm: {g(r['modeB_median_height_mm'])}"]
    return "".join(f"# {s}\n" for s in lines)


def main():
    src, ms = Path(sys.argv[1]), Path(sys.argv[2])
    out = ms / "per_video"
    out.mkdir(exist_ok=True)
    F = pd.read_csv(ms / "fixed_cut_2p4mm" / "events.csv")
    V = pd.read_csv(ms / "valley_cut" / "events.csv")
    assert (F[["recording", "frame"]].to_numpy() == V[["recording", "frame"]].to_numpy()).all()
    F["mode_valley"] = V["mode"]
    RF = pd.read_csv(ms / "fixed_cut_2p4mm" / "runs.csv").set_index("recording")
    RV = pd.read_csv(ms / "valley_cut" / "runs.csv").set_index("recording")
    PX = pd.read_csv(ms.parent / "data_package" / "recordings.csv").set_index("recording")["px_per_mm"]
    allf = sorted(src.glob("*.csv"))
    files = [p for p in allf if not p.name.startswith(".")]          # skip macOS '._' sidecars
    print(f"per-frame files: {len(files)} (skipped {len(allf) - len(files)} dot-files); "
          f"re-strikes {len(F)} in {F.recording.nunique()} recordings")
    assert {p.stem for p in files} == set(F.recording) == set(RF.index) == set(RV.index)

    n_rs = n_frames = 0
    for p in files:
        raw = p.read_text(encoding="utf-8").splitlines()
        D = pd.read_csv(p, dtype=str, keep_default_na=False)
        assert list(D.columns) == COLS and len(D) == len(raw) - 1, p.name
        fr = D["frame"].astype(int).to_numpy()
        assert (fr == np.arange(len(fr))).all()
        e = F[F.recording == p.stem].sort_values("frame").reset_index(drop=True)
        rf = e["frame"].to_numpy()
        assert (D["arc_present"].to_numpy()[rf] == "1").all(), f"{p.stem}: re-strike frame without an arc"

        N = pd.DataFrame(index=range(len(fr)))
        N["restrike"] = 0
        N.loc[rf, "restrike"] = 1
        for c_out, c_in in (("restrike_height_mm", "restrike_height_mm"), ("old_arc_tip_height_mm",
                            "old_arc_tip_height_mm"), ("fall_mm", "fall_mm")):
            N[c_out] = np.nan
            N.loc[rf, c_out] = e[c_in].to_numpy()
        k = np.searchsorted(rf, fr, side="right") - 1               # latest re-strike at or before each frame
        on = k >= 0
        arc = {"arc_id": e.index.to_numpy() + 1, "arc_mode_fixed_2p4mm": e["mode"].to_numpy(object),
               "arc_mode_valley": e["mode_valley"].to_numpy(object),
               "arc_start_height_mm": e["restrike_height_mm"].to_numpy(),
               "arc_glide_velocity_m_s": e["glide_velocity_m_s"].to_numpy(),
               "arc_max_tip_mm": e["glide_max_tip_mm"].to_numpy(),
               "arc_max_length_mm": e["glide_max_arc_length_mm"].to_numpy(),
               "arc_discharge_time_ms": e["discharge_time_ms"].to_numpy()}
        for c, vals in arc.items():
            col = np.full(len(fr), np.nan, object if vals.dtype == object else float)
            col[on] = vals[k[on]]
            N[c] = col
        N["arc_id"] = N["arc_id"].astype("Int64")
        assert list(N.columns) == NEW
        body = N.to_csv(header=False, index=False, float_format=FMT, lineterminator="\n").splitlines()
        assert len(body) == len(raw) - 1

        rr = RF.loc[p.stem].to_dict() | {"gas": RF.loc[p.stem, "gas"]}
        hv = RV.loc[p.stem].to_dict()
        txt = header(p.stem, rr, hv, {"px_per_mm": PX[p.stem]})
        with open(out / p.name, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
            f.write(raw[0] + "," + ",".join(NEW) + "\n")
            for a, b in zip(raw[1:], body):
                f.write(a + "," + b + "\n")

        # checks on the written file: Alex's columns unchanged; values read back match the events
        back = (out / p.name).read_text(encoding="utf-8").splitlines()
        nh = txt.count("\n")
        assert all(b.rsplit(",", len(NEW))[0] == a for b, a in zip(back[nh:], raw)), p.name
        R = pd.read_csv(out / p.name, comment="#", low_memory=False)
        assert len(R) == len(fr) and int(R["restrike"].sum()) == len(e)
        rs = R[R["restrike"] == 1]
        assert np.allclose(rs["restrike_height_mm"], e["restrike_height_mm"], rtol=1e-5, atol=1e-9)
        assert (rs["arc_id"].to_numpy() == np.arange(1, len(e) + 1)).all()
        dt = R.groupby("arc_id")["arc_discharge_time_ms"].first().to_numpy()
        assert np.allclose(dt, e["discharge_time_ms"], rtol=1e-5, equal_nan=True)
        n_rs += len(e)
        n_frames += len(fr)
    print(f"wrote {len(files)} files, {n_frames} frames, {n_rs} re-strikes; Alex columns verified unchanged, "
          f"re-strike heights and discharge times verified on read-back")


if __name__ == "__main__":
    main()
