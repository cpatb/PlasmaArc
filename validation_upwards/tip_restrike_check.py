"""Is each leading-tip re-strike (qa_restrike.py, re_strikes_qa.csv) a breakdown, or tip jitter?

What the tip detector marks. It finds extension PEAKS of the (3-frame smoothed) tip height and puts the re-strike
at the MINIMUM between two consecutive peaks, i.e. at the BOTTOM of a retreat, where the next advance begins.
The fall itself happens BETWEEN the previous peak and that minimum, so any before/after comparison must be taken
over that window, not across the marked frame.

What a breakdown looks like. A gliding tip only RISES (~0.03 px per 5 us frame at ~0.6 m/s, 9 px/mm). A breakdown
replaces the arc by a shorter one within one frame, so the tip FALLS by millimetres in a single step (or across a
break where no complete arc exists and the arc re-forms lower). A retreat made of many small steps is not a
breakdown: it is the detected apex wandering (shape change, skeleton spur, dim apex).

Per re-strike (indices into the detector's own frame list, raw unsmoothed tip):
  fall_mm        tip(previous peak) - tip(marked minimum), mm
  max_step_mm    largest single-step fall between those two frames (a "step" is one entry of the detector's frame
                 list, so it may span a break with no complete arc)
  step_gap_frames  cine frames spanned by that largest step (1 = consecutive frames)
  step_share     max_step / fall: ~1 = one sudden fall, small = many small steps
  centroid_drop  a centroid drop > 1.5 mm (centroid_restrikes.csv) lies inside [previous peak, marked minimum]
Reproduction gate: the detector is re-run here with the run's own parameters and must return EXACTLY the frames in
re_strikes_qa.csv, or the run is reported and skipped.

usage: python tip_restrike_check.py <runs_root> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import arc_filter  # noqa: E402
from qa_restrike import apex_and_range, detect_restrikes  # noqa: E402

# the parameters run_blade_upwards.py passes to qa_restrike.py (defaults of --min-cycle-frames / --smooth)
EXT_PROM_FRAC, MIN_CYCLE, SMOOTH = 0.10, 5, 3
SUDDEN_MM = 1.5   # the centroid detector's threshold (smallest with false-detection <= 5 %); used here for comparability


def one_run(rd: Path):
    meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
    pxmm, H = float(meta["calibration"]["px_per_mm"]), int(meta["H"])
    gap_row = int(round(meta["calibration"]["gap_row"]))
    df = pd.read_csv(rd / "measurements_all_events.csv",
                     usecols=["file", "frame", "arc_detected", "arc_tip_row", "is_flash", "area_px",
                              "arc_length_px", "arc_width_px"])
    df["arc_tip_row"] = pd.to_numeric(df["arc_tip_row"], errors="coerce")
    df = arc_filter.add_flash_flag(df)
    det = pd.to_numeric(df["arc_detected"], errors="coerce").fillna(0).astype(int) == 1
    d = df[det & ~df["is_flash"]].dropna(subset=["arc_tip_row"]).sort_values(["file", "frame"])
    if d["file"].nunique() != 1:
        raise RuntimeError(f"{rd.name}: {d['file'].nunique()} files in one run")
    fr = d["frame"].to_numpy(int)
    tip = d["arc_tip_row"].to_numpy(float)
    erow, l_eff, src = apex_and_range(tip, None, apex_override=float(H - 1 - gap_row))
    rk, peaks, _ = detect_restrikes(tip, erow, l_eff, EXT_PROM_FRAC, MIN_CYCLE, SMOOTH)
    ref = pd.read_csv(rd / "re_strikes_qa.csv")["frame"].to_numpy(int)
    got = np.array([fr[r["idx"]] for r in rk], int)
    if not np.array_equal(got, ref):
        return None, f"REPRODUCTION FAILED: {len(got)} vs {len(ref)} re-strikes, first differing frame " \
                     f"{next((a for a, b in zip(got, ref) if a != b), 'n/a')}"
    cpath = rd / "centroid_restrikes.csv"
    cdrop = pd.read_csv(cpath)["frame_after"].to_numpy(int) if cpath.exists() else np.array([], int)
    step = np.diff(tip)                              # per list entry; negative = fall
    cons = np.diff(fr) == 1
    jit = step[cons]
    jit_sd = 1.4826 * np.median(np.abs(jit - np.median(jit))) / pxmm
    rows = []
    for i, r in enumerate(rk):
        a, k = int(peaks[i]), int(r["idx"])         # previous peak, marked minimum (qa_restrike's own pairing)
        seg = -step[a:k] if k > a else np.array([0.0])
        j = int(np.argmax(seg)) if k > a else 0
        fall = (tip[a] - tip[k]) / pxmm
        ms = seg[j] / pxmm
        rows.append(dict(frame=int(fr[k]), peak_frame=int(fr[a]), n_steps=k - a, fall_mm=fall, max_step_mm=ms,
                         step_gap_frames=int(fr[a + j + 1] - fr[a + j]) if k > a else 0,
                         step_share=ms / fall if fall > 0 else np.nan,
                         tip_after_mm=(tip[k] - erow) / pxmm,
                         centroid_drop=bool(((cdrop > fr[a]) & (cdrop <= fr[k])).any())))
    ev = pd.DataFrame(rows)
    # how many centroid drops fall inside SOME tip retreat window
    win = [(int(fr[int(peaks[i])]), int(fr[int(r["idx"])])) for i, r in enumerate(rk)]
    inside = sum(any(lo < c <= hi for lo, hi in win) for c in cdrop)
    summ = dict(run=rd.name, gas=meta["gas"], Re=meta["Re"], trial=meta["trial"], n_tip=len(ev),
                jitter_sd_mm=jit_sd, cons_pairs=int(cons.sum()),
                fall_med_mm=ev["fall_mm"].median() if len(ev) else np.nan,
                max_step_med_mm=ev["max_step_mm"].median() if len(ev) else np.nan,
                frac_sudden=(ev["max_step_mm"] >= SUDDEN_MM).mean() if len(ev) else np.nan,
                frac_fall_ge=(ev["fall_mm"] >= SUDDEN_MM).mean() if len(ev) else np.nan,
                frac_step_over_break=(ev["step_gap_frames"] > 1).mean() if len(ev) else np.nan,
                frac_with_centroid_drop=ev["centroid_drop"].mean() if len(ev) else np.nan,
                n_centroid=len(cdrop), frac_centroid_inside_tip_window=inside / len(cdrop) if len(cdrop) else np.nan)
    ev.insert(0, "run", rd.name)
    return (ev, summ), None


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    evs, sums, bad = [], [], []
    for rd in sorted(p.parent for p in root.rglob("re_strikes_qa.csv")):
        res, err = one_run(rd)
        if err:
            bad.append((rd.name, err))
            print(f"[SKIP] {rd.name}: {err}", flush=True)
            continue
        evs.append(res[0])
        sums.append(res[1])
    E, S = pd.concat(evs, ignore_index=True), pd.DataFrame(sums)
    E.to_csv(out / "tip_restrike_events.csv", index=False)
    S.to_csv(out / "tip_restrike_by_run.csv", index=False)
    print(f"\nruns checked {len(S)}, reproduction failures {len(bad)}; tip re-strikes {len(E)}")
    print(f"frame-to-frame tip jitter (robust sd, consecutive frames): median {S['jitter_sd_mm'].median():.3f} mm, "
          f"range {S['jitter_sd_mm'].min():.3f}-{S['jitter_sd_mm'].max():.3f}")
    E["gas"] = E["run"].map(S.set_index("run")["gas"])
    q = lambda x: f"{x.quantile(0.1):.2f} / {x.median():.2f} / {x.quantile(0.9):.2f}"  # noqa: E731
    g = E.groupby("gas").agg(
        n=("frame", "size"),
        fall_mm_p10_med_p90=("fall_mm", q),
        max_step_mm_p10_med_p90=("max_step_mm", q),
        step_share_med=("step_share", "median"),
        frac_fall_ge_thr=("fall_mm", lambda x: (x >= SUDDEN_MM).mean()),
        frac_sudden_step_ge_thr=("max_step_mm", lambda x: (x >= SUDDEN_MM).mean()),
        frac_step_over_break=("step_gap_frames", lambda x: (x > 1).mean()),
        frac_with_centroid_drop=("centroid_drop", "mean"))
    print(f"\nper gas (thr = {SUDDEN_MM} mm):\n" + g.round(3).to_string())
    c = S.groupby("gas")[["n_centroid"]].sum()
    c["centroid drops inside a tip retreat window"] = S.groupby("gas").apply(
        lambda d: (d["frac_centroid_inside_tip_window"] * d["n_centroid"]).sum() / d["n_centroid"].sum())
    print("\n" + c.round(3).to_string())
    g.to_csv(out / "tip_restrike_by_gas.csv")


if __name__ == "__main__":
    main()
