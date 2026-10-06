"""Per-cycle quantities of the leading tip: event velocity and maximum arc length (author, 2026-10-06).

Cycle. The frames from one kept re-strike up to (not including) the next, for a given re-strike set:
  direct  tip_restrike_direct.py, 20 % per-gas bar (the author's chosen rule);
  old     tip_restrike_check/threshold, 1.8 mm common rule (printed beside it as the sensitivity of each number to
          the re-strike rule).
Only cycles bounded by two kept re-strikes are used (the partial cycles at the start and end of a record are dropped).

Per cycle, on the tip series exactly as qa_restrike.py sees it (complete, non-flash frames; load_tip):
  arc_length_max_mm  largest arc_length_mm in the cycle (skeleton pixel count / px per mm, as in the package)
  event_velocity_m_s Theil-Sen slope (scipy.stats.theilslopes) of tip height against frame from the cycle's first
                     frame to the frame of its highest tip, x fps / (px/mm) / 1000. Needs >= MIN_PTS frames,
                     otherwise blank. Vertical component (tip height above the gap), as the local velocity.
  tip_max_mm         highest tip in the cycle, above the gap.
Caveats carried into every output: a missed re-strike merges two cycles (max arc length biased HIGH, velocity mixes
two advances); a false re-strike splits one (biased LOW). The direct rule misses ~39 % of eye-seen re-strikes in the
graded windows and ~18 % of its detections there match no click.

usage: python tip_cycles.py <analysis_root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tip_restrike_check import load_tip  # noqa: E402

MIN_PTS = 5


def series(rd: Path):
    meta, pxmm, fr, tip, erow, _ = load_tip(rd)
    m = pd.read_csv(rd / "measurements_all_events.csv", usecols=["frame", "arc_length_mm"]).set_index("frame")
    L = pd.to_numeric(m.loc[fr, "arc_length_mm"], errors="coerce").to_numpy(float)
    assert len(L) == len(fr)
    return meta, pxmm, fr, tip, erow, L


def cycles(meta, pxmm, fr, tip, erow, L, rs_frames, rule):
    fps = float(meta["fps"])
    rs = np.sort(np.unique(rs_frames))
    out = []
    for a, b in zip(rs[:-1], rs[1:]):
        sel = (fr >= a) & (fr < b)
        if not sel.any():
            continue
        f, t, l = fr[sel], tip[sel], L[sel]
        k = int(np.argmax(t))
        v = np.nan
        if k + 1 >= MIN_PTS:
            v = theilslopes(t[:k + 1], f[:k + 1])[0] * fps / pxmm / 1000.0
        out.append(dict(run=meta.get("run", ""), gas=meta["gas"], Re=meta["Re"], trial=meta["trial"], rule=rule,
                        frame_start=int(a), frame_end=int(b), cycle_ms=(b - a) / fps * 1e3, n_frames=int(sel.sum()),
                        tip_start_mm=(t[0] - erow) / pxmm, tip_max_mm=(t[k] - erow) / pxmm,
                        arc_length_max_mm=float(np.nanmax(l)) if np.isfinite(l).any() else np.nan,
                        event_velocity_m_s=v, n_rise_pts=k + 1))
    return out


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(root / "tip_restrike_direct" / "tip_restrike_direct_events.csv")
    O = pd.read_csv(root / "tip_restrike_check" / "threshold" / "tip_restrike_events_kept.csv")
    rows = []
    for rd in sorted(p.parent for p in (root / "runs_annotated").rglob("re_strikes_qa.csv")):
        meta, pxmm, fr, tip, erow, L = series(rd)
        meta = dict(meta, run=rd.name)
        rows += cycles(meta, pxmm, fr, tip, erow, L, D.loc[(D["run"] == rd.name) & D["kept"], "frame"], "direct20")
        rows += cycles(meta, pxmm, fr, tip, erow, L, O.loc[(O["run"] == rd.name) & O["kept"], "frame_min"], "old1.8")
    C = pd.DataFrame(rows)
    C.to_csv(out / "tip_cycles.csv", index=False)
    print("cycles: " + ", ".join(f"{r} {n}" for r, n in C.groupby("rule").size().items()))
    print(f"event velocity blank (< {MIN_PTS} rise points): "
          + ", ".join(f"{r} {d['event_velocity_m_s'].isna().mean():.1%}" for r, d in C.groupby("rule")))
    print("event velocity < 0: " + ", ".join(f"{r} {(d['event_velocity_m_s'] < 0).mean():.1%}" for r, d in C.groupby("rule")))
    R = C.groupby(["rule", "gas", "Re", "trial", "run"]).agg(
        n_cycles=("cycle_ms", "size"),
        event_velocity_median_m_s=("event_velocity_m_s", "median"),
        arc_length_max_mean_mm=("arc_length_max_mm", "mean"),
        arc_length_max_median_mm=("arc_length_max_mm", "median")).reset_index()
    R.to_csv(out / "tip_cycles_by_run.csv", index=False)
    K = R.groupby(["rule", "gas", "Re"]).agg(
        trials=("run", "size"),
        event_velocity_median_m_s_mean=("event_velocity_median_m_s", "mean"),
        event_velocity_median_m_s_sd=("event_velocity_median_m_s", "std"),
        arc_length_max_mean_mm_mean=("arc_length_max_mean_mm", "mean"),
        arc_length_max_mean_mm_sd=("arc_length_max_mean_mm", "std")).reset_index()
    K.to_csv(out / "tip_cycles_by_condition.csv", index=False)
    W = K.pivot_table(index=["gas", "Re"], columns="rule",
                      values=["event_velocity_median_m_s_mean", "arc_length_max_mean_mm_mean"])
    W[("vel_change", "direct/old")] = W[("event_velocity_median_m_s_mean", "direct20")] / W[("event_velocity_median_m_s_mean", "old1.8")]
    W[("Lmax_change", "direct/old")] = W[("arc_length_max_mean_mm_mean", "direct20")] / W[("arc_length_max_mean_mm_mean", "old1.8")]
    print("\nper condition (mean over trials of each trial's median event velocity / mean max arc length):")
    print(W.round(3).to_string())
    W.to_csv(out / "tip_cycles_rule_sensitivity.csv")


if __name__ == "__main__":
    main()
