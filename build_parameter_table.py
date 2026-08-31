#!/usr/bin/env python3
"""
build_parameter_table.py  —  per-recording analysis-parameter table for the GAD paper SI.

Collects, for each of the six J-electrode CO2 recordings, every filter value,
threshold and tunable parameter that the analysis actually used, from the
authoritative per-recording state files:

    run_metadata.json              what run_j_co2.py resolved and passed
    j_geometry.json                calibration (gap-derived px/mm, gap_row, threshold)
    re_strikes_qa_modedef.json     Mode A/B valley cut, TIP detector
    npoint_restrikes_modedef.json  Mode A/B valley cut, 10-segment detector
    tiff/threshold_suggested.txt   data-driven threshold, for comparison with the used one

Constants that were NOT overridden per recording are imported from analyze_blade
rather than transcribed, so this table cannot drift from the code.

Writes two CSVs next to the other master tables:
    j_electrode_parameter_table.csv       one row per recording
    j_electrode_parameter_dictionary.csv  what each column means, units, provenance

Validation (all assertions must hold, or the row is flagged in `qc_flags`):
  * px_per_mm == min_gap_px / min_gap_mm       (calibration is gap-derived)
  * min_arc_px == round(min_gap_px)            (speckle floor tied to this recording)
  * max_arc_px == max(3000, flash_arc_width*H) (geometric bloom cap)

    python build_parameter_table.py
    python build_parameter_table.py --data-root "<J Electrode>"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd

# Constants the pipeline did NOT override per recording — import, never transcribe.
from analyze_blade import (DEFAULT_CLOSE_RADIUS, DEFAULT_DARK_GUARD,
                           DEFAULT_TOP_FRACTION, DEFAULT_QA_INTERVAL, ELEC_TOL_PX)
from npoint_restrike import RECORDINGS

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")

# Reynolds-number set is locked to the collaborator's table (npoint_restrike.RECORDINGS).
# Gas properties used to form it, recorded here so the table is self-contained.
RHO_CO2 = 1.815      # kg/m^3
MU_CO2 = 1.48e-5     # Pa.s


def _read_json(p: Path):
    """run_metadata.json has occasionally been written with trailing content."""
    txt = p.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return json.loads(txt.rsplit("}", 1)[0] + "}")


def _suggested_threshold(rd: Path):
    """Data-driven DN from qa_arc_local_intensity_j.py, if that QA was run."""
    p = rd / "tiff" / "threshold_suggested.txt"
    if not p.exists():
        return None
    for line in p.read_text(errors="replace").splitlines():
        if "threshold" in line and "=" in line:
            try:
                return int(line.split("=")[1].strip())
            except ValueError:
                return None
    return None


def collect(stem: str, data_root: Path) -> dict:
    rd = data_root / f"{stem}_results"
    m = _read_json(rd / "run_metadata.json")
    g = _read_json(rd / "j_geometry.json")
    tip = _read_json(rd / "re_strikes_qa_modedef.json")
    npt = _read_json(rd / "npoint_restrikes_modedef.json")
    info = RECORDINGS[stem]

    ppm = float(m["px_per_mm"])
    min_gap_px, min_gap_mm = float(g["min_gap_px"]), float(g["min_gap_mm"])
    H = int(m["height"])

    # ── validation: the three documented resolution rules ────────────────────
    flags = []
    if abs(ppm - min_gap_px / min_gap_mm) > 1e-6:
        flags.append(f"px_per_mm{ppm}!=min_gap_px/min_gap_mm{min_gap_px/min_gap_mm:.4f}")
    if int(m["min_arc_px"]) != int(round(min_gap_px)):
        flags.append(f"min_arc_px{m['min_arc_px']}!=round(min_gap_px){round(min_gap_px)}")
    exp_max = max(3000, int(float(m["flash_arc_width"]) * H))
    if int(m["max_arc_px"]) != exp_max:
        flags.append(f"max_arc_px{m['max_arc_px']}!={exp_max}")

    sugg = _suggested_threshold(rd)
    bt = int(m["bright_thresh"])

    # finite_segments / segment_margin_px are RECORDED by write_run_metadata but are
    # only PASSED to analyze_blade on the straight-blade branch. All six J recordings
    # carry electrode_edges.npz, so the curved-edge branch runs and these two are
    # never applied. Report both, so the SI cannot quote a parameter that did nothing.
    edges_used = (rd / "electrode_edges.npz").exists() and m.get("blade_lines") is None

    return dict(
        # ── identity ─────────────────────────────────────────────────────────
        recording_id=stem,
        source_cine=m["recording"],
        nozzle_d_mm=info["d_mm"],
        flow_mL_min=info["q"],
        gas="CO2",
        is_repeat=int(info["is_repeat"]),
        reynolds=info["re"],
        # ── acquisition ──────────────────────────────────────────────────────
        fps_Hz=int(m["fps"]),
        frame_interval_us=round(1e6 / float(m["fps"]), 3),
        n_frames=int(m["n_frames"]),
        height_px=H,
        width_px=int(m["width"]),
        # ── calibration (gap-derived, per video, never hardcoded) ────────────
        min_gap_mm=min_gap_mm,
        min_gap_px=min_gap_px,
        px_per_mm=ppm,
        px_per_mm_source=m["px_per_mm_source"],
        gap_row=int(m["gap_row"]),
        arc_direction=m["arc_direction"],
        # ── segmentation threshold ───────────────────────────────────────────
        bright_thresh_DN=bt,
        bright_thresh_source=m["bright_thresh_source"],
        bright_thresh_suggested_DN=sugg,
        bright_thresh_minus_suggested_DN=(bt - sugg) if sugg is not None else None,
        # ── arc-area gates ───────────────────────────────────────────────────
        min_arc_px=int(m["min_arc_px"]),
        max_arc_px=int(m["max_arc_px"]),
        # ── flash rejection ──────────────────────────────────────────────────
        flash_arc_width_px=float(m["flash_arc_width"]),
        flash_eff_width_px=float(m["flash_eff_width"]),
        # ── morphology / detection constants (analyze_blade defaults) ────────
        close_radius_px=DEFAULT_CLOSE_RADIUS,
        dark_guard_DN=DEFAULT_DARK_GUARD,
        top_fraction=DEFAULT_TOP_FRACTION,
        elec_tol_px=ELEC_TOL_PX,
        qa_interval_frames=DEFAULT_QA_INTERVAL,
        # ── electrode geometry ───────────────────────────────────────────────
        geometry_mode=m["geometry_mode"],
        electrode_model=g["electrode_model"],
        hand_segmented=int(bool(g.get("hand_segmented", False))),
        n_edge_pts_A=g.get("n_edge_pts_A"),
        n_edge_pts_B=g.get("n_edge_pts_B"),
        finite_segments_recorded=int(bool(m.get("finite_segments"))),
        finite_segments_applied=0 if edges_used else int(bool(m.get("finite_segments"))),
        segment_margin_px_recorded=m.get("segment_margin_px"),
        # ── local velocity ───────────────────────────────────────────────────
        local_vel_n=int(m["local_vel_n"]),
        # ── re-strike detection (tip detector) ───────────────────────────────
        restrike_method=m["restrike_method"],
        restrike_threshold=float(m["restrike_threshold"]),
        restrike_prominence_frac=float(m["restrike_prominence_frac"]),
        # ── Mode A/B boundary ────────────────────────────────────────────────
        modeAB_rule="dead-zone valley on initiation depth below apex",
        valley_cut_mm_tip=float(tip["valley_cut_mm"]),
        valley_cut_px_tip=float(tip.get("valley_cut_px", tip["valley_cut_mm"] * ppm)),
        valley_cut_mm_npoint=float(npt["valley_cut_mm"]),
        valley_cut_px_npoint=round(float(npt["valley_cut_mm"]) * ppm, 1),
        n_restrikes_tip=int(tip["n"]),
        n_restrikes_npoint=int(npt["n"]),
        frac_B_tip_pre_valley=float(tip["old_frac_B"]),
        frac_B_tip_valley=float(tip["new_frac_B"]),
        frac_B_npoint_pre_valley=float(npt["old_frac_B"]),
        frac_B_npoint_valley=float(npt["new_frac_B"]),
        # ── spectral ─────────────────────────────────────────────────────────
        freq_max_Hz=float(m["freq_max_hz"]),
        # ── uncertainty propagation ──────────────────────────────────────────
        sigma_pxmm_rel=m.get("sigma_pxmm_rel"),
        sigma_fps_rel=m.get("sigma_fps_rel"),
        # ── provenance ───────────────────────────────────────────────────────
        code_version=m["code_version"],
        run_metadata_written_utc=m["written_utc"],
        qc_flags=";".join(flags) if flags else "OK",
    )


DICTIONARY = [
    ("recording_id", "-", "identity", "recording stem; folder is <recording_id>_results"),
    ("nozzle_d_mm", "mm", "identity", "nozzle orifice diameter"),
    ("flow_mL_min", "mL/min", "identity", "CO2 volumetric flow"),
    ("reynolds", "-", "identity", f"nozzle Re, locked to collaborator table (rho={RHO_CO2} kg/m3, mu={MU_CO2} Pa.s)"),
    ("fps_Hz", "Hz", "acquisition", "frame rate, auto-detected per .cine (not assumed)"),
    ("frame_interval_us", "us", "acquisition", "1e6/fps; derived"),
    ("n_frames", "frames", "acquisition", "frames in the recording"),
    ("height_px,width_px", "px", "acquisition", "frame size; sets max_arc_px"),
    ("min_gap_mm", "mm", "calibration", "true minimum electrode gap; 3.2 mm (hand, 25JUN26)"),
    ("min_gap_px", "px", "calibration", "hand-clicked apex gap in pixels, per recording"),
    ("px_per_mm", "px/mm", "calibration", "min_gap_px/min_gap_mm. Gap-derived per video; never hardcoded (no ruler in frame)"),
    ("gap_row", "px row", "calibration", "image row of the apex; anchors Mode A/B depth and the flash cutoff"),
    ("arc_direction", "-", "calibration", "'down' = inverted-V apex at top, arc glides downward"),
    ("bright_thresh_DN", "DN", "threshold", "pixels above this DN are arc signal. HAND-SET per recording"),
    ("bright_thresh_suggested_DN", "DN", "threshold", "data-driven midpoint of plume-99th-pct and arc-10th-pct (qa_arc_local_intensity_j.py); blank where that QA was not run"),
    ("bright_thresh_minus_suggested_DN", "DN", "threshold", "used minus suggested; how far above the data-driven value the hand threshold sits"),
    ("min_arc_px", "px", "area gate", "smallest credible arc mask = round(min_gap_px); speckle floor scaled to this recording"),
    ("max_arc_px", "px", "area gate", "max(3000, flash_arc_width*height); geometric bloom cap"),
    ("flash_arc_width_px", "px", "flash", "FWHM arc width above which the frame is is_flash. CO2 value 50 (BLADE default 16 wrongly flags wide CO2 arcs)"),
    ("flash_eff_width_px", "px", "flash", "effective width (area/length) above which is_flash. CO2 value 25 (real-arc p99 ~13)"),
    ("close_radius_px", "px", "morphology", "morphological closing disk radius; analyze_blade default, not overridden"),
    ("dark_guard_DN", "DN", "morphology", "frames whose max DN is below this are skipped; default, not overridden"),
    ("top_fraction", "-", "morphology", "top fraction of frame used for electrode detection; default"),
    ("elec_tol_px", "px", "morphology", "electrode-attachment tolerance; default"),
    ("finite_segments_recorded", "bool", "geometry", "value stored in run_metadata.json"),
    ("finite_segments_applied", "bool", "geometry", "whether it reached analyze_blade. 0 for all six: electrode_edges.npz exists, so the curved-J-hook branch runs and --finite-segments is never passed. Do NOT quote the recorded value as an applied parameter"),
    ("local_vel_n", "-", "velocity", "number of arc segments for the local-velocity map"),
    ("restrike_threshold", "-", "restrike", "norm_pos cut for the INITIAL A/B split; superseded downstream by the valley cut"),
    ("restrike_prominence_frac", "-", "restrike", "peak prominence as a fraction of L_eff for re-strike detection"),
    ("valley_cut_mm_tip", "mm below apex", "Mode A/B", "dead-zone valley, TIP detector. A: depth < cut; B: depth >= cut"),
    ("valley_cut_mm_npoint", "mm below apex", "Mode A/B", "dead-zone valley, 10-segment detector (PRIMARY method)"),
    ("frac_B_*_pre_valley", "-", "Mode A/B", "Mode-B fraction under the old norm_pos>0.10 cut"),
    ("frac_B_*_valley", "-", "Mode A/B", "Mode-B fraction under the valley cut"),
    ("freq_max_Hz", "Hz", "spectral", "upper frequency bound of the Fourier analysis"),
    ("sigma_pxmm_rel,sigma_fps_rel", "-", "uncertainty", "relative systematics; null = not propagated in this run"),
    ("code_version", "-", "provenance", "engine commit that produced the run"),
    ("qc_flags", "-", "provenance", "OK, or which validation rule failed"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    args = ap.parse_args()
    data_root = Path(args.data_root)

    rows = [collect(s, data_root) for s in RECORDINGS]
    df = pd.DataFrame(rows).sort_values(["is_repeat", "reynolds"]).reset_index(drop=True)

    out_dir = data_root / "j_electrode_data_master"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = out_dir / "j_electrode_parameter_table.csv"
    dct = out_dir / "j_electrode_parameter_dictionary.csv"
    df.to_csv(tbl, index=False)
    pd.DataFrame(DICTIONARY, columns=["column", "units", "group", "meaning"]).to_csv(dct, index=False)

    pd.set_option("display.width", 250)
    print(df[["recording_id", "reynolds", "px_per_mm", "gap_row", "bright_thresh_DN",
              "bright_thresh_suggested_DN", "min_arc_px", "max_arc_px",
              "valley_cut_mm_tip", "valley_cut_mm_npoint", "qc_flags"]].to_string(index=False))

    bad = df[df["qc_flags"] != "OK"]
    print(f"\n{len(df)} recordings, {len(df.columns)} parameter columns.")
    print("QC: all validation rules hold." if bad.empty
          else f"QC FAILURES:\n{bad[['recording_id', 'qc_flags']].to_string(index=False)}")
    print(f"Wrote {tbl}\n      {dct}")
    return df


if __name__ == "__main__":
    main()
