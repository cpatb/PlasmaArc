"""Two 20-s showcase videos of one BLADE-upwards recording (showcase_common.STEM), same 12 ms of the record.
Layout is the author's (2026-10-01): nothing on the frame but a scale bar (upper left) and the elapsed time (upper
right); every run parameter goes to run_info.txt beside the videos.

  1. <stem>_raw.mp4      the camera frames (defect-corrected 12-bit DN, NOT background-subtracted), one fixed
                         grey map for the whole clip (no per-frame normalisation).
  2. <stem>_measured.mp4 the frame the detector sees (background-subtracted, dimmed), the pipeline's cleaned arc
                         mask (hysteresis -> closing -> largest component; analyze_blade.binarize), a red dot at
                         the leading tip and green dots at the two attached feet; beside live traces of tip height
                         and arc length with a cursor (velocity left out for now, author 2026-10-01).

Every frame is re-measured here with analyze_blade.measure_frame and the run's own parameters, and the result is
CHECKED against the run's measurements_all_events.csv (= measurements.csv plus velocity columns) on every frame of the clip (detection, completeness, area, arc length,
tip, feet). Any mismatch stops the script: the overlay is the pipeline's measurement, not a look-alike.
Velocity and re-strikes are not recomputed (they need the whole record); they are read from the run's
measurements_all_events.csv and tip_restrike_check/threshold/tip_restrike_events_kept.csv.

usage: python make_videos.py
"""
import json
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import showcase_common as S  # noqa: E402
import cine_frames as cf  # noqa: E402
import analyze_blade as ab  # noqa: E402
import _common as C  # noqa: E402

FFMPEG = r"C:\Users\compl\ffmpeg\bin\ffmpeg.exe"
UP = 4                      # 256 px -> 1024 px, nearest neighbour (pixels stay pixels)
GAMMA = 0.5                 # display only: DN/full-scale ** GAMMA, same for every frame
SCALE_MM = 2.0              # spatial scale bar length
FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_B = r"C:\Windows\Fonts\arialbd.ttf"
COL = dict(tip=(255, 40, 40), foot=(60, 255, 90), mask=(0, 200, 255), mask_bad=(255, 150, 0))


def _text(dr, *a, **k):
    """Text with a black outline so it stays legible over the arc and the bright mask."""
    dr.text(*a, stroke_width=3, stroke_fill=(0, 0, 0), **k)


def font(px, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, px)


def gray8(dn):
    return (np.clip(dn.astype(np.float32) / cf.SAT_DN, 0, 1) ** GAMMA * 255).astype(np.uint8)


def up(img):
    return cv2.resize(img, (img.shape[1] * UP, img.shape[0] * UP), interpolation=cv2.INTER_NEAREST)


def P(rc):
    """(row, col) in frame pixels -> (x, y) in the upscaled image, at the pixel CENTRE."""
    return (int(round((rc[1] + 0.5) * UP)), int(round((rc[0] + 0.5) * UP)))


def corners(canvas, t_ms, pxmm):
    """Author's layout (2026-10-01): scale bar upper LEFT, elapsed time upper RIGHT, nothing else on the frame."""
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    L = SCALE_MM * pxmm * UP
    x0, y0 = 32, 52
    dr.rectangle([x0, y0, x0 + L, y0 + 8], fill=(255, 255, 255))
    _text(dr, (x0 + L / 2, y0 - 6), f"{SCALE_MM:g} mm", font=font(24, True), fill=(255, 255, 255), anchor="md")
    _text(dr, (canvas.shape[1] - 24, 22), f"{t_ms:.3f} ms", font=font(30, True), fill=(255, 255, 255), anchor="ra")
    return np.asarray(im)


def trace_panel(d, s, E_win, fps, width, height):
    """Pre-rendered live-trace panel (house style) and the pixel mapping of its time axis."""
    C.style()
    dpi = 128
    fig, axs = plt.subplots(2, 1, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True)
    w = d.iloc[s:s + S.WIN]
    t = w["t_ms"].to_numpy() - w["t_ms"].iloc[0]
    for ax, col, lab in zip(axs, ["h_mm", "L_mm"], ["Tip Height (mm)", "Arc Length (mm)"]):
        ax.plot(t, w[col], ".", ms=2.2, color="#2C7FB8", mec="none", rasterized=True)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.08 * np.nanmax(w[col]))
        for _, e in E_win.iterrows():
            ax.axvline((e["frame"] - s) / fps * 1e3, color="#253494" if e["mode"] == "A" else "#41B6C4",
                       ls="--" if e["mode"] == "A" else "-", lw=0.9, zorder=0)
    axs[-1].set_xlabel("Elapsed Time (ms)")
    axs[-1].set_xlim(0, S.WIN / fps * 1e3)
    for ax, k in zip(axs, "ab"):
        C.panel(ax, k)
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([0], [0], color="#253494", ls="--", lw=0.9, label="Mode A (gap)"),
                           Line2D([0], [0], color="#41B6C4", lw=0.9, label="Mode B (blades)")],
                  title="Tip re-strike", loc="upper center", bbox_to_anchor=(0.42, 1.0))
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    H = img.shape[0]
    x_of = lambda tt: axs[0].transData.transform((tt, 0))[0]           # noqa: E731
    y_top = H - axs[0].get_window_extent().y1
    y_bot = H - axs[-1].get_window_extent().y0
    plt.close(fig)
    return img, x_of, y_top, y_bot


class Encoder:
    def __init__(self, path, w, h):
        self.p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-r", str(S.FPS_OUT), "-i", "-", "-c:v", "libx264",
                                   "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p", "-movflags",
                                   "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise SystemExit(f"ffmpeg rc {self.p.returncode}")


def same(a, b):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or a == b


def main():
    m = S.meta()
    det, cal = m["detector"], m["calibration"]
    cp = json.loads((S.RUN / "chunks" / "c00" / "chunk_params.json").read_text())
    pxmm, gap_row = float(cal["px_per_mm"]), float(cal["gap_row"])
    if not np.isclose(cp["px_per_mm"], pxmm, atol=1e-5):
        raise SystemExit("chunk px/mm != metadata px/mm")
    z = np.load(S.RUN / "calibration" / f"{S.STEM}_edges_annotated.npz")
    edges = (np.asarray(z["A"], float), np.asarray(z["B"], float), np.asarray(z["A_tip"], float),
             np.asarray(z["B_tip"], float))
    gap_px = float(np.hypot(*(edges[2] - edges[3])))
    print(f"px/mm {pxmm:.4f}, gap {gap_px:.2f} px = {gap_px / pxmm:.3f} mm, gap row {gap_row:.2f}, "
          f"T_low {det['T_low']}, T_high {det['T_high']}, low_frac {m['low_frac']}, tol {m['elec_tol_px']} px")

    d = S.frames_table(m)
    E = S.restrikes()
    s = S.pick_window(d, E)
    E_win = E[(E["frame"] >= s) & (E["frame"] < s + S.WIN)]
    fps = float(m["fps"])

    src = cf.open_cine(str(S.CINE), background=str(S.RUN / "background_b10.npy"))
    if src.n != int(m["n_frames"]) or src.defects.n != m["defects"]["n"]:
        raise SystemExit("cine header / defect map differ from the run")
    sub = src.block(s, S.WIN)                                    # what the detector saw
    raw = src.raw_block(s, S.WIN)
    raw = cf.correct_defects(raw, src.defects.mask)              # what the camera recorded (defects fixed)

    # ---- re-measure every frame with the run's parameters, check against measurements.csv ----
    M, bad = [], []
    keys = ["arc_tip_row", "arc_tip_col", "area_px", "arc_length_px"]
    for i in range(S.WIN):
        mm = ab.measure_frame(sub[i], det["close_r"], pxmm, ab.DEFAULT_TOP_FRACTION, keep_overlays=True,
                              dark_guard=det["T_high"], bright_thresh=det["T_high"], min_arc_px=det["min_arc_px"],
                              max_arc_px=10 ** 9, elec_tol_px=m["elec_tol_px"], arc_direction="up",
                              electrode_edges=edges, flash_eff_width=0, flash_arc_width=0,
                              low_thresh=det["T_low"], low_frac=m["low_frac"], npoint_n=m["npoint_n"])
        r = d.iloc[s + i]
        ok = (int(mm["arc_detected"]) == r["arc_detected"] and int(mm["arc_complete"]) == r["arc_complete"]
              and all(same(mm[k], r[k]) for k in keys))
        for ab_, nm in ((mm["tip_a"], "tip_a"), (mm["tip_b"], "tip_b")):
            if ab_ is not None:
                ok &= same(ab_[0], r[f"{nm}_row"]) and same(ab_[1], r[f"{nm}_col"])
        if not ok:
            bad.append(s + i)
        M.append(mm)
    print(f"re-measured {S.WIN} frames; mismatches with measurements.csv: {len(bad)}")
    if bad:
        raise SystemExit(f"mismatch at frames {bad[:10]}")

    S.OUT.mkdir(parents=True, exist_ok=True)
    w = d.iloc[s:s + S.WIN][["frame", "t_ms", "arc_detected", "arc_complete", "is_flash", "area_px",
                             "arc_length_px", "arc_tip_row", "arc_tip_col", "tip_a_row", "tip_a_col", "tip_b_row",
                             "tip_b_col", "h_mm", "L_mm", "v_ms"]].copy()
    w.insert(2, "t_clip_ms", (w["frame"] - s) / fps * 1e3)
    w.to_csv(S.OUT / "clip_per_frame.csv", index=False, float_format="%.6g")
    E_win.to_csv(S.OUT / "clip_restrikes.csv", index=False, float_format="%.6g")

    t_tot = S.WIN / fps * 1e3
    panel, x_of, y_top, y_bot = trace_panel(d, s, E_win, fps, 1024, 1024)
    v1 = Encoder(S.OUT / f"{S.STEM}_raw.mp4", 1024, 1024)
    v2 = Encoder(S.OUT / f"{S.STEM}_measured.mp4", 2048, 1024)
    for i in range(0, S.WIN, S.STRIDE):
        f = s + i
        t_ms = i / fps * 1e3
        # ---------- video 1: camera frame, scale bar, elapsed time ----------
        v1.write(corners(np.repeat(up(gray8(raw[i]))[..., None], 3, 2), t_ms, pxmm))

        # ---------- video 2: the cleaned arc mask, leading tip (red), attached feet (green) ----------
        mm = M[i]
        img = (up(gray8(sub[i])).astype(np.float32) * 0.55)[..., None].repeat(3, 2)
        msk = up(mm["_binary"].astype(np.uint8)) > 0
        complete = bool(mm["arc_complete"])
        if msk.any():
            colr = np.array(COL["mask"] if complete else COL["mask_bad"], np.float32)
            img[msk] = 0.45 * img[msk] + 0.55 * colr
        img = img.astype(np.uint8)
        if mm["_binary"].any():
            tip = ((m["H"] - 1) - float(d.at[f, "arc_tip_row"]), float(d.at[f, "arc_tip_col"]))  # CSV row is flipped
            if complete:
                for ft in (mm["tip_a"], mm["tip_b"]):
                    cv2.circle(img, P(ft), 9, COL["foot"], -1, cv2.LINE_AA)
            cv2.circle(img, P(tip), 10, COL["tip"], -1 if complete else 2, cv2.LINE_AA)
        right = panel.copy()
        xc = int(round(x_of(t_ms)))
        cv2.line(right, (xc, int(y_top)), (xc, int(y_bot)), (220, 30, 30), 2, cv2.LINE_AA)
        v2.write(np.concatenate([corners(img, t_ms, pxmm), right], axis=1))
        if i % 400 == 0:
            print(f"  frame {i}/{S.WIN}", flush=True)
    v1.close()
    v2.close()
    json.dump(dict(window_start=int(s), window_frames=S.WIN, stride=S.STRIDE, fps_out=S.FPS_OUT,
                   t0_ms=s / fps * 1e3, clip_ms=t_tot, slowdown=fps / (S.STRIDE * S.FPS_OUT), gamma=GAMMA,
                   remeasure_mismatches=len(bad), px_per_mm=pxmm, gap_px=gap_px,
                   restrikes_A=int((E_win["mode"] == "A").sum()), restrikes_B=int((E_win["mode"] == "B").sum())),
              open(S.OUT / "clip_info.json", "w"), indent=1)
    (S.OUT / "run_info.txt").write_text(f"""Recording      {S.CINE}
Gas / Re / trial  {m['gas']} / {m['Re']} / {m['trial']}   ({m['angle_deg']} deg blades, {m['gap_mm_folder']} mm gap)
Camera         {fps:.0f} fps, exposure {m['shutter_ns']} ns, {m['W']}x{m['H']} px, {m['real_bpp']}-bit, {m['n_frames']} frames ({m['n_frames'] / fps * 1e3:.1f} ms)
Clip           source frames {s}-{s + S.WIN - 1} (t = {s / fps * 1e3:.3f}-{(s + S.WIN) / fps * 1e3:.3f} ms of the record), {t_tot:.1f} ms
Video          every {S.STRIDE}nd source frame at {S.FPS_OUT} fps = {S.WIN // S.STRIDE / S.FPS_OUT:.1f} s, {fps / (S.STRIDE * S.FPS_OUT):.0f}x slower than real time
               elapsed time (upper right) is real time since the clip start; scale bar (upper left) {SCALE_MM:g} mm
Display        raw: defect-corrected DN, not background-subtracted, (DN/{cf.SAT_DN})^{GAMMA} grey, x{UP} nearest-neighbour
               measured: background-subtracted frame (55 % grey); cleaned arc mask cyan (complete arc) / orange
               (not attached to both edges, excluded); red dot = leading tip (filled when complete);
               green dots = attachment feet; right: tip height above the gap and arc length, re-strike lines by mode
Calibration    {pxmm:.4f} px/mm from the hand-annotated {cal['gap_mm']} mm gap ({gap_px:.2f} px); gap row {gap_row:.2f}
Detector       hysteresis on background-subtracted DN: T_low {det['T_low']}, T_high {det['T_high']}, low_frac {m['low_frac']},
               closing radius {det['close_r']} px, speckle floor {det['min_arc_px']} px, edge tolerance {m['elec_tol_px']} px
Check          all {S.WIN} clip frames re-measured with analyze_blade.measure_frame: {len(bad)} mismatches with the run CSV
Re-strikes     kept (fall >= {S.t_tip_mm()} mm) in the clip: Mode A {int((E_win['mode'] == 'A').sum())}, Mode B {int((E_win['mode'] == 'B').sum())} (valley {S.mode_valley_mm()} mm)
Run outputs    {S.RUN}
""", encoding="utf-8")
    print(f"wrote {S.OUT}")


if __name__ == "__main__":
    main()
