"""Two 20-s showcase videos of one BLADE-upwards recording (showcase_common.STEM), same 12 ms of the record:

  1. <stem>_raw.mp4      the camera frames (defect-corrected 12-bit DN, NOT background-subtracted), one fixed
                         grey map for the whole clip (no per-frame normalisation), a spatial scale bar and an
                         elapsed-time bar.
  2. <stem>_measured.mp4 the frame the detector sees (background-subtracted), the pipeline's cleaned arc mask
                         (hysteresis -> closing -> largest component; analyze_blade.binarize), its skeleton, the
                         leading tip, the two attached feet, the hand-annotated electrode edges, the minimum gap,
                         the gap reference row, the tip height, arc length, local tip velocity and kept tip
                         re-strikes, beside live traces of h, L and v with a cursor.

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
STRIP = 128                 # info strip under the image (time bar, labels)
GAMMA = 0.5                 # display only: DN/full-scale ** GAMMA, same for every frame
SCALE_MM = 2.0              # spatial scale bar length
BANNER_FRAMES = 60          # a re-strike label stays up for 60 source frames (0.3 ms = 0.5 s of video)
FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_B = r"C:\Windows\Fonts\arialbd.ttf"
COL = dict(edge=(255, 214, 0), gap=(255, 64, 255), tip=(255, 40, 40), foot=(60, 255, 90),
           mask=(0, 200, 255), mask_bad=(255, 150, 0), skel=(255, 255, 255), row=(170, 170, 170),
           A=(0, 220, 255), B=(255, 120, 200))


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


def draw_strip(canvas, t_ms, t_tot, frame, label, x0=40):
    """Elapsed-time bar: filled up to the current time, ticks every 1 ms, labelled every 2 ms."""
    h0 = canvas.shape[0] - STRIP
    canvas[h0:] = 0
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    W = canvas.shape[1] if canvas.shape[1] <= 1024 else 1024
    x1 = W - 40
    yb = h0 + 62
    dr.rectangle([x0, yb - 9, x1, yb + 9], outline=(200, 200, 200), width=2)
    xf = x0 + (x1 - x0) * t_ms / t_tot
    dr.rectangle([x0 + 2, yb - 7, max(x0 + 2, xf), yb + 7], fill=(255, 190, 40))
    for k in range(int(np.floor(t_tot)) + 1):
        xt = x0 + (x1 - x0) * k / t_tot
        dr.line([xt, yb + 9, xt, yb + (19 if k % 2 == 0 else 14)], fill=(200, 200, 200), width=2)
        if k % 2 == 0:
            _text(dr, (xt, yb + 21), f"{k} ms", font=font(18), fill=(200, 200, 200), anchor="ma")
    _text(dr, (x0, h0 + 14), f"elapsed  t = {t_ms:6.3f} ms", font=font(26, True), fill=(255, 255, 255))
    _text(dr, (x1, h0 + 18), f"frame {frame}", font=font(20), fill=(170, 170, 170), anchor="ra")
    _text(dr, (x0 + 330, h0 + 18), label, font=font(20), fill=(170, 170, 170))
    canvas[:] = np.asarray(im)


def scale_bar(dr, pxmm, x0=36, y0=1024 - 44):
    L = SCALE_MM * pxmm * UP
    dr.rectangle([x0, y0, x0 + L, y0 + 8], fill=(255, 255, 255))
    _text(dr, (x0 + L / 2, y0 - 6), f"{SCALE_MM:g} mm", font=font(22, True), fill=(255, 255, 255), anchor="md")


def trace_panel(d, s, E_win, fps, width, height):
    """Pre-rendered live-trace panel (house style) and the pixel mapping of its time axis."""
    C.style()
    dpi = 128
    fig, axs = plt.subplots(3, 1, figsize=(width / dpi, height / dpi), dpi=dpi, sharex=True)
    w = d.iloc[s:s + S.WIN]
    t = w["t_ms"].to_numpy() - w["t_ms"].iloc[0]
    for ax, col, lab in zip(axs, ["h_mm", "L_mm", "v_ms"],
                            ["Tip Height (mm)", "Arc Length (mm)", "Local Tip Velocity (m/s)"]):
        ax.plot(t, w[col], ".", ms=1.8, color="#2C7FB8", mec="none", rasterized=True)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.08 * np.nanmax(w[col]))
        for _, e in E_win.iterrows():
            ax.axvline((e["frame"] - s) / fps * 1e3, color="#253494" if e["mode"] == "A" else "#41B6C4",
                       ls="--" if e["mode"] == "A" else "-", lw=0.9, zorder=0)
    axs[-1].set_xlabel("Elapsed Time (ms)")
    axs[-1].set_xlim(0, S.WIN / fps * 1e3)
    for ax, k in zip(axs, "abc"):
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
    lab = f"Argon, Re 8594, trial 1 · {fps / 1e3:.0f} kfps · {fps / (S.STRIDE * S.FPS_OUT):.0f}x slower"
    panel, x_of, y_top, y_bot = trace_panel(d, s, E_win, fps, 1024, 1024 + STRIP)
    v1 = Encoder(S.OUT / f"{S.STEM}_raw.mp4", 1024, 1024 + STRIP)
    v2 = Encoder(S.OUT / f"{S.STEM}_measured.mp4", 2048, 1024 + STRIP)
    A, B = edges[0], edges[1]
    banner = None
    for i in range(0, S.WIN, S.STRIDE):
        f = s + i
        t_ms = i / fps * 1e3
        # restrike banner: an event in the source frames covered since the last output frame
        ev = E_win[(E_win["frame"] > f - S.STRIDE) & (E_win["frame"] <= f)]
        if len(ev):
            banner = (ev.iloc[-1], f)
        if banner is not None and f - banner[1] > BANNER_FRAMES:
            banner = None

        # ---------- video 1 ----------
        c1 = np.zeros((1024 + STRIP, 1024, 3), np.uint8)
        c1[:1024] = up(gray8(raw[i]))[..., None]
        im = Image.fromarray(c1)
        dr = ImageDraw.Draw(im)
        scale_bar(dr, pxmm)
        c1 = np.asarray(im).copy()
        draw_strip(c1, t_ms, t_tot, f, lab)
        v1.write(c1)

        # ---------- video 2 ----------
        mm = M[i]
        r = d.iloc[f]
        img = (up(gray8(sub[i])).astype(np.float32) * 0.55)[..., None].repeat(3, 2)
        msk = up(mm["_binary"].astype(np.uint8)) > 0
        if msk.any():
            colr = np.array(COL["mask"] if mm["arc_complete"] else COL["mask_bad"], np.float32)
            img[msk] = 0.45 * img[msk] + 0.55 * colr
        img = img.astype(np.uint8)
        sk = mm.get("_skel_coords")
        skm = np.zeros((256, 256), bool)
        if sk is not None and len(sk) and mm["_binary"].any():
            # the full skeleton (every branch: the arc length is skel.sum()), redrawn from the mask
            from skimage.morphology import skeletonize
            skm = skeletonize(mm["_binary"])
        ys, xs = np.nonzero(skm)
        for y, x in zip(ys, xs):
            cv2.rectangle(img, (x * UP + 1, y * UP + 1), (x * UP + UP - 2, y * UP + UP - 2), COL["skel"], -1)
        cv2.polylines(img, [np.array([P(p) for p in A[[0, -1]]])], False, COL["edge"], 3, cv2.LINE_AA)
        cv2.polylines(img, [np.array([P(p) for p in B[[0, -1]]])], False, COL["edge"], 3, cv2.LINE_AA)
        yg = int(round((gap_row + 0.5) * UP))
        for x in range(0, 1024, 24):
            cv2.line(img, (x, yg), (x + 12, yg), COL["row"], 1, cv2.LINE_AA)
        cv2.line(img, P(edges[2]), P(edges[3]), COL["gap"], 3, cv2.LINE_AA)
        for e in (edges[2], edges[3]):
            cv2.circle(img, P(e), 6, COL["gap"], -1, cv2.LINE_AA)
        complete = bool(mm["arc_complete"])
        tip = None
        if mm["_binary"].any():
            tip = (float(d.at[f, "arc_tip_row"]), float(d.at[f, "arc_tip_col"]))
            tip = ((m["H"] - 1) - tip[0], tip[1])                  # CSV row is flipped (distance from electrodes)
        if complete:
            for ft in (mm["tip_a"], mm["tip_b"]):
                cv2.circle(img, P(ft), 9, COL["foot"], 3, cv2.LINE_AA)
            xt, yt = P(tip)
            cv2.arrowedLine(img, (xt, yg), (xt, yt + 10), COL["tip"], 2, cv2.LINE_AA, tipLength=0.04)
        if tip is not None:
            cv2.circle(img, P(tip), 10, COL["tip"], -1 if complete else 2, cv2.LINE_AA)
        if banner is not None and tip is not None:
            cv2.circle(img, P(tip), 26,
                       COL[banner[0]["mode"]], 4, cv2.LINE_AA)

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        scale_bar(dr, pxmm)
        ga, gb = P(edges[2]), P(edges[3])
        _text(dr, (min(ga[0], gb[0]) - 24, max(ga[1], gb[1]) + 14), f"min gap {gap_px / pxmm:.2f} mm",
                font=font(22, True), fill=COL["gap"], anchor="ra")
        _text(dr, (8, yg - 6), "gap row (h = 0)", font=font(18), fill=COL["row"], anchor="ld")
        pa = P(A[len(A) // 2])
        pb = P(B[len(B) // 2])
        _text(dr, (pa[0] - 10, pa[1] + 10), "electrode A edge", font=font(20, True), fill=COL["edge"], anchor="ra")
        _text(dr, (pb[0] + 10, pb[1] + 10), "electrode B edge", font=font(20, True), fill=COL["edge"], anchor="la")
        if complete:
            xt, yt = P(tip)
            _text(dr, (xt + 16, (yg + yt) / 2), f"h = {r['h_mm']:.2f} mm", font=font(22, True), fill=COL["tip"],
                    anchor="lm")
            _text(dr, (xt + 34, yt - 10), "leading tip", font=font(20), fill=COL["tip"], anchor="ld")
        status = ("COMPLETE ARC (both feet attached)" if complete else
                  "arc not attached to both edges: excluded" if mm["_binary"].any() else "no arc above threshold")
        vtxt = "—" if np.isnan(r["v_ms"]) else f"{r['v_ms']:.2f} m/s"
        lines = [(status, (255, 255, 255) if complete else COL["mask_bad"]),
                 (f"tip height  h = {r['h_mm']:.2f} mm" if complete else "tip height  —", COL["tip"]),
                 (f"arc length  L = {r['L_mm']:.2f} mm" if complete else "arc length  —", COL["skel"]),
                 (f"local tip velocity  v = {vtxt}" if complete else "local tip velocity  —", (120, 220, 255)),
                 (f"mask area {mm['area_px']} px²", (170, 170, 170))]
        y = 14
        for txt, cl in lines:
            _text(dr, (14, y), txt, font=font(24, True) if y == 14 else font(23), fill=cl)
            y += 32
        if banner is not None:
            e = banner[0]
            nm = "Mode A (re-strike at the gap)" if e["mode"] == "A" else "Mode B (re-strike up the blades)"
            _text(dr, (1010, 1024 - 84), f"TIP RE-STRIKE · {nm}", font=font(24, True), fill=COL[e["mode"]], anchor="ra")
            _text(dr, (1010, 1024 - 52), f"tip fell {e['fall_mm']:.2f} mm, new tip at {e['tip_after_mm']:.2f} mm",
                    font=font(21), fill=COL[e["mode"]], anchor="ra")
        _text(dr, (1010, 92), "cyan: cleaned arc mask (orange: not attached)\nwhite: skeleton   green: attached feet\n"
                "red: leading tip   yellow: hand-annotated edges", font=font(18), fill=(200, 200, 200), anchor="ra",
                align="right")
        left = np.zeros((1024 + STRIP, 1024, 3), np.uint8)
        left[:1024] = np.asarray(im)
        draw_strip(left, t_ms, t_tot, f, lab)
        right = panel.copy()
        xc = int(round(x_of(t_ms)))
        cv2.line(right, (xc, int(y_top)), (xc, int(y_bot)), (220, 30, 30), 2, cv2.LINE_AA)
        v2.write(np.concatenate([left, right], axis=1))
        if i % 400 == 0:
            print(f"  frame {i}/{S.WIN}", flush=True)
    v1.close()
    v2.close()
    json.dump(dict(window_start=int(s), window_frames=S.WIN, stride=S.STRIDE, fps_out=S.FPS_OUT,
                   t0_ms=s / fps * 1e3, clip_ms=t_tot, slowdown=fps / (S.STRIDE * S.FPS_OUT), gamma=GAMMA,
                   remeasure_mismatches=len(bad), px_per_mm=pxmm, gap_px=gap_px,
                   restrikes_A=int((E_win["mode"] == "A").sum()), restrikes_B=int((E_win["mode"] == "B").sum())),
              open(S.OUT / "clip_info.json", "w"), indent=1)
    print(f"wrote {S.OUT}")


if __name__ == "__main__":
    main()
