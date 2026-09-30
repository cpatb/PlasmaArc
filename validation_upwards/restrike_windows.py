"""Completeness audit of leading-tip re-strikes: a person clicks every re-strike on a detector-independent streak
image, and the pipeline's re-strikes in the same window are matched to the clicks. Plan and decision rule:
E:\\BladeReactor_Upwards_analysis\\restrike_windows\\WINDOWS_PLAN.md (written before any window was drawn).

Streak image: column = one time bin of BIN frames; row = image row; value = max over the bin's frames of the row-wise
max of the background-subtracted, defect-corrected frame. Nothing from the detector is used to make it.

usage: python restrike_windows.py preview <analysis_root> <run_stem> <start_frame> <n_frames>   (display check only)
       python restrike_windows.py build   <analysis_root>
       python restrike_windows.py grade   <analysis_root> <clicks.csv>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cine_frames as cf  # noqa: E402

SEED = 20260930
CONDITIONS = [("Air", 2250), ("Air", 5875), ("Air", 8594), ("Argon", 2250), ("Argon", 5875), ("Argon", 8594),
              ("CO2", 3156), ("CO2", 5875), ("CO2", 8594), ("Ar-CO2(1_1)", 3156), ("Ar-CO2(1_1)", 5875),
              ("Ar-CO2(1_1)", 8594)]
TARGET_EVENTS, T_MIN_S = 15, 0.050
MAX_WIDTH_PX = 6000
ROW_SCALE = 2          # each image row drawn twice as tall
AXIS_PX = 22
MATCH_FRAMES = 100
N_BOOT = 10000
READ_BLOCK = 2000
GAMMA = 0.35          # strong stretch so the dim arc top (apex) is visible next to the bright feet


def run_dir(root: Path, stem: str) -> Path:
    return next((root / "runs_annotated").rglob(f"{stem}/run_metadata.json")).parent


def streak(root: Path, stem: str, i0: int, n: int):
    """(image uint8 array, bin, meta) for frames [i0, i0+n)."""
    rd = run_dir(root, stem)
    meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
    src = cf.open_cine(meta["cine"], measure=True, background=str(rd / "background_b10.npy"))
    b = max(1, math.ceil(n / MAX_WIDTH_PX))
    n = (n // b) * b
    rows = []
    for a in range(i0, i0 + n, READ_BLOCK):
        blk = src.block(a, min(READ_BLOCK, i0 + n - a)).astype(np.float32)
        rows.append(blk.max(axis=2))                       # (frames, H): brightest pixel in each row
    R = np.concatenate(rows)                               # (n, H)
    S = R.reshape(n // b, b, -1).max(axis=1).T             # (H, n/b)
    vmax = max(float(np.percentile(S, 99.5)), 1.0)
    img = (255 * np.clip(S / vmax, 0, 1) ** GAMMA).astype(np.uint8)
    return img, b, dict(meta=meta, gap_row=float(meta["calibration"]["gap_row"]), fps=float(meta["fps"]))


def to_png(img: np.ndarray, b: int, gap_row: float, fps: float, path: Path):
    """Rows scaled by ROW_SCALE; blue gap-row ticks every 250 px; time axis (ms from window start) below."""
    H, W = img.shape
    rgb = np.repeat(np.repeat(img, ROW_SCALE, axis=0)[:, :, None], 3, axis=2)
    canvas = Image.new("RGB", (W, H * ROW_SCALE + AXIS_PX), "white")
    canvas.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(canvas)
    gy = int(round(gap_row * ROW_SCALE))
    for x in range(0, W, 250):
        d.line([(x, gy), (x + 12, gy)], fill=(60, 150, 220), width=2)
    ms_per_px = b / fps * 1e3
    for k in range(0, int(W * ms_per_px) + 1):
        x = int(round(k / ms_per_px))
        top = H * ROW_SCALE
        d.line([(x, top), (x, top + (8 if k % 5 == 0 else 4))], fill="black")
        if k % 5 == 0:
            d.text((x + 2, top + 8), f"{k} ms", fill="black")
    canvas.save(path)


def preview(root: Path, stem: str, i0: int, n: int):
    img, b, m = streak(root, stem, i0, n)
    out = root / "restrike_windows" / "preview"
    out.mkdir(parents=True, exist_ok=True)
    to_png(img, b, m["gap_row"], m["fps"], out / f"preview_{stem}_{i0}.png")
    print(f"bin {b} frames, image {img.shape}")


def draw_windows(root: Path):
    thr = root / "tip_restrike_check" / "threshold"
    K = pd.read_csv(thr / "tip_restrikes_kept_by_condition.csv").set_index(["gas", "Re"])
    F = pd.read_csv(root / "aggregate_annotated" / "per_recording.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for gas, re_ in CONDITIONS:
        runs = F[(F["gas"] == gas) & (F["Re"] == re_) & (F["status"] == "DONE")].sort_values("trial")
        r = runs.iloc[int(rng.integers(len(runs)))]
        nfr, fps = int(r["n_frames"]), float(r["fps"])
        rate = float(K.loc[(gas, re_), "rate_per_s"])
        n = int(round(min(max(TARGET_EVENTS / rate, T_MIN_S), nfr / fps) * fps))
        i0 = int(rng.integers(0, nfr - n + 1))
        rows.append(dict(gas=gas, Re=re_, run=r["stem"], trial=int(r["trial"]), i0=i0, n=n, rate_counted=rate))
    W = pd.DataFrame(rows)
    W = W.iloc[rng.permutation(len(W))].reset_index(drop=True)
    W.insert(0, "code", [f"W{i + 1:02d}" for i in range(len(W))])
    return W


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Re-strike windows</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}}
.win{{border-top:2px solid #333;padding:10px 0 18px}} .sc{{overflow-x:auto;border:1px solid #bbb}}
.wr{{position:relative;display:inline-block;cursor:crosshair}} .wr img{{display:block}}
.mk{{position:absolute;top:0;width:3px;margin-left:-1px;background:rgba(255,40,40,.85);cursor:pointer}}
#bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #999;z-index:5}}
</style></head><body>
<div id="bar"><b>Re-strike windows</b> ({n}). <span id="tot"></span>
<button onclick="exportCsv()">Export clicks CSV</button></div>
<p>Each image is height (vertical, gap = blue ticks near the bottom) against time (horizontal, ms below). A gliding
arc draws a trace that climbs. <b>Click once on every re-strike</b>: where the trace jumps back down, or where a new
trace starts at the gap. Click a red marker to remove it. Scroll sideways inside each image. Clicks are saved in this
browser; export at the end.</p>
{wins}
<script>
const W={winjs};
function k(c){{return 'win_'+c}}
function get(c){{try{{return JSON.parse(localStorage.getItem(k(c))||'[]')}}catch(_){{return []}}}}
function put(c,a){{try{{localStorage.setItem(k(c),JSON.stringify(a))}}catch(_){{}}draw()}}
function draw(){{let t=0;for(const w of W){{const wr=document.getElementById('wr_'+w.code),img=wr.querySelector('img');
 wr.querySelectorAll('.mk').forEach(e=>e.remove());const a=get(w.code);t+=a.length;
 document.getElementById('n_'+w.code).textContent=a.length+' clicked';
 for(const x of a){{const m=document.createElement('div');m.className='mk';m.style.left=(x*img.clientWidth/img.naturalWidth)+'px';
  m.style.height=img.clientHeight+'px';m.onclick=ev=>{{ev.stopPropagation();put(w.code,get(w.code).filter(v=>v!==x))}};wr.appendChild(m)}}}}
 document.getElementById('tot').textContent=t+' clicks in total'}}
for(const w of W){{const wr=document.getElementById('wr_'+w.code);wr.addEventListener('click',ev=>{{
 const img=wr.querySelector('img'),r=img.getBoundingClientRect();
 const x=Math.round((ev.clientX-r.left)*img.naturalWidth/img.clientWidth);
 if(ev.clientY-r.top>img.clientHeight-{axis})return;const a=get(w.code);a.push(x);put(w.code,a)}})}}
window.addEventListener('load',draw);
function exportCsv(){{let s='code,x_px\\n';for(const w of W)for(const x of get(w.code).sort((p,q)=>p-q))s+=w.code+','+x+'\\n';
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));
 a.download='restrike_window_clicks.csv';a.click()}}
</script></body></html>"""


def build(root: Path):
    out = root / "restrike_windows"
    if (out / "windows_key.csv").exists():
        raise SystemExit(f"{out / 'windows_key.csv'} exists: windows already drawn")
    W = draw_windows(root)
    (out / "img").mkdir(parents=True, exist_ok=True)
    bins = []
    for _, w in W.iterrows():
        img, b, m = streak(root, w["run"], int(w["i0"]), int(w["n"]))
        to_png(img, b, m["gap_row"], m["fps"], out / "img" / f"{w['code']}.png")
        bins.append(b)
        print(f"  {w['code']}: {w['n']} frames, bin {b}, width {img.shape[1]} px", flush=True)
    W["bin"] = bins
    W["n"] = (W["n"] // W["bin"]) * W["bin"]
    W.to_csv(out / "windows_key.csv", index=False)
    wins = "\n".join(f'<div class="win"><b>{c}</b> <span id="n_{c}"></span><div class="sc"><div class="wr" id="wr_{c}">'
                     f'<img src="img/{c}.png"></div></div></div>' for c in W["code"])
    (out / "windows.html").write_text(PAGE.format(n=len(W), wins=wins, axis=AXIS_PX,
                                                  winjs=json.dumps([{"code": c} for c in W["code"]])), encoding="utf-8")
    print(f"wrote {out / 'windows.html'}; windows_key.csv is sealed until grading")


def match(clicks: np.ndarray, events: np.ndarray):
    """Greedy one-to-one nearest matching within MATCH_FRAMES; returns a bool per click."""
    pairs = sorted((abs(int(c) - int(e)), i, j) for i, c in enumerate(clicks) for j, e in enumerate(events)
                   if abs(int(c) - int(e)) <= MATCH_FRAMES)
    used_c, used_e = set(), set()
    for _, i, j in pairs:
        if i not in used_c and j not in used_e:
            used_c.add(i)
            used_e.add(j)
    return np.array([i in used_c for i in range(len(clicks))]), len(events) - len(used_e)


def grade(root: Path, clicks_csv: Path):
    from statsmodels.stats.proportion import proportion_confint
    W = pd.read_csv(root / "restrike_windows" / "windows_key.csv")
    Cl = pd.read_csv(clicks_csv)
    E = pd.read_csv(root / "tip_restrike_check" / "threshold" / "tip_restrike_events_kept.csv")
    rows, hits = [], {}
    for _, w in W.iterrows():
        c = Cl.loc[Cl["code"] == w["code"], "x_px"].to_numpy(int)
        cf_ = w["i0"] + c * w["bin"] + w["bin"] // 2                       # click -> centre frame of its bin
        ev = E[(E["run"] == w["run"]) & (E["frame_min"] >= w["i0"]) & (E["frame_min"] < w["i0"] + w["n"])]
        mk, unk = match(cf_, ev.loc[ev["kept"], "frame_min"].to_numpy())
        ma, _ = match(cf_, ev["frame_min"].to_numpy())
        hits[w["code"]] = mk
        H, M = len(c), int(mk.sum())
        lo, hi = proportion_confint(M, H, method="wilson") if H else (np.nan, np.nan)
        rows.append(dict(code=w["code"], gas=w["gas"], Re=w["Re"], run=w["run"], window_ms=w["n"] / 200.0, H=H,
                         kept_in_window=int(ev["kept"].sum()), M_kept=M, kept_unmatched=unk,
                         M_any_detector=int(ma.sum()), C=M / H if H else np.nan, C_lo=lo, C_hi=hi,
                         measured=H >= 5, rate_counted=w["rate_counted"]))
    T = pd.DataFrame(rows).sort_values(["gas", "Re"])
    T["rate_corrected"] = np.where(T["measured"], T["rate_counted"] / T["C"], np.nan)
    print(T.drop(columns=["run"]).round(3).to_string(index=False))
    rng = np.random.default_rng(SEED)
    for gas, d in T.groupby("gas"):
        d = d.sort_values("Re")
        lo_, hi_ = d.iloc[0], d.iloc[-1]
        if not (lo_["measured"] and hi_["measured"]):
            print(f"{gas}: rise NOT MEASURED (H < 5 at an end)")
            continue
        def boot(code):
            h = hits[code]
            return rng.choice(h, (N_BOOT, len(h)), replace=True).mean(1)
        cl, ch = boot(lo_["code"]), boot(hi_["code"])
        with np.errstate(divide="ignore"):
            ratio = (hi_["rate_counted"] / ch) / (lo_["rate_counted"] / cl)
        q = np.nanpercentile(ratio, [2.5, 97.5])
        pt = hi_["rate_corrected"] / lo_["rate_corrected"]
        verdict = "QUOTABLE rise" if q[0] > 1 else ("QUOTABLE fall" if q[1] < 1 else "no resolved Re dependence")
        print(f"{gas}: corrected rate Re{lo_['Re']} {lo_['rate_corrected']:.0f} /s -> Re{hi_['Re']} "
              f"{hi_['rate_corrected']:.0f} /s, ratio {pt:.2f} [{q[0]:.2f}, {q[1]:.2f}] -> {verdict} "
              f"(counted-rate ratio {hi_['rate_counted'] / lo_['rate_counted']:.2f})")
    T.to_csv(root / "restrike_windows" / "windows_graded.csv", index=False)


if __name__ == "__main__":
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == "preview":
        preview(root, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "build":
        build(root)
    elif cmd == "grade":
        grade(root, Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
