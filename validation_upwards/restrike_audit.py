"""Blind visual audit of leading-tip re-strikes. Plan, classes and decision rule: E:\\BladeReactor_Upwards_analysis\\
restrike_audit\\AUDIT_PLAN.md (written before the sample was drawn).

  build: draw the seeded stratified sample, render one strip of raw frames per event, write audit.html (verdict
         form, answers kept in the browser and exported as CSV), key.csv (sealed) and sample_meta.json.
  grade: read the exported verdicts, open the key, apply the pre-registered rule.

usage: python restrike_audit.py build <analysis_root>
       python restrike_audit.py grade <analysis_root> <verdicts.csv>
"""
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cine_frames as cf  # noqa: E402
from tip_restrike_check import load_tip  # noqa: E402

SEED = 20260930
T_TIP_MM, VALLEY_MM = 1.8, 2.2                # values at commit 82ef70e; checked against the files below
N_KA = {"Argon": 8, "Air": 8, "Ar-Air(1_1)": 8, "Ar-CO2(1_1)": 8, "CO2": 8}
N_KB = {"Argon": 12, "Air": 12, "Ar-Air(1_1)": 12, "Ar-CO2(1_1)": 4, "CO2": 0}
N_REJ = {g: 4 for g in N_KA}
OFFSETS_BEFORE, OFFSETS_AFTER, N_INSIDE = (-20, -5, 0), (0, 5, 20), 3
FDR_BAR = 0.15
CROP_PAD = 12


def draw(root: Path):
    thr = root / "tip_restrike_check" / "threshold"
    t_tip = float((thr / "T_TIP.txt").read_text().strip())
    val = float([ln for ln in (thr / "tip_modes.txt").read_text(encoding="utf-8").splitlines()
                 if ln.startswith("valley value used")][0].split(":")[1].split()[0])
    if not (np.isclose(t_tip, T_TIP_MM) and np.isclose(val, VALLEY_MM)):
        raise SystemExit(f"T_TIP {t_tip} / valley {val} differ from the registered {T_TIP_MM} / {VALLEY_MM}")
    E = pd.read_csv(thr / "tip_restrike_events_kept.csv")
    F = pd.read_csv(root / "aggregate_annotated" / "per_recording.csv").set_index("stem")
    E["gas"] = E["run"].map(F["gas"])
    E = E[E["n_steps"] > 0]                               # an event with no step has nothing to show
    E["cls"] = np.where(~E["kept"], "REJ", np.where(E["tip_after_mm"] > VALLEY_MM, "KB", "KA"))
    rng = np.random.default_rng(SEED)
    parts = []
    for cls, want in (("KA", N_KA), ("KB", N_KB), ("REJ", N_REJ)):
        for g, n in want.items():
            pool = E[(E["cls"] == cls) & (E["gas"] == g)]
            if n > len(pool):
                raise SystemExit(f"{cls}/{g}: want {n}, pool has {len(pool)}")
            if n:
                parts.append(pool.iloc[np.sort(rng.choice(len(pool), n, replace=False))])
    S = pd.concat(parts, ignore_index=True)
    S = S.iloc[rng.permutation(len(S))].reset_index(drop=True)
    S.insert(0, "code", [f"E{i + 1:03d}" for i in range(len(S))])
    print("sample:", S.groupby(["cls", "gas"]).size().to_dict())
    return S


def strip_frames(rd: Path, peak_frame: int, frame_min: int):
    """Cine frames to show: around the largest single-step fall between the peak and the re-strike."""
    _, pxmm, fr, tip, erow, _ = load_tip(rd)
    a, k = int(np.searchsorted(fr, peak_frame)), int(np.searchsorted(fr, frame_min))
    if fr[a] != peak_frame or fr[k] != frame_min or k <= a:
        raise RuntimeError(f"{rd.name}: peak/min frames not in the tip series")
    j = a + int(np.argmax(-np.diff(tip[a:k + 1])))
    s0, s1 = int(fr[j]), int(fr[j + 1])
    inside = [] if s1 - s0 <= 1 else sorted(set(np.linspace(s0, s1, N_INSIDE + 2)[1:-1].round().astype(int)))
    frames = [s0 + d for d in OFFSETS_BEFORE] + inside + [s1 + d for d in OFFSETS_AFTER]
    return frames, s0, s1


def render(S: pd.DataFrame, root: Path, out: Path):
    (out / "strips").mkdir(parents=True, exist_ok=True)
    rows, opened = [], {}
    for _, e in S.sort_values("run").iterrows():
        rd = next((root / "runs_annotated").rglob(f"{e['run']}/run_metadata.json")).parent
        meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
        if e["run"] not in opened:
            opened.clear()                                   # one cine open at a time
            opened[e["run"]] = cf.open_cine(meta["cine"], measure=True, background=str(rd / "background_b10.npy"))
        src = opened[e["run"]]
        frames, s0, s1 = strip_frames(rd, int(e["peak_frame"]), int(e["frame_min"]))
        frames = [f for f in frames if 0 <= f < len(src)]
        imgs = np.stack([src.frame(f).astype(np.float32) for f in frames])
        vmax = max(float(np.percentile(imgs, 99.8)), 1.0)
        gap_row = float(meta["calibration"]["gap_row"])
        # one crop for the whole strip: the box holding every pixel above 15 % of vmax in any frame, plus the gap
        # row, padded by CROP_PAD px (zooms onto the discharge; identical for all frames of the event)
        rr, cc = np.nonzero((imgs > 0.15 * vmax).any(0))
        H, W = imgs.shape[1:]
        r0 = max(int(min(rr.min(), gap_row)) - CROP_PAD, 0) if rr.size else 0
        r1 = min(int(max(rr.max(), gap_row)) + CROP_PAD, H - 1) if rr.size else H - 1
        c0 = max(int(cc.min()) - CROP_PAD, 0) if cc.size else 0
        c1 = min(int(cc.max()) + CROP_PAD, W - 1) if cc.size else W - 1
        pw = 2.2
        fig, axs = plt.subplots(1, len(frames), figsize=(pw * len(frames), pw * (r1 - r0 + 1) / (c1 - c0 + 1) + 0.5))
        for ax, f, im in zip(np.atleast_1d(axs), frames, imgs):
            ax.imshow(np.sqrt(np.clip(im[r0:r1 + 1, c0:c1 + 1] / vmax, 0, 1)), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            w = c1 - c0
            for x0, x1 in ((0, 0.06 * w), (0.94 * w, w)):
                ax.plot([x0, x1], [gap_row - r0, gap_row - r0], color="#39c", lw=1.2)
            tag = "before" if f <= s0 else ("inside break" if f < s1 else "after")
            ax.set_title(f"{tag}\n{(f - s0) * 5:+d} µs", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(e["code"], fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "strips" / f"{e['code']}.png", dpi=110)
        plt.close(fig)
        rows.append(dict(code=e["code"], frames=" ".join(map(str, frames)), s0=s0, s1=s1, break_frames=s1 - s0))
        print(f"  {e['code']}", flush=True)
    return pd.DataFrame(rows)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Re-strike audit</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;max-width:1500px;background:#fff;color:#111}}
.ev{{border-top:1px solid #ccc;padding:10px 0}} img{{max-width:100%}} label{{margin-right:14px}}
.q{{margin:4px 0}} #bar{{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #333}}
</style></head><body>
<div id="bar"><b>Re-strike audit</b>: {n} events. <span id="done"></span>
<button onclick="exportCsv()">Export verdicts CSV</button></div>
<p><b>Q1</b> R = real re-strike (the arc is replaced by a new, lower arc, or vanishes and a new arc appears lower);
N = not a re-strike (same arc continues: apex jitter, branch switch, artefact, flash); U = cannot tell.<br>
<b>Q2</b> (only if R) where the new arc forms: G = at or next to the minimum gap (blue ticks); P = partway up the
blades; ? = unclear. Answers are saved in this browser as you go; export when done.</p>
{events}
<script>
const codes={codes};
function key(c,q){{return 'audit_'+c+'_'+q}}
function load(){{let d=0;for(const c of codes){{for(const q of ['q1','q2']){{const v=localStorage.getItem(key(c,q));
 if(v){{const el=document.querySelector(`input[name=${{c}}_${{q}}][value="${{v}}"]`);if(el)el.checked=true}}}}
 const n=document.getElementById(c+'_note');n.value=localStorage.getItem(key(c,'note'))||'';
 if(localStorage.getItem(key(c,'q1')))d++}}document.getElementById('done').textContent=d+' / '+codes.length+' answered'}}
document.addEventListener('change',e=>{{const t=e.target;if(t.name){{const [c,q]=t.name.split('_');
 try{{localStorage.setItem(key(c,q),t.value)}}catch(_){{}}load()}}}});
document.addEventListener('input',e=>{{const t=e.target;if(t.id.endsWith('_note')){{try{{localStorage.setItem(key(t.id.split('_')[0],'note'),t.value)}}catch(_){{}}}}}});
function exportCsv(){{let s='code,q1,q2,note\\n';for(const c of codes){{const g=q=>localStorage.getItem(key(c,q))||'';
 s+=[c,g('q1'),g('q2'),'"'+g('note').replace(/"/g,'""')+'"'].join(',')+'\\n'}}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));
 a.download='restrike_audit_verdicts.csv';a.click()}}
load();
</script></body></html>"""


def radios(code, q, opts):
    return " ".join(f'<label><input type="radio" name="{code}_{q}" value="{html.escape(o)}">{html.escape(o)}</label>'
                    for o in opts)


def build(root: Path):
    out = root / "restrike_audit"
    if (out / "key.csv").exists():
        raise SystemExit(f"{out / 'key.csv'} exists: the sample is already drawn; delete the folder to redraw")
    S = draw(root)
    R = render(S, root, out)
    S = S.merge(R, on="code", validate="one_to_one")
    S.to_csv(out / "key.csv", index=False)
    ev = "\n".join(f'<div class="ev"><img src="strips/{c}.png" alt="{c}"><div class="q"><b>{c}</b> Q1: '
                   f'{radios(c, "q1", "RNU")} &nbsp; Q2: {radios(c, "q2", ["G", "P", "?"])} &nbsp; note: '
                   f'<input id="{c}_note" size="40"></div></div>' for c in sorted(S["code"]))
    (out / "audit.html").write_text(PAGE.format(n=len(S), events=ev, codes=json.dumps(sorted(S["code"].tolist()))),
                                    encoding="utf-8")
    (out / "sample_meta.json").write_text(json.dumps(dict(seed=SEED, t_tip_mm=T_TIP_MM, valley_mm=VALLEY_MM,
                                                          n=len(S), ka=N_KA, kb=N_KB, rej=N_REJ), indent=1))
    print(f"wrote {out / 'audit.html'} ({len(S)} strips); key.csv is sealed until grading")


def wilson_hi(k, n):
    from statsmodels.stats.proportion import proportion_confint
    return float(proportion_confint(k, n, method="wilson")[1]) if n else np.nan


def grade(root: Path, verdicts: Path):
    K = pd.read_csv(root / "restrike_audit" / "key.csv")
    V = pd.read_csv(verdicts, dtype=str).fillna("")
    D = K.merge(V, on="code", how="left", validate="one_to_one")
    miss = D["q1"].isin(["R", "N", "U"]).eq(False).sum()
    print(f"events {len(D)}, verdicts missing or invalid: {int(miss)}")
    for cls in ("KA", "KB", "REJ"):
        d = D[D["cls"] == cls]
        r, n, u = int((d["q1"] == "R").sum()), int((d["q1"] == "N").sum()), int((d["q1"] == "U").sum())
        hi = wilson_hi(n, r + n)
        verdict = "" if cls == "REJ" else ("TRUSTWORTHY" if hi <= FDR_BAR else "FAILS the bar")
        extra = (f"; R among rejected = {r}/{r + n} (real re-strikes the rule discards)" if cls == "REJ" else "")
        print(f"{cls}: R {r}, N {n}, U {u}; false fraction {n}/{r + n} = {n / max(r + n, 1):.3f}, "
              f"Wilson 95 % upper {hi:.3f} {verdict}{extra}")
        print("   per gas (descriptive): " + ", ".join(
            f"{g} R{int((x['q1'] == 'R').sum())}/N{int((x['q1'] == 'N').sum())}/U{int((x['q1'] == 'U').sum())}"
            for g, x in d.groupby("gas")))
    L = D[D["cls"].isin(["KA", "KB"]) & (D["q1"] == "R")]
    lab = L[L["q2"].isin(["G", "P"])]
    mism = int(((lab["cls"] == "KA") & (lab["q2"] == "P")).sum() + ((lab["cls"] == "KB") & (lab["q2"] == "G")).sum())
    hi = wilson_hi(mism, len(lab))
    print(f"Mode label: {mism} mismatches of {len(lab)} located R events (Q2 '?' or blank: {len(L) - len(lab)}); "
          f"Wilson 95 % upper {hi:.3f} {'TRUSTWORTHY' if hi <= FDR_BAR else 'FAILS the bar'}")
    D.to_csv(root / "restrike_audit" / "graded.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]))
    elif sys.argv[1] == "grade":
        grade(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
