"""
figstyle.py — paper-grade matplotlib defaults + saver for the robustness pass.

Every figure produced by this pass goes through ``apply()`` for the rcParams
and ``save()`` for the actual write. ``save()`` writes both a vector (PDF) and
a 300-DPI PNG and a sidecar ``.txt`` caption that records the provenance for
the figure (recording stem, fps, px/mm, threshold, N, sigma budget, seed)
without that information having to live in the figure itself.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler


CIVIDIS_DISCRETE = ['#00204d', '#1d4f7a', '#5d7588', '#979c91', '#cfc28f', '#fde725']


def apply():
    """Set rcParams for the paper-grade standard (§5)."""
    plt.rcParams.update({
        'font.size':         10,
        'axes.labelsize':    11,
        'axes.titlesize':    11,
        'legend.fontsize':    9,
        'xtick.labelsize':    9,
        'ytick.labelsize':    9,
        'figure.dpi':       120,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'image.cmap':       'cividis',
        'axes.prop_cycle':  cycler(color=CIVIDIS_DISCRETE),
        'pdf.fonttype':     42,    # TrueType — fonts editable in vector output
        'ps.fonttype':      42,
    })


def save(fig, stem, caption, out_dir=None):
    """Write ``<stem>.pdf``, ``<stem>.png``, and ``<stem>.txt`` (caption sidecar).

    Parameters
    ----------
    fig : matplotlib Figure
    stem : str
        Filename without extension (or absolute path without extension).
    caption : str
        Multi-line caption written verbatim to ``<stem>.txt``.
    out_dir : str | Path | None
        Where to put the files. If None and ``stem`` is absolute, uses that
        directory; otherwise cwd.
    """
    p = Path(stem)
    if out_dir is not None:
        p = Path(out_dir) / p.name
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf = p.with_suffix('.pdf')
    png = p.with_suffix('.png')
    txt = p.with_suffix('.txt')
    fig.savefig(pdf)
    fig.savefig(png)
    txt.write_text(caption.strip() + '\n', encoding='utf-8')
    plt.close(fig)
    return pdf, png, txt


def verify(stem, out_dir=None):
    """Assert that the three sidecar files exist and the PNG is non-empty."""
    p = Path(stem)
    if out_dir is not None:
        p = Path(out_dir) / p.name
    pdf = p.with_suffix('.pdf'); png = p.with_suffix('.png'); txt = p.with_suffix('.txt')
    assert pdf.exists() and pdf.stat().st_size > 0, pdf
    assert png.exists() and png.stat().st_size > 0, png
    assert txt.exists() and txt.stat().st_size > 0, txt
    return True


def caption_block(*, recording, fps, px_per_mm, bright_thresh, n,
                  sigma_pxmm_rel, sigma_fps_rel, seed=None, extra=''):
    """Standardised caption block matching the brief §5 sidecar requirement."""
    bits = [
        f'recording        : {recording}',
        f'fps              : {int(fps)}',
        f'px/mm            : {float(px_per_mm):.4f}',
        f'bright_thresh    : {int(bright_thresh)}',
        f'N                : {int(n)}',
        f'sigma_pxmm_rel   : {float(sigma_pxmm_rel):.4f}',
        f'sigma_fps_rel    : {float(sigma_fps_rel):.5f}',
    ]
    if seed is not None:
        bits.append(f'random seed      : {seed}')
    if extra:
        bits.append('')
        bits.append(extra.strip())
    return '\n'.join(bits)
