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

import plot_config

# Re-exported for callers that referenced figstyle.CIVIDIS_DISCRETE directly.
CIVIDIS_DISCRETE = plot_config.CIVIDIS_DISCRETE


def apply():
    """Set rcParams for the paper-grade standard, sourced from plot_config."""
    s = plot_config.STYLE
    o = plot_config.OUTPUT
    plt.rcParams.update({
        'font.size':         s['font_size'],
        'axes.labelsize':    s['axes_label_size'],
        'axes.titlesize':    s['axes_title_size'],
        'legend.fontsize':   s['legend_size'],
        'xtick.labelsize':   s['tick_size'],
        'ytick.labelsize':   s['tick_size'],
        'figure.dpi':        s['screen_dpi'],
        'savefig.dpi':       o['png_dpi'],
        'savefig.bbox':      'tight' if o['tight_bbox'] else 'standard',
        'axes.spines.top':   not s['hide_top_right'],
        'axes.spines.right': not s['hide_top_right'],
        'image.cmap':        s['colormap'],
        'axes.prop_cycle':   cycler(color=CIVIDIS_DISCRETE),
        'pdf.fonttype':      42 if s['embed_fonts'] else 3,
        'ps.fonttype':       42 if s['embed_fonts'] else 3,
    })


def _ext_paths(stem, out_dir=None):
    """Append .pdf/.png/.txt to ``stem`` literally (NOT via with_suffix, which
    breaks on stems containing dots like ``warmup_fit_HS_0.4mm_...``)."""
    p = Path(stem)
    if out_dir is not None:
        p = Path(out_dir) / p.name
    base = str(p)
    return Path(base + '.pdf'), Path(base + '.png'), Path(base + '.txt')


def save(fig, stem, caption, out_dir=None):
    """Write ``<stem>.pdf``, ``<stem>.png``, and ``<stem>.txt`` (caption sidecar).

    Parameters
    ----------
    fig : matplotlib Figure
    stem : str
        Filename without extension (or absolute path without extension). May
        contain dots — they are preserved literally (NOT treated as a
        suffix), unlike ``Path.with_suffix``.
    caption : str
        Multi-line caption written verbatim to ``<stem>.txt``.
    out_dir : str | Path | None
        Where to put the files. If None and ``stem`` is absolute, uses that
        directory; otherwise cwd.
    """
    pdf, png, txt = _ext_paths(stem, out_dir)
    o = plot_config.OUTPUT
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if o['save_pdf']:
        fig.savefig(pdf)
    if o['save_png']:
        fig.savefig(png)
    if o['save_caption_txt']:
        txt.write_text(caption.strip() + '\n', encoding='utf-8')
    plt.close(fig)
    return pdf, png, txt


def verify(stem, out_dir=None):
    """Assert that the three sidecar files exist and the PNG is non-empty."""
    pdf, png, txt = _ext_paths(stem, out_dir)
    o = plot_config.OUTPUT
    if o['save_pdf']:
        assert pdf.exists() and pdf.stat().st_size > 0, pdf
    if o['save_png']:
        assert png.exists() and png.stat().st_size > 0, png
    if o['save_caption_txt']:
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
