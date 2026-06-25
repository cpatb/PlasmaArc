"""
plot_config.py — ONE place to customise every figure in the J-electrode pass.

If you want to change how the figures look (colours, sizes, fonts, output
formats, axis text) or *what* the cycle-time panel reports, edit the values in
this file. Nothing else needs touching — `figstyle.apply()` reads its rcParams
from here, and the figure scripts (`reynolds_comparison.py`,
`mode_ratio_vs_re.py`, `asymmetry_vs_time.py`, ...) pull their palette, figure
sizes and shared axis labels from here too.

Quick map of what lives where
------------------------------
  * Output formats (PDF/PNG/DPI, caption sidecars) ....... OUTPUT
  * Fonts, spines, colormap ............................. STYLE
  * Per-nozzle colours + the discrete cividis ramp ...... PALETTE
  * Figure sizes, keyed by figure stem .................. FIGSIZE
  * Error-bar marker appearance ......................... MARKER
  * Shared axis-label / caption text .................... LABELS
  * Cycle panel: duration (ms) vs frequency (Hz) ........ CYCLE

Anything that is genuinely per-run (px/mm, fps, bright_thresh, the σ budget)
is NOT here — that is measured per recording and threaded through the pipeline
CLI / run_metadata.json on purpose. This file is presentation only.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT — what gets written for every figure
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT = {
    'save_pdf':          True,    # vector copy (paper / editing)
    'save_png':          True,    # raster copy (slides / preview)
    'png_dpi':           300,     # raster resolution
    'save_caption_txt':  True,    # the .txt provenance sidecar next to each fig
    'tight_bbox':        True,
}


# ─────────────────────────────────────────────────────────────────────────────
# STYLE — global matplotlib look (consumed by figstyle.apply())
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    'font_size':          10,
    'axes_label_size':    11,
    'axes_title_size':    11,
    'legend_size':         9,
    'tick_size':           9,
    'screen_dpi':        120,     # figure.dpi (preview only; png_dpi controls files)
    'hide_top_right':    True,    # drop the top/right spines
    'colormap':          'cividis',
    'embed_fonts':       True,    # TrueType (editable) text in the PDF
}


# ─────────────────────────────────────────────────────────────────────────────
# PALETTE — colours
# ─────────────────────────────────────────────────────────────────────────────
# Discrete cividis ramp used for the default property cycle (colourblind-safe).
CIVIDIS_DISCRETE = ['#00204d', '#1d4f7a', '#5d7588', '#979c91', '#cfc28f', '#fde725']

# One colour per nozzle diameter (mm). This is the single source of truth — the
# cross-recording scripts all read NOZZLE_COLORS instead of redefining it.
NOZZLE_COLORS = {
    0.4: '#00204d',   # dark blue
    0.8: '#5d7588',   # slate
    1.2: '#fde725',   # yellow
}
DEFAULT_COLOR = '#979c91'   # used for any diameter not listed above


def nozzle_color(d_mm):
    """Colour for a nozzle diameter, falling back to DEFAULT_COLOR."""
    return NOZZLE_COLORS.get(float(d_mm), DEFAULT_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# FIGSIZE — width × height in inches, keyed by figure stem
# ─────────────────────────────────────────────────────────────────────────────
FIGSIZE = {
    'reynolds_arc_velocity':       (6.5, 4.2),
    'reynolds_mode_b_fraction':    (6.5, 4.2),
    'reynolds_cycle_time':         (6.5, 4.2),   # the cycle panel (see CYCLE)
    'mode_ratio_vs_re':            (7.2, 4.4),
    'asymmetry_vs_re':             (13.0, 4.0),
    'asymmetry_timeseries':        (9.5, 5.6),
    'warmup_summary':              (11.5, 4.4),
    'warmup_fit':                  (7.5, 4.4),
    'temporal_profiles':           (11.5, 4.4),
    'temporal_summary_xrecording': (10.0, 4.8),
}
DEFAULT_FIGSIZE = (6.5, 4.2)


def figsize(name):
    """Figure size for a named figure, falling back to DEFAULT_FIGSIZE."""
    return FIGSIZE.get(name, DEFAULT_FIGSIZE)


# ─────────────────────────────────────────────────────────────────────────────
# MARKER — error-bar marker appearance on the Re scatter plots
# ─────────────────────────────────────────────────────────────────────────────
MARKER = {
    'fmt':           'o',
    'size':          8,
    'capsize':       4,
    'edge_color':   '#000000',
    'edge_width':    0.6,
    'grid_alpha':    0.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# LABELS — shared axis / legend text (edit once, applies everywhere)
# ─────────────────────────────────────────────────────────────────────────────
LABELS = {
    'reynolds_x':     'Reynolds number (CO₂ at 295 K, 101 kPa)',
    'legend_title':   'Nozzle diameter',
    'arc_velocity_y': 'Median event arc velocity (m/s)',
    'mode_b_y':       'Mode B re-strike fraction',
}


# ─────────────────────────────────────────────────────────────────────────────
# CYCLE — how the re-strike cycling panel is reported
# ─────────────────────────────────────────────────────────────────────────────
# The arc re-strikes periodically. We can report that either as a period
# (how long one cycle lasts, in ms) or as a rate (how many per second, in Hz).
# Duration is the more intuitive read for most audiences, so it is the default.
#
#   'duration_ms'  -> cycle duration  = 1000 * cycle_frames / fps      [ms]
#   'frequency_hz' -> cycle frequency = fps / cycle_frames             [Hz]
#
# To switch back to the old frequency plot, set this to 'frequency_hz'.
CYCLE = {
    'metric':         'duration_ms',
    'duration_label': 'Mean re-strike cycle duration (ms)',
    'frequency_label':'Mean re-strike cycle frequency (Hz)',
    'duration_stem':  'reynolds_cycle_time',
    'frequency_stem': 'reynolds_cycle_frequency',
}
