"""
plasma_utils/io.py
------------------
I/O helpers for plasma-arc recordings.

Supported formats
~~~~~~~~~~~~~~~~~
* **.cine**   – Vision Research Phantom (via pycine)
* **.tiff / .tif** – Multi-page TIFF stacks as written by high-speed cameras
                     (e.g. Photron, Phantom export, or any ImageJ-compatible
                      multi-frame TIFF).  Single-page TIFFs are also accepted
                      and treated as a 1-frame sequence.

FPS detection order (TIFF)
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. ``ImageDescription`` tag → look for ``finterval=`` (ImageJ format) or
   ``fps=`` / ``framerate=`` key-value pairs.
2. TIFF ``XResolution`` / ``YResolution`` tags (some cameras encode timing
   there, though non-standard).
3. Fall back to *default_fps* argument (default 200 000).
"""

from pathlib import Path
from typing import Tuple, List

import numpy as np


# ---------------------------------------------------------------------------
# TIFF loader
# ---------------------------------------------------------------------------

def _parse_fps_from_imagej_description(desc: str) -> float | None:
    """Try to extract fps from an ImageJ / Micro-Manager ImageDescription."""
    import re
    # ImageJ stores 'finterval=<seconds_per_frame>'
    m = re.search(r'finterval\s*=\s*([\d.eE+\-]+)', desc, re.IGNORECASE)
    if m:
        finterval = float(m.group(1))
        if finterval > 0:
            return 1.0 / finterval
    # Some cameras write 'fps=<value>' or 'framerate=<value>'
    for key in ('fps', 'framerate', 'frame_rate'):
        m = re.search(rf'{key}\s*[=:]\s*([\d.eE+\-]+)', desc, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def load_tiff(tiff_path: Path, default_fps: float = 200_000) -> Tuple[np.ndarray, float]:
    """Load a (multi-page) TIFF stack.

    Parameters
    ----------
    tiff_path : Path
        Path to the .tiff / .tif file.
    default_fps : float
        Fallback frame rate when no metadata is found (default 200 000 Hz).

    Returns
    -------
    frames : np.ndarray, shape (N, H, W) or (N, H, W, C)
        All frames as a NumPy array. dtype is preserved from the file.
    fps : float
        Frames per second (from metadata or *default_fps*).
    """
    import tifffile

    with tifffile.TiffFile(str(tiff_path)) as tif:
        frames = tif.asarray()          # shape: (N,H,W) or (H,W) or (N,H,W,C)
        fps: float = default_fps

        # --- try ImageJ / OME metadata ---
        if tif.is_imagej:
            meta = tif.imagej_metadata or {}
            fi = meta.get('finterval')
            if fi and fi > 0:
                fps = 1.0 / fi
            elif 'fps' in meta:
                fps = float(meta['fps'])
        elif tif.ome_metadata:
            # OME-XML: look for PhysicalSizeT
            import re
            m = re.search(r'TimeIncrementUnit\s*=\s*"(\w+)"', tif.ome_metadata or '')
            m2 = re.search(r'TimeIncrement\s*=\s*"([\d.eE+\-]+)"', tif.ome_metadata or '')
            if m2:
                ti = float(m2.group(1))
                unit = m.group(1).lower() if m else 's'
                scale = {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'µs': 1e-6, 'ns': 1e-9}
                fps = 1.0 / (ti * scale.get(unit, 1))

        # --- fallback: scan ImageDescription tag of first IFD ---
        if fps == default_fps:
            for page in tif.pages[:1]:
                desc_tag = page.tags.get('ImageDescription')
                if desc_tag:
                    parsed = _parse_fps_from_imagej_description(str(desc_tag.value))
                    if parsed:
                        fps = parsed
                        break

    # Ensure (N, H, W) shape — squeeze single-frame files
    if frames.ndim == 2:
        frames = frames[np.newaxis, ...]          # (1, H, W)
    elif frames.ndim == 4 and frames.shape[-1] in (1, 3, 4):
        # colour TIFF — convert to grayscale luminance
        frames = frames[..., :3].mean(axis=-1).astype(frames.dtype)

    return frames, fps


# ---------------------------------------------------------------------------
# CINE loader  (unchanged — kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_cine(cine_path: Path) -> Tuple[list, float]:
    """Returns (frames_list, fps) from a Vision Research .cine file."""
    from pycine.raw import read_frames
    raw_images, setup, _ = read_frames(str(cine_path))
    fps = int(getattr(setup, 'FrameRate', 0))
    return list(raw_images), fps


# ---------------------------------------------------------------------------
# Generic loader (auto-dispatch)
# ---------------------------------------------------------------------------

def load_recording(path: Path, default_fps: float = 200_000) -> Tuple[np.ndarray, float]:
    """Load a plasma-arc recording regardless of format.

    Dispatches to :func:`load_tiff` or :func:`load_cine` based on extension.

    Returns
    -------
    frames : np.ndarray, shape (N, H, W)
    fps    : float
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in ('.tif', '.tiff'):
        return load_tiff(path, default_fps=default_fps)
    elif ext == '.cine':
        frames_list, fps = load_cine(path)
        return np.stack(frames_list, axis=0), float(fps)
    else:
        raise ValueError(f"Unsupported format: {ext!r}. Expected .tiff or .cine")
