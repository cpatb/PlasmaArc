"""plasma_utils/roi.py — ROI polygon mask utility."""
from pathlib import Path
import numpy as np
from skimage.draw import polygon as sk_polygon


def load_roi_mask(roi_path, shape):
    """Rasterise a saved napari polygon into a boolean mask.

    Parameters
    ----------
    roi_path : str or Path  — path to the .npy file of (N, 2) row/col vertices
    shape    : (H, W)       — output mask shape

    Returns
    -------
    bool ndarray of *shape*, or None if the file does not exist.
    """
    p = Path(roi_path)
    if not p.exists():
        return None
    pts = np.load(p)                                  # (N, 2): row, col
    rr, cc = sk_polygon(
        pts[:, 0].clip(0, shape[0] - 1),
        pts[:, 1].clip(0, shape[1] - 1),
        shape=shape,
    )
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask
