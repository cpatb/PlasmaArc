import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, skeletonize, disk


def normalize_frame(frame):
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    if hi == lo:
        return np.zeros_like(f, dtype=np.uint8)
    return ((f - lo) / (hi - lo) * 255).astype(np.uint8)


def binarize(gray, close_r=3, open_r=2):
    """Per-frame Otsu threshold + morphological cleanup.

    Per-frame Otsu adapts to AC-driven intensity fluctuations automatically.
    """
    binary = gray > threshold_otsu(gray)
    binary = binary_closing(binary, disk(close_r))
    binary = binary_opening(binary, disk(open_r))
    return binary


def measure(binary):
    """Return (area_px, arc_length_px) from a binary frame.

    Arc length is the total skeleton pixel count (morphological thinning).
    """
    area = int(np.sum(binary))
    arc_length = int(np.sum(skeletonize(binary)))
    return area, arc_length
