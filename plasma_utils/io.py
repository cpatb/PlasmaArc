from pathlib import Path
from pycine.raw import read_frames


def load_cine(cine_path):
    """Returns (frames_list, fps)."""
    raw_images, setup, _ = read_frames(str(cine_path))
    fps = int(getattr(setup, "FrameRate", 0))
    return list(raw_images), fps
