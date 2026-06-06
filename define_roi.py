#!/usr/bin/env python3
"""
define_roi.py  —  Draw a polygon ROI on a reference image using napari.

Run once before (or after) the pipeline to mark the valid region of the frame:
    python define_roi.py

The polygon is saved as roi_polygon.npy beside this script.
  - blade_run2.py  loads it automatically on the next pipeline run
    (re-run velmap + finalize phases to update the velocity map)
  - visualize_blade.py loads it automatically to mask figures immediately

Re-run this script at any time to update the ROI.
"""
import os, sys, glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_FILE   = os.path.join(BLADE_DIR, 'roi_polygon.npy')

_home     = os.path.expanduser('~')
BLADE_DIR = os.path.join(_home, 'OneDrive', 'Desktop', 'BLADE')
TIFF_FILE = os.path.join(BLADE_DIR, 'ADCB_0.8mmNoz_1000mL_N2_1.tiff')


def _load_background():
    """Best background for drawing: most recent velocity map, else max-proj of frames."""
    # look for any velocity map produced by the pipeline
    pattern = os.path.join(BLADE_DIR, '**', '*_velocity_map.npy')
    npy_files = sorted(glob.glob(pattern, recursive=True))
    if npy_files:
        vmap = np.load(npy_files[-1])
        vmap = np.nan_to_num(vmap, nan=0.0).astype(np.float32)
        lo, hi = np.nanpercentile(vmap[vmap > 0], [2, 98]) if (vmap > 0).any() else (0, 1)
        bg = np.clip((vmap - lo) / max(hi - lo, 1e-6), 0, 1)
        label = f'velocity map ({os.path.basename(npy_files[-1])})'
        return bg, label

    # fallback: max projection of 100 evenly-spaced raw frames
    import tifffile
    print(f'No velocity map found — loading reference frames from TIFF...')
    with tifffile.TiffFile(TIFF_FILE) as tif:
        n     = len(tif.pages)
        idxs  = np.linspace(0, n - 1, min(100, n), dtype=int)
        stack = np.stack([tif.pages[i].asarray() for i in idxs])
    ref = stack.max(axis=0).astype(np.float32)
    ref = (ref - ref.min()) / max(ref.max() - ref.min(), 1)
    return ref, 'max projection (100 frames)'


def main():
    try:
        import napari
    except ImportError:
        print('napari is not installed.  Run:  pip install "napari[all]"')
        sys.exit(1)

    bg, bg_label = _load_background()
    H, W = bg.shape[:2]
    print(f'Background: {bg_label}  ({H}×{W} px)')

    if os.path.exists(ROI_FILE):
        existing = np.load(ROI_FILE)
        print(f'Existing ROI loaded ({len(existing)} vertices) — edit or replace it.')
    else:
        existing = None
        print('No existing ROI.')

    viewer = napari.Viewer(title='Define ROI  —  close when done')
    viewer.add_image(bg, name=bg_label, colormap='plasma')
    shapes = viewer.add_shapes(
        data=[existing] if existing is not None else [],
        shape_type='polygon' if existing is not None else None,
        name='ROI',
        face_color=[1, 1, 1, 0.12],
        edge_color='cyan',
        edge_width=1,
    )

    print('\nInstructions:')
    print('  1. Select the Polygon tool  (P key, or the pentagon icon in the toolbar)')
    print('  2. Click to place vertices around the valid region')
    print('  3. Double-click to close the polygon')
    print('  4. Close the napari window when satisfied\n')

    napari.run()

    if len(shapes.data) == 0:
        print('No polygon drawn — ROI not saved.')
        return

    poly = shapes.data[0]   # first shape only
    np.save(ROI_FILE, poly)
    r0, r1 = poly[:, 0].min(), poly[:, 0].max()
    c0, c1 = poly[:, 1].min(), poly[:, 1].max()
    print(f'ROI saved: {ROI_FILE}')
    print(f'  {len(poly)} vertices  |  rows {r0:.0f}–{r1:.0f}  cols {c0:.0f}–{c1:.0f}')
    print()
    print('Next steps:')
    print('  - Figures: python visualize_blade.py    (ROI applied automatically)')
    print('  - Pipeline re-run (to update the velocity map):')
    print('      delete pipeline_state/f*_velmap* files, then python run_pipeline.py')


if __name__ == '__main__':
    main()
