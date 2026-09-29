"""
cine_frames.py -- read Phantom .cine frames directly (no TIFF conversion), with
a fixed per-recording defect-pixel map.

Why this exists
---------------
The BLADE pipeline used to convert every .cine to 8-bit TIFF chunks first. That
costs disk the machine does not have (114 upward recordings = 565 GB of cine),
and it had two silent defects:

* the frame count came from ``filesize / unpacked_frame_bytes``; on 10-bit
  PACKED cines (81 920 B/frame on disk vs 131 072 B unpacked) that analysed
  only 62.5 % of each recording (run_11may26.probe_cine);
* the 8-bit stretch used per-recording lo/hi percentiles, so a fixed
  ``--bright-thresh`` meant a different detector brightness in every file.

This module decodes frames exactly as ``pycine.raw.create_raw_array`` does
(same 10-bit unpack, same pycine ``linLUT``, same ``np.interp`` onto 0..4095 and
the same truncating cast), so the DN scale is pycine's 12-bit linearised scale
for every recording, and the frame count is the header's ``ImageCount``.

Defect pixels
-------------
The camera (Phantom serial 21598) has a fixed lattice of 128 pixels stuck at
full scale (4095 after linearisation, zero temporal variance): rows 1,9,...,121
x cols 11,27,...,123 in pycine orientation. Measured identical in all 114 upward
recordings (census 2026-09-28). METHODS_BLADE.md s2.1 documents the same grid on
the 11MAY26 recordings.

The old correction (Drop Impact ``correct_bad_pixels_stack``) was a
data-adaptive 3x3 median: on a dark frame its MAD is ~0, so it replaced EVERY
pixel that differed from its local median -- a full median filter that can thin
1-2 px arc filaments. Here the defect set is measured once per recording from
frames spread across the whole record (``measure_defects``), and ONLY those
pixels are replaced, by the median of their non-defective 8-neighbours. Every
other pixel is returned bit-for-bit as pycine decodes it.

Orientation: pycine does NOT flip packed frames (it flips only unpacked ones).
In that orientation the upward BLADE rig is physically upright (electrodes at
the bottom, gap near row 233, arc rising). Alex's MATLAB LeadArc images are
flipped vertically relative to this (his gapRow 22.4 = 255 - 232.6).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

import numpy as np
from pycine.file import read_header
from pycine.linLUT import linLUT

# pycine create_raw_array, packed branch: linLUT then interp [64, 4064] -> [0, 4095]
_PYCINE_BLACK, _PYCINE_WHITE, _OUT_MAX = 64, 4064, 2 ** 12 - 1
LUT10 = np.interp(np.asarray(linLUT)[:1024], [_PYCINE_BLACK, _PYCINE_WHITE],
                  [0, _OUT_MAX]).astype(np.uint16)
DN_MAX = _OUT_MAX

# Defect detection: a pixel is STUCK if it never changes over the sampled frames
# while its neighbourhood does. 6 windows x 200 frames = 1200 frames spread over
# the record (the census used the same sampling).
DEFECT_WINDOWS = 6
DEFECT_WINDOW_FRAMES = 200


def _unpack10(buf: np.ndarray, n: int, h: int, w: int) -> np.ndarray:
    """Phantom 10-bit packed (big-endian, 4 px in 5 bytes) -> uint16 codes 0..1023.

    Same bit layout as pycine.raw.unpack_10bit, vectorised over n frames.
    """
    p = buf.reshape(n, -1, 5).astype(np.uint16)
    out = np.empty((n, p.shape[1], 4), np.uint16)
    out[..., 0] = (p[..., 0] << 2) | (p[..., 1] >> 6)
    out[..., 1] = ((p[..., 1] & 0x3F) << 4) | (p[..., 2] >> 4)
    out[..., 2] = ((p[..., 2] & 0x0F) << 6) | (p[..., 3] >> 2)
    out[..., 3] = ((p[..., 3] & 0x03) << 8) | p[..., 4]
    return out.reshape(n, h, w)


@dataclass
class DefectMap:
    mask: np.ndarray                      # bool (H, W), True = defective
    values: list = field(default_factory=list)   # distinct stuck values
    n_frames_sampled: int = 0

    @property
    def n(self) -> int:
        return int(self.mask.sum())

    def lattice_signature(self):
        rr, cc = np.nonzero(self.mask)
        return sorted(set(rr.tolist())), sorted(set(cc.tolist()))


class CineFrames:
    """Random-access frame source for a packed-10-bit Phantom cine.

    ``frame(i)`` / ``block(i0, n)`` return uint16 DN on pycine's 0..4095 scale,
    0-based frame indices, pycine orientation, defect pixels replaced if a
    DefectMap has been set (``self.defects``).
    """

    def __init__(self, path: str, defects: DefectMap | None = None):
        self.path = str(path)
        h = read_header(self.path)
        self.header = h
        ch, bi, s = h["cinefileheader"], h["bitmapinfoheader"], h["setup"]
        self.n = int(ch.ImageCount)
        self.first_image_no = int(ch.FirstImageNo)
        self.W, self.H = int(bi.biWidth), int(bi.biHeight)
        self.fps = float(s.FrameRate)
        self.shutter_ns = int(s.ShutterNs)
        self.real_bpp = int(s.RealBPP)
        self.serial = int(s.Serial)
        if int(bi.biCompression) == 0:
            raise NotImplementedError("unpacked cine: use pycine directly (not needed for BLADE upwards)")
        if int(bi.biBitCount) != 16 or self.real_bpp != 10:
            raise ValueError(f"unexpected packing biBitCount={bi.biBitCount} RealBPP={self.real_bpp}")
        self.img_bytes = self.W * self.H * 10 // 8
        pim = np.asarray(h["pImage"], np.int64)
        if pim.size != self.n:
            raise ValueError(f"pImage has {pim.size} entries, header ImageCount {self.n}")
        with open(self.path, "rb") as f:
            f.seek(int(pim[0]))
            ann = struct.unpack("I", f.read(4))[0]
            f.seek(int(pim[0]) + ann - 4)
            isz = struct.unpack("I", f.read(4))[0]
        if isz != self.img_bytes:
            raise ValueError(f"image_size {isz} != W*H*10/8 = {self.img_bytes}")
        stride = np.diff(pim)
        if stride.size and not np.all(stride == stride[0]):
            raise ValueError("non-uniform frame stride; per-frame seek reader required")
        self.stride = int(stride[0]) if stride.size else ann + isz
        if self.stride != ann + isz:
            raise ValueError(f"stride {self.stride} != annotation {ann} + image {isz}")
        need = int(pim[0]) + self.n * self.stride
        size = os.path.getsize(self.path)
        if size < need:
            raise ValueError(f"file truncated: {size} B < {need} B needed for {self.n} frames")
        self._mm = np.memmap(self.path, np.uint8, "r", offset=int(pim[0]),
                             shape=(self.n, self.stride))
        self._img_off = ann
        self.defects = defects

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.H, self.W)

    def raw_block(self, i0: int, n: int) -> np.ndarray:
        """Decoded frames i0..i0+n-1 WITHOUT defect correction (uint16, 0..4095)."""
        if i0 < 0 or n < 0 or i0 + n > self.n:
            raise IndexError(f"frames [{i0}, {i0 + n}) outside [0, {self.n})")
        buf = np.ascontiguousarray(self._mm[i0:i0 + n, self._img_off:self._img_off + self.img_bytes])
        return LUT10[_unpack10(buf, n, self.H, self.W)]

    def block(self, i0: int, n: int) -> np.ndarray:
        fr = self.raw_block(i0, n)
        if self.defects is not None and self.defects.n:
            fr = correct_defects(fr, self.defects.mask)
        bg = getattr(self, "background", None)
        if bg is not None:
            # background-subtracted DN, clipped at 0 (uint16 preserved)
            fr = np.clip(fr.astype(np.float32) - bg, 0, None).astype(np.uint16)
        return fr

    def set_background(self, bg: np.ndarray):
        bg = np.asarray(bg, np.float32)
        if bg.shape != (self.H, self.W):
            raise ValueError(f"background shape {bg.shape} != frame {(self.H, self.W)}")
        self.background = bg
        self._cache = None

    CACHE_FRAMES = 250    # sequential per-frame access decodes this many at once (was 1000: ~1 GB/worker
                          # with unpack temporaries; the harness reaped the batch for low memory 2026-09-29)

    def frame(self, i: int) -> np.ndarray:
        """Single frame. Sequential access (i == previous + 1) decodes CACHE_FRAMES ahead;
        random access decodes only the frame asked for."""
        c = getattr(self, "_cache", None)
        last = getattr(self, "_last", None)
        self._last = i
        if c is None or not (c[0] <= i < c[0] + len(c[1])):
            if not 0 <= i < self.n:
                raise IndexError(f"frame {i} outside [0, {self.n})")
            ahead = self.CACHE_FRAMES if last is not None and i == last + 1 else 1
            c = (i, self.block(i, min(ahead, self.n - i)))
            self._cache = c
        return c[1][i - c[0]]

    def iter_blocks(self, i0: int = 0, i1: int | None = None, size: int = 2000):
        i1 = self.n if i1 is None else min(i1, self.n)
        for a in range(i0, i1, size):
            b = min(a + size, i1)
            yield a, self.block(a, b - a)


def measure_defects(src: CineFrames, windows: int = DEFECT_WINDOWS,
                    window_frames: int = DEFECT_WINDOW_FRAMES) -> DefectMap:
    """Stuck pixels: zero temporal variance over frames spread across the record."""
    starts = np.linspace(0, src.n - window_frames, windows).astype(int)
    fr = np.concatenate([src.raw_block(int(a), window_frames) for a in starts])
    stuck = fr.min(0) == fr.max(0)
    vals = sorted(set(fr[0][stuck].astype(int).tolist()))
    return DefectMap(mask=stuck, values=vals, n_frames_sampled=int(fr.shape[0]))


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


_NB_CACHE: dict = {}


def _neighbour_table(mask: np.ndarray):
    """(defect flat idx, (n_def, 8) neighbour flat idx, (n_def, 8) valid) for a mask."""
    key = (mask.shape, mask.tobytes())
    if key in _NB_CACHE:
        return _NB_CACHE[key]
    H, W = mask.shape
    defs = np.argwhere(mask)
    nbi = np.zeros((len(defs), 8), np.int64)
    ok = np.zeros((len(defs), 8), bool)
    for k, (r, c) in enumerate(defs):
        for j, (dr, dc) in enumerate(_NB):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not mask[rr, cc]:
                nbi[k, j] = rr * W + cc
                ok[k, j] = True
        if not ok[k].any():
            raise ValueError(f"defect pixel ({r},{c}) has no good neighbour")
    out = (defs[:, 0] * W + defs[:, 1], nbi, ok)
    _NB_CACHE[key] = out
    return out


def correct_defects(frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace mask pixels by the median of their non-defective 8-neighbours.

    frames: (n, H, W) or (H, W) uint16. Returns a copy; non-mask pixels untouched.
    Median of an even count is the mean of the two middle values (numpy), rounded
    to nearest (half to even) and cast back to the input dtype.
    """
    single = frames.ndim == 2
    f = frames[None] if single else frames
    n = f.shape[0]
    flat = f.reshape(n, -1)
    didx, nbi, ok = _neighbour_table(mask)
    vals = flat[:, nbi].astype(np.float64)          # (n, n_def, 8)
    vals[:, ~ok] = np.nan
    med = np.nanmedian(vals, axis=2)
    out = f.copy()
    out.reshape(n, -1)[:, didx] = np.rint(med).astype(f.dtype)
    return out[0] if single else out


def open_cine(path: str, measure: bool = True, background: str | None = None) -> CineFrames:
    src = CineFrames(path)
    if measure:
        src.defects = measure_defects(src)
    if background is not None:
        src.set_background(np.load(background))
    return src


# ── background and noise (AMENDMENT 3, OVERNIGHT_LOG) ─────────────────────────
BG_SAMPLE = 2000     # frames spread evenly over the record
BG_QUANTILE = 10     # per-pixel temporal percentile taken as the static background
NOISE_Q = 99.99      # noise ceiling percentile of the residual on quiet pixels
QUIET_FRAC = 0.25    # quiet pixels: lowest quarter by temporal max above background
SAT_DN = 4078        # linearised full scale reached by saturated pixels (pycine LUT maps 1023 -> 4078)
SMEAR_Q = 80


def measure_background(src: "CineFrames", n: int = BG_SAMPLE):
    """Per-pixel temporal 10th percentile B10 (static glow + fixed-pattern offset) and the residual
    noise ceiling N = 99.99th pct of (frame - B10) over QUIET pixels (temporal p90 - p10 at or below
    its median over pixels), i.e. without needing frames that contain no arc."""
    idx = np.linspace(0, src.n - 1, n).astype(int)
    fr = np.stack([src.frame(int(i)) for i in idx]).astype(np.float32)
    b10 = np.percentile(fr, BG_QUANTILE, axis=0)
    # quiet = pixels whose temporal MAX (above their own background) is in the lowest QUIET_FRAC of
    # pixels: places the arc never reaches in the sampled frames.  (A p90-p10 spread was tried first
    # and FAILED: rarely-visited arc pixels count as quiet by it, N = 303 vs 33 on Air Re8594.)
    tmax = fr.max(0) - b10
    quiet = tmax <= np.quantile(tmax, QUIET_FRAC)
    resid = fr[:, quiet] - b10[quiet]
    N = float(np.percentile(resid, NOISE_Q))
    # smear ceiling S: readout smear runs to the RIGHT of saturated pixels along the row.  S = 80th
    # pct of the residual 10..60 px right of the last saturated pixel of each saturated row (80th,
    # not 99th: an arc leg can cross such a row segment, occupying a minority of it).
    seg = []
    for f in fr[fr.reshape(len(fr), -1).max(1) >= SAT_DN]:
        rr, cc = np.nonzero(f >= SAT_DN)
        for r in np.unique(rr):
            c0 = int(cc[rr == r].max())
            seg.append(f[r, c0 + 10:c0 + 61] - b10[r, c0 + 10:c0 + 61])
    seg = np.concatenate(seg) if seg else np.array([])
    S = float(np.percentile(seg, SMEAR_Q)) if seg.size >= 100 else 0.0
    return b10.astype(np.float32), N, dict(n_frames=int(n), quiet_px=int(quiet.sum()),
                                          b10_max=float(b10.max()), b10_median=float(np.median(b10)),
                                          quiet_tmax_max=float(tmax[quiet].max()),
                                          smear_S=S, smear_samples=int(seg.size),
                                          sat_frames=int((fr.reshape(len(fr), -1).max(1) >= SAT_DN).sum()))


class TiffFrames:
    """Same interface as CineFrames over a multipage TIFF (legacy chunked path)."""

    def __init__(self, path: str):
        import tifffile
        self.path = str(path)
        self._tf = tifffile.TiffFile(self.path)
        self.n = len(self._tf.pages)
        self.H, self.W = self._tf.pages[0].asarray().shape
        self.defects = None

    def __len__(self):
        return self.n

    @property
    def shape(self):
        return (self.H, self.W)

    def frame(self, i: int) -> np.ndarray:
        return self._tf.pages[i].asarray()

    def block(self, i0: int, n: int) -> np.ndarray:
        return np.stack([self.frame(i) for i in range(i0, i0 + n)])

    def close(self):
        self._tf.close()


def open_frames(path: str, background: str | None = None):
    """Frame source for a .cine (defect-corrected, 12-bit, optionally background-subtracted with the
    .npy from measure_background) or a TIFF (as stored)."""
    if str(path).lower().endswith(".cine"):
        return open_cine(path, background=background)
    if background is not None:
        raise ValueError("background subtraction is only implemented for .cine input")
    return TiffFrames(path)
