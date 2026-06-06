# Plasma Arc Analysis — Methods

> **Living document.** Updated in lockstep with the codebase.  
> Last revised: 2026-05-11  
> Code modules: `plasma_utils/io.py`, `plasma_utils/image.py`, `plasma_utils/nephrogram.py`, `plasma_utils/qa.py`, `analyze_tiff.py`, `event_velocity.py`, `per_event_figures.py`

---

## Table of Contents

1. [Experimental Setup](#1-experimental-setup)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Loading & Preprocessing](#3-data-loading--preprocessing)
4. [Arc Detection (Thresholding & Morphology)](#4-arc-detection-thresholding--morphology)
5. [Skeletonisation & Path Ordering](#5-skeletonisation--path-ordering)
6. [Physical Calibration](#6-physical-calibration)
7. [Arc Measurements](#7-arc-measurements)
   - 7.1 [Arc Length](#71-arc-length)
   - 7.2 [Arc Width (FWHM)](#72-arc-width-fwhm)
   - 7.3 [Brightness Statistics](#73-brightness-statistics)
   - 7.4 [Geometric Descriptors](#74-geometric-descriptors)
   - 7.5 [Curvature](#75-curvature)
8. [Nephrogram Construction](#8-nephrogram-construction)
9. [Propagation Velocity Extraction (Radon — legacy)](#9-propagation-velocity-extraction-radon--legacy)
10. [Event-Based Velocity Analysis](#10-event-based-velocity-analysis)
    - 10.1 [Discharge Event Detection](#101-discharge-event-detection)
    - 10.2 [Advancing-Front Regression](#102-advancing-front-regression)
    - 10.3 [Sliding-Window Local Velocity](#103-sliding-window-local-velocity)
    - 10.4 [Mode A / Mode B Classification](#104-mode-a--mode-b-classification)
    - 10.5 [Per-Event Data Output](#105-per-event-data-output)
11. [Velocity Distribution Map](#11-velocity-distribution-map)
12. [Cycle Detection](#12-cycle-detection)
13. [QA Frame Rendering](#13-qa-frame-rendering)
14. [Output Files](#14-output-files)
15. [Known Limitations & Future Work](#15-known-limitations--future-work)
16. [References](#16-references)

---

## 1 Experimental Setup

| Parameter | Value | Notes |
|---|---|---|
| Gas | CO₂ | Pushed through nozzle during discharge |
| Minimum electrode gap | ~2.4 mm | Used for spatial calibration |
| Electrode opening angle | ~15 ° | Half-angle between electrode axes |
| Frame rate | ~200 000 fps | TIFF metadata or `--fps` argument |
| Bit depth | 16-bit (typical) | Normalised per frame to float32 [0,1] |

The CO₂ gas flow enables arc propagation by providing a conducting medium;
the arc repeatedly breaks down, extends downward (driven by gas flow and
electric field), and re-strokes at the electrode tips, producing the cyclic
propagation visible in the recordings.

---

## 2 Data Acquisition

Frames are stored as multi-page TIFF stacks.  Each page is one frame.
The loader (`plasma_utils/io.py : load_tiff`) reads metadata in the
following priority order to determine frame rate:

1. ImageJ `finterval` metadata key (seconds per frame)
2. ImageJ `fps` or `framerate` metadata key
3. OME-XML `TimeIncrement` and `TimeIncrementUnit`
4. `ImageDescription` tag free-text scan for `fps=` / `framerate=`
5. Default: 200 000 Hz (`--fps` CLI flag overrides at any point)

---

## 3 Data Loading & Preprocessing

**File:** `plasma_utils/io.py`

```
raw frame  (uint16, e.g. 4096×4096)
    │
    ▼  normalize_frame()
float32 [0, 1]   ← per-frame min/max stretch
```

Per-frame normalisation adapts to:
- AC-driven intensity fluctuations (arc brightness varies with discharge cycle)
- Sensor non-uniformity across the sequence
- Varying arc-to-background contrast between propagation phases

> **Note:** Per-frame normalisation means absolute brightness comparisons
> across frames require the unnormalised `integrated_intensity` metric
> (sum of raw pixel values), not `mean_intensity`.

---

## 4 Arc Detection (Thresholding & Morphology)

**File:** `plasma_utils/image.py : binarize()`

The arc occupies a very small bright region on a near-black background.
A global Otsu threshold computed on the full image histogram is biased
heavily toward the background mode and tends to over-threshold the arc.

### Percentile-filtered Otsu

We apply Otsu's method only to the upper 50th-percentile pixel values
(i.e. pixels above the per-frame median).  This concentrates the
threshold computation around the arc-intensity distribution.

```
gray[gray > percentile(gray, 50)]  → bright_pixels
threshold = threshold_otsu(bright_pixels)
binary = gray > threshold
```

### Morphological cleanup

| Operation | Kernel radius | Purpose |
|---|---|---|
| Closing (disk) | 3 px | Fills small gaps/breaks in the arc channel |
| Opening (disk) | 2 px | Removes isolated speckle noise |

### Largest-component selection

After morphological cleanup, the single largest connected component is
retained.  This rejects small satellite blobs from electrode glow or
reflections.

---

## 5 Skeletonisation & Path Ordering

**File:** `plasma_utils/image.py : _skeleton_ordered_coords()`

The arc binary mask is thinned to a 1-pixel-wide skeleton using
morphological skeletonisation (Zhang–Suen thinning via
`skimage.morphology.skeletonize`).

The skeleton pixels are then ordered into a connected path from one
endpoint to the other using a depth-first graph walk on the 8-connected
pixel adjacency graph.  Endpoints are defined as skeleton pixels with
exactly one 8-connected neighbour (degree = 1).

If no endpoints are found (closed-loop or isolated pixel), the walk
starts at an arbitrary skeleton pixel.

---

## 6 Physical Calibration

**File:** `plasma_utils/image.py : estimate_px_per_mm()`

Spatial scale is calibrated against the known minimum electrode gap
(2.4 mm).  The electrode gap in each frame is estimated as the Euclidean
distance between the two skeleton endpoints (arc tips):

```
gap_px[i] = √( (tip_b_row - tip_a_row)² + (tip_b_col - tip_a_col)² )
```

At electrical breakdown the arc is shortest, corresponding to the minimum
physical gap.  We take the **5th percentile** of `gap_px` across the
recording as the minimum:

```
px_per_mm = percentile(gap_px, 5) / electrode_gap_mm
```

Using a percentile (rather than the absolute minimum) guards against
single-frame outliers from noise.

An explicit pixel size can also be passed via `--pixel-size <mm/px>`.

---

## 7 Arc Measurements

### 7.1 Arc Length

**File:** `plasma_utils/image.py : measure_frame()`

Arc length is the **skeleton pixel count** multiplied by the pixel pitch:

```
arc_length_px  = count(skeleton pixels)
arc_length_mm  = arc_length_px / px_per_mm
```

This is an estimate of the contour length along the arc centreline.
Diagonal skeleton steps are not corrected to √2 (the error is small
for smooth arcs with gentle curvature).

### 7.2 Arc Width (FWHM)

**File:** `plasma_utils/image.py : _fwhm_perpendicular()`

At each sampled skeleton point (every 5 pixels by default), we extract
an intensity profile perpendicular to the local arc tangent.

The **local tangent** at point `i` is the central difference:

```
tangent[i] = path[i+1] - path[i-1]   (normalised)
```

The **perpendicular direction** is the tangent rotated 90 °.

A 41-pixel profile (±20 px either side) is sampled by bilinear
interpolation along the perpendicular.  The **full width at half maximum
(FWHM)** of this profile is returned as the local arc width.

```
width_px  = last_index(profile ≥ peak/2) - first_index(profile ≥ peak/2) + 1
width_mm  = width_px / px_per_mm
```

Per-frame statistics: **mean FWHM** and **median FWHM** over all sampled
skeleton points.

> The FWHM is a proxy for the physical arc channel diameter plus any
> optical blurring (diffraction + motion blur at 200 kfps).

### 7.3 Brightness Statistics

Computed on arc-masked pixels of the normalised frame:

| Metric | Formula | Physical meaning |
|---|---|---|
| `mean_intensity` | mean(gray[binary]) | Time-averaged emissivity proxy |
| `peak_intensity` | max(gray[binary]) | Hottest / most emissive arc point |
| `integrated_intensity` | sum(gray[binary]) | Total optical power proxy |

### 7.4 Geometric Descriptors

| Metric | Definition |
|---|---|
| Centroid (row, col) | Intensity-weighted mean position of arc pixels |
| Bounding box | Tight axis-aligned rectangle around the binary mask |
| Tip A, Tip B | First and last points on the ordered skeleton path |
| Electrode gap (px, mm) | Euclidean distance between tips |

### 7.5 Curvature

Mean absolute angular change along the skeleton:

```
angles[i] = arctan2(Δrow[i], Δcol[i])   for each step
δθ[i]     = wrap_to_π( angles[i+1] - angles[i] )
curvature  = mean(|δθ|)   [rad / px]
```

High curvature indicates a strongly bent arc (common mid-propagation
cycle when the arc bows downward under CO₂ flow).

---

## 8 Nephrogram Construction

**File:** `plasma_utils/nephrogram.py : build_nephrogram()`

A nephrogram is a **space–time kymograph**: a 2-D image in which
- **rows** correspond to time (frame index)
- **columns** correspond to spatial position along the projection axis

### Projection modes

For each frame, the full 2-D intensity field is reduced to a 1-D profile:

| Mode | Formula | Best for |
|---|---|---|
| `max` | column-wise maximum | Locating arc position (sparse bright object) |
| `mean` | column-wise mean | Integrated emissivity |
| `sum` | column-wise sum | Total optical power |

### Projection axes

| Axis | Spatial direction | Interpretation |
|---|---|---|
| `vertical` | Row axis (y) | Arc reaching downward; captures propagation depth |
| `horizontal` | Column axis (x) | Arc drifting laterally between electrodes |

### Post-processing

After stacking the 1-D profiles into a 2-D image:
1. Optional Gaussian smoothing (σ = 0.8 px by default) to suppress inter-frame jitter
2. Linear stretch to [0, 1]

### Reading the nephrogram

- **Bright horizontal band** at a given time row = arc present at that spatial position
- **Downward-sloping bright streak** = arc tip propagating downward over time
- The **slope** of bright features encodes propagation velocity:
  `v = Δ(spatial axis) / Δt`

---

## 9 Propagation Velocity Extraction (Radon — legacy)

**File:** `plasma_utils/nephrogram.py : radon_velocity()`

Propagation velocities are extracted from the nephrogram using the
**Radon transform** (equivalent to a projective Hough transform).

The nephrogram is treated as an image in (space × time) coordinates and
transposed to (space, time) for the Radon call.  For each projection
angle θ ∈ [−80°, 80°], the Radon sinogram column energy is computed.
The angle maximising column energy corresponds to the dominant bright
streak direction.

```
v_mm_s = cot(θ) × fps / px_per_mm
        = cos(θ)/sin(θ) × fps / px_per_mm
```

> **Convention note:** skimage's Radon transform uses θ measured from the
> *vertical* (space) axis of the (space × time) image.  θ = 0 captures
> vertical lines (instantaneous events); θ → 90° captures horizontal lines
> (stationary features).  A feature moving at v px/frame makes angle
> arctan(v) from the time axis = (90° − arctan(v)) from the space axis,
> so the Radon peak is at θ = 90° − arctan(v), giving v = cot(θ).
> Using tan(θ) would return the reciprocal of the true velocity.

Multiple velocity candidates are returned (top-N by energy), enabling
identification of co-propagating features (e.g., the propagating arc
tip and the stationary glow region).

---

## 10 Event-Based Velocity Analysis

**File:** `event_velocity.py`

The event-based pipeline supersedes the Radon approach for quantitative velocity measurements.  It operates on the per-frame `measurements_all.csv` produced by the TIFF analysis pipeline and produces event-level statistics plus per-frame local velocities.

### 10.1 Discharge Event Detection

A new primary discharge event is declared when the arc tip returns to within `NUCLEATION_SLACK_FRAC` (default 15%) of the effective arc range above the electrode row.  This is a data-driven threshold: it scales with the measured arc range for each recording rather than using a hard-coded pixel distance.

The electrode row is estimated as the 5th percentile of `arc_tip_row` from complete-arc frames (frames where the arc bridges both electrodes).  The effective arc range `L_eff_px` is then `p90(tip_rows) − electrode_row`, and the nucleation threshold is `electrode_row + 0.15 × L_eff_px`.

Within a primary event, intermediate retreats larger than 50% of the current arc extension are labelled **sub-events** — these correspond to partial re-ignition without full return to the gap (Fridman et al. 2005).  Detection gaps longer than 500 frames (~2.5 ms) always force a new primary event.

### 10.2 Advancing-Front Regression

For each event, only frames where the arc tip is at a new maximum row (the monotonic advancing envelope) are used for regression.  Retreating frames — including sub-event oscillations — are excluded.  A least-squares line is then fit to the advancing-front trajectory:

```
tip_row = slope × frame + intercept
velocity = slope [px/frame] × FPS / px_per_mm / 1000   →   m/s
```

R² is computed on the advancing-front frames.  An R² near 1 indicates a steady linear advance; lower values indicate irregular or multi-phase propagation.

Regression preferentially uses complete-arc frames (where the arc bridges both electrodes and the tip position is unambiguous).  When fewer than `MIN_EVENT_FRAMES` complete frames exist within an event, all detected frames are used instead.

### 10.3 Sliding-Window Local Velocity

A per-frame local velocity is computed for every advancing-front frame using a centered OLS window of `LOCAL_VEL_N = 10` advancing-front neighbors:

```
window = f_front[ max(0, k − 5) : min(n, k + 6) ]
local_vel_ms = slope(window) × FPS / px_per_mm / 1000
```

This is equivalent to a Savitzky-Golay first-derivative filter.  It provides time-resolved velocity that resolves within-event acceleration and deceleration while remaining resistant to frame-to-frame noise.  The local velocity is NaN for retreating frames and for events too short to fit.

### 10.4 Mode A / Mode B Classification

Gliding arcs exhibit two distinct propagation modes (Czernichowski 1994; Zhang et al. 2020):
- **Mode A** — the arc extinguishes completely and re-nucleates at the minimum electrode gap.
- **Mode B** — the previous arc channel persists partially and the next event re-ignites (back-breakdown) at an intermediate position away from the gap.

Classification is based on a dimensionless **normalised nucleation position**:

```
norm_nucleation_pos = (tip_row_start − electrode_row) / L_eff_px
```

This ratio is in [0, 1] and requires no spatial calibration.  Events with `norm_nucleation_pos > NUCLEATION_SLACK_FRAC` are labelled Mode B; all others are Mode A.  The threshold is self-consistent with the event detector: an event the detector considers "returned to the gap" has `norm_nucleation_pos ≤ 0.15` by construction.

The frames-since-last-event gap (`inter_event_gap_frames`) is stored as a secondary observable.  Mode B events are expected to have shorter gaps (the previous channel has not fully cooled before back-breakdown), but this is not used for primary classification.

### 10.5 Arc Shape Tracking and Normal Velocity

**File:** `arc_shape.py`

The arc is a deformable filament, not a rigid body.  The tip velocity measured
by the advancing-front regression captures only the leading edge; the rest of
the arc body can flex, bow, and sweep laterally independently.  `arc_shape.py`
measures this full-body deformation by tracking N equally-spaced points along
the arc skeleton frame-to-frame.

**Arc parameterization** — For each detected frame, the ordered skeleton path
is resampled to N points equally spaced by arc length using linear interpolation:

```
s_i = i × L_total / (N − 1),   i = 0, 1, ..., N − 1
```

Point 0 is always the electrode-proximal end (smallest row), so the
parameterisation is consistent across frames within an event.

**Velocity decomposition** — For each consecutive frame pair (t, t+1) within
the same event, the displacement at each arc point is decomposed into components
parallel and perpendicular to the local arc tangent:

```
tangent_i(t)  = (p_{i+1}(t) − p_{i−1}(t)) / |...|     (central difference)
normal_i(t)   = tangent rotated 90°
v_normal_i    = dot(p_i(t+1) − p_i(t), normal_i(t)) / Δt
v_tangential_i = dot(p_i(t+1) − p_i(t), tangent_i(t)) / Δt
```

The tangential component at the arc tip (i = N−1) approximates the tip velocity
from `event_velocity.py`, but is computed on ALL consecutive frame pairs
(not only the advancing front), so it also captures retreating motion.
The normal component is physically distinct: it measures lateral sweep driven
by gas flow and electromagnetic forces.

Only complete-arc frames (`arc_complete == True`) are used.  These are frames
where the arc bridges both electrodes, giving reliable skeleton endpoints and a
well-defined path from electrode to tip.  Partial-arc frames are excluded because
the skeleton is truncated at one end, making the arc-length parameterisation and
normal decomposition unreliable.

**Output** — `arc_shape_velocities.csv` with columns:
`file, frame_t, frame_t1, event_id, seg_idx, norm_pos, v_normal_ms, v_tangential_ms, arc_length_px`

where `norm_pos = seg_idx / (N − 1)` is the normalized arc-length position.

### 10.6 Per-Event Data Output

The pipeline writes a structured `events/` directory alongside the combined CSVs:

```
events/
    events_index.csv          — one row per event: velocity, R², duration,
                                 mode, norm_nucleation_pos, sub_event_count
    event_NNNN_frames.csv     — all frame-level rows for event NNNN
    event_NNNN_dynamics.csv   — advancing-front local velocities for NNNN
```

Per-event figures (two-panel: arc-tip trajectory + local velocity profile) can be generated with `per_event_figures.py events/`.

---

## 11 Velocity Distribution Map

**File:** `plasma_utils/nephrogram.py : velocity_distribution_map()`

To capture time-resolved velocity information, a sliding temporal window
(default 50 frames wide, 10-frame hop) is applied to the nephrogram.
In each window, the top-30 Radon velocity candidates are histogrammed by
their velocity magnitude, weighted by their spectral energy.

The resulting 2-D map (time × velocity) shows how the **distribution of
propagation velocities** evolves through the recording.  This is
equivalent to a time-resolved power spectrum of the spatial velocity field.

---

## 12 Cycle Detection

**File:** `plasma_utils/nephrogram.py : detect_cycles()`

Arc propagation in CO₂ is cyclic: the arc breaks down at the electrode
gap, propagates (extends) driven by the gas flow, and re-strokes.  Each
cycle is visible as a repeating pattern in the nephrogram.

To detect cycles:
1. A 1-D temporal signal is computed by averaging nephrogram intensity
   over the central 20% of the spatial axis (where propagation is most
   pronounced).
2. `scipy.signal.find_peaks` is applied with a minimum prominence of
   0.5 × signal standard deviation and a minimum inter-peak distance
   corresponding to ≥ 10 µs (at 200 kfps = 2 frames).
3. Peak frame indices → cycle periods → mean frequency.

---

## 13 QA Frame Rendering

**File:** `plasma_utils/qa.py : render_qa_frame()`

QA frames are annotated visualisations of selected frames that allow
rapid manual inspection of the detection and measurement quality.

### Overlays

| Overlay | Colour | Description |
|---|---|---|
| Arc mask fill | Magenta (α = 0.25) | Thresholded arc region |
| Skeleton | Cyan dots | 1-px-wide centreline |
| Width sample lines | Orange | ±12 px perpendicular segments at sampled skeleton points |
| Arc tips | Yellow circles | Skeleton endpoints (electrode contact proxies) |
| Electrode gap | Yellow dashed line | Straight line connecting tips |
| Centroid | Lime cross (+) | Intensity-weighted arc centroid |
| Bounding box | White dashed rectangle | Tight axis-aligned bounding rectangle |

### Text panel

Frame number, time in µs, arc length, mean FWHM width, brightness
statistics, electrode gap, and curvature are printed in a monospace
overlay in the top-left corner.

### Display gamma

The base image is displayed with γ = 0.5 (square-root stretch) to make
the dim arc region visible while preserving dynamic range for overlays.
This is a display-only adjustment; all measurements use the linear
normalised values.

---

## 14 Output Files

| File | Description |
|---|---|
| `measurements.csv` | Per-frame table: all ArcMeasurement fields + time |
| `summary_dashboard.png` | 6–10 panel figure: time series, nephrogram, velocity map |
| `nephrogram_vertical.png` | Vertical-axis nephrogram with calibrated axes |
| `nephrogram_horizontal.png` | Horizontal-axis nephrogram (if `--nephrogram-axis both`) |
| `velocity_map.png` | Time-resolved velocity distribution map |
| `qa_frames/qa_NNNNNN.png` | Annotated QA frames (every `--qa-interval` frames) |
| `METHODS.md` | This document |

### CSV column reference

| Column | Unit | Description |
|---|---|---|
| `frame` | — | Frame index (0-based) |
| `time_s` | s | Frame timestamp |
| `arc_detected` | bool | Was an arc successfully detected? |
| `area_px` | px² | Binary arc mask area |
| `arc_length_px` | px | Skeleton pixel count |
| `arc_length_mm` | mm | Physical arc length |
| `arc_width_px_mean` | px | Mean FWHM across skeleton samples |
| `arc_width_mm_mean` | mm | Physical mean FWHM |
| `mean_intensity` | — | Mean norm. intensity of arc pixels |
| `peak_intensity` | — | Max norm. intensity in arc |
| `integrated_intensity` | — | Sum of norm. intensity in arc |
| `centroid_row/col` | px | Intensity-weighted centroid |
| `tip_a/b_row/col` | px | Skeleton endpoint coordinates |
| `electrode_gap_px/mm` | px / mm | Distance between tips |
| `curvature_mean` | rad/px | Mean skeleton angular change |
| `bbox_*` | px | Bounding box corners |

---

## 15 Known Limitations & Future Work

- **Motion blur:** At 200 kfps with a bright arc, individual frames may
  show motion blur.  The FWHM width metric therefore includes a blurring
  contribution proportional to arc tip velocity × exposure time.
  Future: deblur using the known tip velocity from the nephrogram.

- **Otsu bias in very dim frames:** During the inter-stroke dark phase,
  Otsu may segment noise.  The largest-component filter mitigates this,
  but `arc_detected = False` frames should be cross-checked manually.

- **Single-electrode-gap calibration:** Physical scale is derived from
  one known distance (electrode gap at breakdown).  Any lens distortion
  across the field of view is not corrected.  Future: calibration grid.

- **Velocity wrap-around:** The Radon search is limited to ±80°.
  Very fast propagation events (steep streaks in the nephrogram) may
  alias to the boundary.  Extend `angle_range_deg` if needed.

- **Colour / multi-channel TIFF:** The loader converts colour TIFFs to
  grayscale by luminance averaging.  Spectroscopic multi-band recordings
  should be analysed channel-by-channel with the `load_tiff` return value.

---

## 16 References

1. Otsu, N. (1979). "A threshold selection method from gray-level histograms."
   *IEEE Trans. Syst. Man Cybern.* **9**(1), 62–66.

2. Zhang, T. Y., & Suen, C. Y. (1984). "A fast parallel algorithm for thinning
   digital patterns." *Commun. ACM* **27**(3), 236–239.

3. Deans, S. R. (1983). *The Radon Transform and Some of Its Applications.*
   Wiley.

4. van der Walt, S. et al. (2014). "scikit-image: image processing in Python."
   *PeerJ* **2**, e453.

5. Lieberman, M. A., & Lichtenberg, A. J. (2005).
   *Principles of Plasma Discharges and Materials Processing.* Wiley.

6. Czernichowski, A. (1994). "Gliding arc: applications to engineering and environment
   control." *Pure and Applied Chemistry* **66**(6), 1301–1310.

7. Zhang, H., et al. (2020). "Optical study of gliding arc discharge modes."
   *IET High Voltage* **5**(4), 386–392.

8. Fridman, A., Gutsol, A., & Kennedy, L. A. (2005). "Comparison of non-equilibrium
   plasma species in different electrical discharges." In *Non-Equilibrium Air Plasmas
   at Atmospheric Pressure.* IOP Publishing.

---

*End of document.*
