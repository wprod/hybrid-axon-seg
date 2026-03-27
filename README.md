# hybrid-axon-seg for Marie 🩵

**Automated nerve morphometry pipeline.**

Detects myelinated axons in cross-sectional nerve images, segments the axon
and myelin compartments, and computes g-ratio, myelin thickness, axon/fiber
diameters, axon volume fraction (AVF), myelin volume fraction (MVF), N-ratio,
and axon density — with full QC overlay output.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Pipeline Architecture](#pipeline-architecture)
   - [Step 1 — Cellpose Pass 1: Outer Fibers](#step-1--cellpose-pass-1-outer-fibers)
   - [Step 2 — Normalized Inversion: Building `axon_input`](#step-2--normalized-inversion-building-axon_input)
   - [Step 3 — Global Otsu Axon Detection](#step-3--global-otsu-axon-detection)
   - [Step 4 — Morphometrics](#step-4--morphometrics)
   - [Step 5 — QC Filtering](#step-5--qc-filtering)
   - [Step 6 — Visualizations](#step-6--visualizations)
4. [Mathematics](#mathematics)
5. [Configuration Reference](#configuration-reference)
6. [Output Files](#output-files)
7. [Module Structure](#module-structure)
8. [Overlay Color Scheme](#overlay-color-scheme)
9. [References](#references)

---

## Overview

Toluidine-blue staining of semi-thin peripheral nerve cross-sections produces
images where **myelin sheaths appear dark** and **axon interiors appear bright**.
This pipeline automates the full morphometric analysis:

```
edited TIFF  →  Cellpose (outer fibers)  →  normalized inversion
             →  Otsu threshold (axon blobs)  →  morphometrics  →  QC  →  outputs
```

Key design decisions:

| Decision | Rationale |
|---|---|
| **Cellpose is the source of truth for fiber count** | Deep-learning segmentation handles touching fibers, irregular shapes, and staining variability better than any threshold-based approach |
| **Per-fiber normalization before inversion** | Each fiber is contrast-stretched independently, making the axon/myelin contrast invariant to global staining intensity gradients |
| **Distance-transform erosion at fiber boundary** | Strips endoneurium contamination introduced by Cellpose boundary pixels before any intensity measurement |
| **Global Otsu on the full `axon_input` image** | All normalized fiber pixels together form a clean bimodal distribution — one threshold for the entire image, no per-fiber fitting |
| **Crop-based morphometrics** | All per-fiber operations use small bounding-box crops instead of full-image masks, keeping memory and runtime linear in fiber count |
| **Two-level cache** | Cellpose results and axon detections are cached separately so detection parameters can be tuned without re-running Cellpose (~5 min on GPU) |
| **Fascicle area denominator** | AVF, MVF, N-ratio and density use the nerve fascicle area (convex Gaussian-smoothed envelope of the fiber mask) rather than total image area |

> **Image preparation:** contrast and brightness must be adjusted manually
> (with a medical eye) in ImageJ/Fiji before running the pipeline, as optimal
> settings vary per image and staining batch. Place corrected images in `edited/`.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.11+, CUDA or MPS recommended for Cellpose)
pip install -r requirements.txt

# 2. Place your contrast-adjusted TIFF images in the edited/ directory

# 3. Run the full pipeline on all images
python segment.py

# 4. Or process a single image
python segment.py "edited/MyImage.tif"
```

All parameters are in `config.py` — no code changes needed for routine use.

---

## Pipeline Architecture

### Step 1 — Cellpose Pass 1: Outer Fibers

**File:** `detection.py → run_cellpose_fibers()`

Cellpose ([Stringer et al., 2021](https://doi.org/10.1038/s41592-020-01018-x))
is a deep-learning instance segmentation model trained on a large corpus of
biological cell images. The `cyto3` model is used here to segment entire
myelinated fiber cross-sections (axon + myelin sheath together).

**How Cellpose works:**
Rather than detecting explicit boundaries, Cellpose uses a gradient-flow
formulation. It predicts, for every pixel, the 2-D vector pointing toward
the center of its containing cell. Starting from each pixel, these vectors
are integrated as a flow until convergence — pixels that converge to the
same attractor belong to the same cell. This approach is robust to touching
and overlapping objects, where classical watershed methods fail.

**Parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `CP_MODEL` | `cyto3` | Pretrained Cellpose 3 model |
| `CP_DIAM_UM` | `7.0 µm` | Expected fiber diameter (sets internal rescaling) |
| `CP_FLOW_THR` | `0.4` | Flow error threshold — lower is more permissive |
| `CP_CELLPROB` | `0.0` | Cell probability threshold — lower detects more |

The diameter in pixels fed to Cellpose is:

$$d_{px} = \frac{\texttt{CP\_DIAM\_UM}}{\texttt{PIXEL\_SIZE}}$$

**Post-processing:**
- Each fiber mask is **eroded** by `OUTER_ERODE_PX` pixels (via distance transform) to shrink the fiber boundary and reduce myelin bleed into adjacent regions.
- **Isolated cluster removal** (`MAIN_CLUSTER_DILATION_PX`): fiber masks are dilated and grouped into connected components; any fiber whose centroid lies outside the largest component (the main nerve fascicle) is discarded. This removes parasitic fiber clusters at image edges.

**Output:** An integer label array where each pixel contains its fiber ID
(1 to N) or 0 for background. This label map is the **source of truth for
fiber count** and is cached as `*_cellpose_outer.npy`.

---

### Step 2 — Normalized Inversion: Building `axon_input`

**File:** `preprocessing.py → _invert_crop(), build_axon_input()`

This is the most critical preprocessing step. The goal is to transform each
fiber crop so that:

- **Axon interior** (originally bright in toluidine blue) → **dark blob ≈ 0**
- **Myelin sheath** (originally dark in toluidine blue) → **bright ring ≈ 255**
- **Background / endoneurium boundary** → **white = 255** (excluded from analysis)

This representation is then globally thresholded in Step 3 to cleanly separate
axons from myelin without any per-fiber fitting.

#### 2a — Distance-Transform Boundary Erosion

Cellpose masks include a boundary zone of ~1–3 px where the staining is
ambiguous (endoneurium, adjacent fiber contamination). If these pixels are
included in the normalization, bright boundary artefacts invert to near-black
and create false dark rings around every fiber.

To remove this zone, a **Euclidean distance transform** is computed on the
fiber mask:

$$D(x,y) = \min_{(x', y') \notin \text{fiber}} \|(x,y) - (x',y')\|_2$$

Only pixels with $D(x,y) > \texttt{MASK\_ERODE\_PX}$ form the **clean
interior** (`inner`). This is geometrically equivalent to morphological
erosion by `MASK_ERODE_PX` pixels, but the continuous distance values are
reused in the fade step below. Erosion depth is also capped at 25% of the
fiber equivalent radius to preserve signal in small or thin-myelin fibers.

#### 2b — Per-Fiber Percentile Stretch

Contrast is normalized independently for each fiber using the **5th–95th
percentile** of the clean interior pixels:

$$I_{\text{stretched}}(x,y) = \text{clip}\!\left(\frac{I(x,y) - p_5}{p_{95} - p_5} \cdot 255,\ 0,\ 255\right)$$

Using percentiles rather than min/max makes the stretch robust to isolated
bright or dark outlier pixels (artefacts, staining clumps). After this step,
every fiber uses the full 0–255 dynamic range regardless of absolute staining
intensity.

#### 2c — Inversion

$$I_{\text{inverted}}(x,y) = 255 - I_{\text{stretched}}(x,y)$$

After inversion: myelin (originally near 0) → near 255; axon (originally
near 255) → near 0.

#### 2d — Inward Fade (Gradient Blending)

To prevent any hard black ring at the eroded boundary, a **smooth inward
fade** blends from white (255) at the eroded edge to fully inverted at
`FADE_PX` pixels inward.

A second distance transform is computed on the `inner` mask:

$$D_{\text{inner}}(x,y) = \text{distance from pixel to the eroded boundary}$$

The fade weight:

$$\alpha(x,y) = \text{clip}\!\left(\frac{D_{\text{inner}}(x,y)}{\texttt{FADE\_PX}},\ 0,\ 1\right)$$

Final blended value:

$$I_{\text{result}}(x,y) = 255 + \bigl(I_{\text{inverted}}(x,y) - 255\bigr) \cdot \alpha(x,y)$$

- At the eroded boundary ($\alpha = 0$): result = **255** (white).
- At `FADE_PX` px inward ($\alpha = 1$): result = $I_{\text{inverted}}$ (fully inverted).
- In between: smooth linear interpolation.

All pixels outside `inner` (the boundary zone) are set to **255** so they
do not influence the Otsu threshold.

The resulting `axon_input` image is saved as `*_axon_input.png` for visual
debugging.

---

### Step 3 — Global Otsu Axon Detection

**File:** `detection.py → find_axons()`

#### 3a — Global Otsu Threshold

Otsu's method ([Otsu, 1979](https://doi.org/10.1109/TSMC.1979.4310076))
finds the threshold $t^*$ that **minimises intra-class variance**
(equivalently, maximises inter-class variance) over all fiber pixels:

$$t^* = \arg\max_t \; \omega_0(t)\,\omega_1(t)\,\bigl[\mu_0(t) - \mu_1(t)\bigr]^2$$

where $\omega_0(t), \omega_1(t)$ are the proportions of pixels below / above $t$,
and $\mu_0(t), \mu_1(t)$ are their respective means.

**Why global (not per-fiber)?**
Because `axon_input` has already been *per-fiber* normalized, all axon pixels
across the entire image cluster around values ~0–80, and all myelin pixels
cluster around ~180–255. Pooling all fiber pixels into one histogram produces
a clean bimodal distribution, making global Otsu both simpler and more stable
than per-fiber thresholding (which is unstable for small or thin-myelin fibers).

#### 3b — Connected-Component Selection

After thresholding, each fiber crop may contain multiple dark blobs (e.g.
two touching axons, or residual noise). The **centroid-based** rule selects
the correct axon:

1. Label all connected components within the fiber mask.
2. Look up which component label falls at the **fiber centroid** (from Cellpose).
3. If the centroid pixel is dark → that component is the axon.
4. If the centroid pixel is bright (no dark blob at center) → fall back to
   the **largest** connected component.

This rule is robust because the axon is nearly always centered within its
fiber. The fallback to "largest component" handles the rare case of a
de-centered axon.

Fibers with **multiple distinct dark blobs** (potential multicore fibers) are
flagged and tracked separately — they are excluded from QC pass but recorded.

#### 3c — Hole Filling + Minimum Size Filter

`scipy.ndimage.binary_fill_holes` fills any internal holes in the selected
blob (e.g. staining voids within a large axon). Blobs smaller than
`MIN_AXON_SIZE` pixels are discarded.

---

### Step 4 — Morphometrics

**File:** `morphometrics.py → process_fibers()`

All measurements are computed in physical units (µm, µm²) using `PIXEL_SIZE`.
All operations run on **small bounding-box crops** of the full image — never
on full-size arrays — keeping runtime $O(N \cdot \bar{A})$ rather than
$O(N \cdot H \cdot W)$.

#### Equivalent Circle Diameter (ECD)

The diameter of the circle with the same area as the measured region:

$$d = \sqrt{\frac{4\,A}{\pi}} \cdot \texttt{PIXEL\_SIZE}$$

Applied to both the outer fiber ($A_{\text{fiber}}$ from Cellpose) and the
inner axon ($A_{\text{axon}}$ from the thresholded blob).

#### G-ratio

The fundamental myelination metric, introduced by Rushton (1951):

$$g = \frac{d_{\text{axon}}}{d_{\text{fiber}}}$$

A g-ratio of ~0.6 is considered optimal for conduction velocity in mammalian
peripheral nerve ([Chomiak & Bhatt, 2009](#references)). Values approaching
0 indicate extreme hypermyelination; values approaching 1 indicate minimal
myelin.

#### Myelin Thickness

Average radial thickness of the myelin sheath:

$$t_{\text{myelin}} = \frac{d_{\text{fiber}} - d_{\text{axon}}}{2}$$

#### Centroid Offset

Normalised displacement of the axon centroid relative to the fiber centroid:

$$\delta = \frac{\|\mathbf{c}_{\text{axon}} - \mathbf{c}_{\text{fiber}}\|_2}{\sqrt{A_{\text{fiber}}/\pi}}$$

$\delta = 0$ means perfectly centered; $\delta = 1$ means the axon centroid
lies at the fiber boundary. High values indicate detection errors or
pathological fiber geometry.

#### Shape Descriptors

**Solidity:**

$$\text{solidity} = \frac{A_{\text{axon}}}{A_{\text{convex hull}}}$$

Close to 1 for compact round shapes; low for fragmented or concave shapes.

**Eccentricity:**

$$\varepsilon = \sqrt{1 - \frac{b^2}{a^2}}$$

where $a$ and $b$ are the semi-major and semi-minor axes of the best-fit
ellipse. $\varepsilon = 0$ for a perfect circle; $\varepsilon \to 1$ for
a needle-like shape.

#### Fascicle Area & Aggregate Metrics

The nerve fascicle area is estimated by:
1. Morphologically closing the union of all fiber masks (`disk(30)`) to fill inter-fiber gaps.
2. Filling remaining internal holes (`binary_fill_holes`).
3. Applying a Gaussian blur (`sigma=40`) and thresholding at 0.35 to produce a smooth, slightly expanded convex envelope.

This fascicle mask is used as the denominator for all volume fractions and density, giving biologically meaningful values independent of image crop size.

| Metric | Formula | Unit |
|---|---|---|
| AVF | $\displaystyle\frac{\sum_i A_{\text{axon},i}}{A_{\text{fascicle}}}$ | fraction |
| MVF | $\displaystyle\frac{\sum_i \left(A_{\text{fiber},i} - A_{\text{axon},i}\right)}{A_{\text{fascicle}}}$ | fraction |
| N-ratio | $\text{AVF} + \text{MVF}$ | fraction |
| Aggregate g-ratio | $\bar{g} = \text{mean}(g_i)$ | — |
| Axon density | $\displaystyle\frac{N}{A_{\text{fascicle}} \cdot 10^{-6}}$ | axons/mm² |

where $A_{\text{fascicle}}$ is in µm².

---

### Step 5 — QC Filtering

**File:** `qc.py → apply_qc()`

Each fiber is independently tested against six filters. The **first failing
filter** determines the rejection code shown on the overlay.

| Code | Filter | Default threshold | Biological rationale |
|---|---|---|---|
| `G` | G-ratio | $[0.3,\ 0.9]$ | Outside normal myelination range |
| `lgG` | Large-fiber g-ratio | large fibers (≥ p85) with g < 0.5 | Oversized axon in large fiber — likely detection error |
| `shp` | Shape discordance | fiber_solidity − axon_solidity > 0.2 | Round fiber but irregular axon — likely artefact |
| `sol` | Solidity | $\geq 0.1$ | Non-convex / fragmented blob |
| `off` | Centroid offset | $\leq 0.95$ | Severely off-center axon |
| `Ø` | Axon diameter | $\geq 0.5\,\mu\text{m}$ | Below resolution limit |
| `brd` | Image border | excluded | Incomplete fiber at image edge |

> **Note:** All thresholds are intentionally permissive — the clinician
> reviews the color-coded overlay and adjusts `config.py` to fit the study
> protocol and tissue preparation.

---

### Step 6 — Visualizations

**File:** `visualization.py`

#### Overlay (`*_overlay.png`)

Full-resolution colour-coded image with every fiber annotated:

- Each QC-passed fiber shows a **green axon contour** and **blue myelin ring**.
- Each rejected fiber is **color-coded by rejection reason** (see color scheme below), with the rejection code printed in yellow at its centroid.
- Fibers with no detected axon are shaded **red**.
- A **white contour** outlines the estimated nerve fascicle boundary.
- A semi-transparent legend box is embedded in the bottom-left corner.

#### Numbered Image (`*_numbered.png`)

Same overlay with sequential **yellow numbers** on each QC-passed axon and
a top banner displaying: count, nerve area, mean axon diameter, mean fiber
diameter, mean g-ratio.

#### G-ratio Heatmap (`*_gratio_map.png`)

Each axon pixel is coloured by its g-ratio on the **RdYlGn** colormap:
red = low g-ratio (thin myelin relative to axon), green = high g-ratio
(thick myelin), yellow = ~0.6 (optimal). A colorbar is included.

#### Dashboard (`*_dashboard.png`)

Summary figure with distributions, scatter plots, and aggregate metrics table
(AVF, MVF, N-ratio, mean g-ratio, axon density, nerve area, QC counts).

---

## Mathematics

Complete formula reference:

$$d_{\text{ECD}} = \sqrt{\frac{4A}{\pi}} \cdot s \qquad (s = \texttt{PIXEL\_SIZE},\ \text{µm/px})$$

$$g = \frac{d_{\text{axon}}}{d_{\text{fiber}}}$$

$$t_{\text{myelin}} = \frac{d_{\text{fiber}} - d_{\text{axon}}}{2}$$

$$\delta_{\text{offset}} = \frac{\|\mathbf{c}_{\text{axon}} - \mathbf{c}_{\text{fiber}}\|_2}{\sqrt{A_{\text{fiber}}/\pi}}$$

$$\text{AVF} = \frac{\displaystyle\sum_i A_{\text{axon},i}}{A_{\text{fascicle}}}$$

$$\text{MVF} = \frac{\displaystyle\sum_i \bigl(A_{\text{fiber},i} - A_{\text{axon},i}\bigr)}{A_{\text{fascicle}}}$$

$$\text{N-ratio} = \text{AVF} + \text{MVF}$$

$$\text{density} = \frac{N}{A_{\text{fascicle}} \cdot 10^{-6}} \quad [\text{axons/mm}^2]$$

$$\text{solidity} = \frac{A_{\text{region}}}{A_{\text{convex hull}}}$$

$$\varepsilon = \sqrt{1 - \frac{b^2}{a^2}} \quad (a \geq b = \text{ellipse semi-axes})$$

$$t^*_{\text{Otsu}} = \arg\max_t\; \omega_0(t)\,\omega_1(t)\,\bigl[\mu_0(t)-\mu_1(t)\bigr]^2$$

---

## Configuration Reference

All parameters live in `config.py`. Edit there — no code changes needed.

```python
# I/O
INPUT_DIR  = Path("edited")   # source TIFF directory (contrast-adjusted images)
OUTPUT_DIR = Path("output")   # results directory
PIXEL_SIZE = 0.09             # µm per pixel at acquisition resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL    = "cyto3"   # Cellpose model name
CP_DIAM_UM  = 7.0       # expected fiber diameter (µm)
CP_FLOW_THR = 0.4       # flow error threshold (lower = more permissive)
CP_CELLPROB = 0.0       # cell probability threshold

# Inversion / preprocessing
MASK_ERODE_PX          = 4    # boundary erosion depth (px)
FADE_PX                = 8    # inward fade length (px)
MIN_AXON_SIZE          = 40   # minimum axon blob area (px²)
AXON_SMOOTH_SIGMA      = 1.5  # Gaussian sigma for axon perimeter smoothing (0 = off)
AXON_DILATE_PX         = 3    # expand axon mask after convex hull (0 = off)
AXON_MIN_MYELIN_PX     = 2    # minimum myelin ring thickness (px)
OUTER_ERODE_PX         = 2    # erode fiber mask before morphometrics (px)
AXON_INPUT_WHITE_POINT = 160  # clip white point of axon_input (255 = off)

# QC filters  (permissive by default — clinician adjusts)
MIN_GRATIO             = 0.3
MAX_GRATIO             = 0.9
LARGE_FIBER_MIN_GRATIO = 0.5   # g-ratio floor for large fibers
LARGE_FIBER_PERCENTILE = 85    # "large fiber" = top 15% by fiber area
MIN_AXON_DIAM_UM       = 0.5
MIN_SOLIDITY           = 0.1
MAX_SHAPE_DISCORDANCE  = 0.2   # max (fiber_solidity - axon_solidity)
MAX_CENTROID_OFFSET    = 0.95
MAX_AXON_ECCEN         = 1.0   # effectively disabled
EXCLUDE_BORDER         = True

# Isolated cluster removal
MAIN_CLUSTER_DILATION_PX = 15  # set to 0 to disable
```

**Tuning guide:**

| Symptom | Fix |
|---|---|
| Cellpose over-segments large fibers | Increase `CP_DIAM_UM` |
| Cellpose misses small fibers | Decrease `CP_DIAM_UM` or `CP_CELLPROB` |
| Dark rings visible in `*_axon_input.png` | Increase `MASK_ERODE_PX` |
| Hard edge visible at fiber boundary | Increase `FADE_PX` |
| Too many rejected fibers | Loosen QC thresholds in `config.py` |
| Parasitic fiber clusters outside nerve | Increase `MAIN_CLUSTER_DILATION_PX` |
| Want to re-run axon detection only | Delete `*_axon_inner.npy` |
| Want to re-run everything | Delete `*_cellpose_outer.npy` |

---

## Output Files

For each input image `Foo.tif`, results are saved in `output/Foo/`:

| File | Description |
|---|---|
| `Foo_cellpose_outer.npy` | Cellpose pass-1 label array **(cache)** |
| `Foo_axon_inner.npy` | Axon detection label array **(cache)** |
| `Foo_multicore_labels.npy` | Fiber IDs flagged as multicore **(cache)** |
| `Foo_axon_input.png` | Normalized-inverted debug image (axon=dark, myelin=bright) |
| `Foo_overlay.png` | Full-resolution colour-coded overlay with legend |
| `Foo_numbered.png` | Overlay with numbered QC-passed axons + stats banner |
| `Foo_gratio_map.png` | Per-axon g-ratio heatmap (RdYlGn) |
| `Foo_dashboard.png` | Morphometry dashboard |
| `Foo_morphometrics.csv` | Per-axon measurements (QC-passed only) |
| `Foo_morphometrics.xlsx` | Same as CSV in Excel format |
| `Foo_aggregate.csv` | Image-level aggregates (AVF, MVF, N-ratio, density, mean g-ratio) |

A global `output/summary.csv` collects aggregate metrics across all images.

### Morphometrics CSV columns

| Column | Unit | Description |
|---|---|---|
| `axon_diam` | µm | Axon equivalent circle diameter |
| `fiber_diam` | µm | Fiber (axon + myelin) ECD |
| `gratio` | — | g-ratio = axon_diam / fiber_diam |
| `myelin_thickness` | µm | (fiber_diam − axon_diam) / 2 |
| `axon_area` | µm² | Axon cross-sectional area |
| `fiber_area` | µm² | Fiber cross-sectional area |
| `solidity` | — | Axon blob solidity (0–1) |
| `eccentricity` | — | Axon blob eccentricity (0 = circle, 1 = line) |
| `centroid_offset` | — | Normalised axon–fiber centroid displacement |
| `x0`, `y0` | px | Axon centroid coordinates (image space) |
| `image_border_touching` | bool | True if fiber touches image edge |

---

## Module Structure

```
hybrid-axon-seg/
├── segment.py          # Entry point — process_image() + main()
├── config.py           # All tunable parameters
├── utils.py            # Image type conversion + font loading
├── preprocessing.py    # Per-fiber normalized inversion → axon_input
├── detection.py        # Cellpose pass 1 + Otsu axon detection
├── morphometrics.py    # Per-fiber geometry + aggregate metrics
├── qc.py               # QC filters with rejection reason tracking
├── visualization.py    # Overlay, numbered image, g-ratio map, dashboard
├── compare.py          # Group comparison dashboard (L vs R, or custom groups)
└── test_one.py         # Single-image test runner
```

**Data flow:**

```
img (TIFF)
  │
  ├── [detection.py]      run_cellpose_fibers()  →  outer_labels
  │                       _keep_main_cluster()   →  outer_labels (cleaned)
  │
  ├── [preprocessing.py]  build_axon_input()     →  axon_input  →  *_axon_input.png
  │
  ├── [detection.py]      find_axons()           →  axon_assignments
  │                                                 { fiber_label → (r0, c0, crop_bool) }
  │
  ├── [morphometrics.py]  process_fibers()       →  inner_labels
  │                                                 df_all (per-fiber measurements)
  │                                                 index_image, agg
  │
  ├── [qc.py]             apply_qc()             →  df_pass, df_rej
  │                                                 (df_rej has reject_reason column)
  │
  └── [visualization.py]  make_overlay()         →  *_overlay.png
                          make_numbered()        →  *_numbered.png
                          make_gratio_map()      →  *_gratio_map.png
                          make_dashboard()       →  *_dashboard.png
```

---

## Overlay Color Scheme

**QC-passed fibers:**
```
  █  Green   #00F050  Axon contour      — passed all QC filters
  █  Blue    #4646DC  Myelin ring       — passed all QC filters
```

**Rejected fibers (color-coded by rejection reason):**
```
  █  Orange       #FF8C00  G      g-ratio outside [MIN_GRATIO, MAX_GRATIO]
  █  Red-orange   #FF461E  lgG    large fiber with g-ratio below floor
  █  Amber        #DCC800  shp    shape discordance (round fiber, irregular axon)
  █  Purple       #B43CD2  sol    solidity below threshold
  █  Sky blue     #1EAAE6  off    centroid offset above threshold
  █  Pink         #F03C8C  ecc    eccentricity above threshold
  █  Teal         #14C8A0  Ø      axon diameter below resolution limit
  █  Grey         #A0A0A0  brd    fiber touches image border
```

**Other:**
```
  █  Red     #DC3232  No axon detected  — Otsu found no dark blob in this fiber
  ─  White            Fascicle boundary — estimated nerve cross-section envelope
```

---

## References

- **Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).**
  Cellpose: a generalist algorithm for cellular segmentation.
  *Nature Methods*, 18, 100–106.
  https://doi.org/10.1038/s41592-020-01018-x

- **Otsu, N. (1979).**
  A threshold selection method from gray-level histograms.
  *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
  https://doi.org/10.1109/TSMC.1979.4310076

- **Rushton, W. A. H. (1951).**
  A theory of the effects of fibre size in medullated nerve.
  *Journal of Physiology*, 115(1), 101–122.
  *(Original formulation of the g-ratio concept)*

- **Chomiak, T., & Hu, B. (2009).**
  What is the optimal value of the g-ratio for myelinated fibers in the rat CNS?
  *PLOS ONE*, 4(11), e7754.
  https://doi.org/10.1371/journal.pone.0007754
