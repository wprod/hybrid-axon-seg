# hybrid-axon-seg for Marie 🩵

**Automated nerve morphometry pipeline.**

Detects myelinated axons in cross-sectional nerve images, segments the axon
and myelin compartments, and computes g-ratio, myelin thickness, axon/fiber
diameters, axon volume fraction (AVF), myelin volume fraction (MVF), and
axon density — with full QC overlay output.

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
raw TIFF  →  Cellpose (outer fibers)  →  normalized inversion
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

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.11+)
pip install cellpose scikit-image scipy numpy pandas pillow matplotlib openpyxl

# 2. Place your TIFF images in the edited/ directory

# 3. Run the full pipeline on all images
python segment.py

# 4. Or test on a single image first
python test_one.py
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
| `CP_DIAM_UM` | `8.0 µm` | Expected fiber diameter (sets internal rescaling) |
| `CP_FLOW_THR` | `0.4` | Flow error threshold — lower is more permissive |
| `CP_CELLPROB` | `0.0` | Cell probability threshold — lower detects more |

The diameter in pixels fed to Cellpose is:

$$d_{px} = \frac{\texttt{CP\_DIAM\_UM}}{\texttt{PIXEL\_SIZE}}$$

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
reused in the fade step below.

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

#### Aggregate Metrics

| Metric | Formula | Unit |
|---|---|---|
| AVF | $\displaystyle\frac{\sum_i A_{\text{axon},i}}{H \cdot W \cdot s^2}$ | fraction |
| MVF | $\displaystyle\frac{\sum_i \left(A_{\text{fiber},i} - A_{\text{axon},i}\right)}{H \cdot W \cdot s^2}$ | fraction |
| Aggregate g-ratio | $\bar{g} = \text{mean}(g_i)$ | — |
| Axon density | $\displaystyle\frac{N}{H \cdot W \cdot s^2 \cdot 10^{-6}}$ | axons/mm² |

where $s = \texttt{PIXEL\_SIZE}$ (µm/px), $H \times W$ is the image size in
pixels, and $N$ is the number of QC-passed fibers.

---

### Step 5 — QC Filtering

**File:** `qc.py → apply_qc()`

Each fiber is independently tested against six filters. The **first failing
filter** determines the rejection code printed on the overlay.

| Code | Filter | Default threshold | Biological rationale |
|---|---|---|---|
| `G` | G-ratio | $[0.215,\ 0.9]$ | Outside normal myelination range — likely a detection error |
| `Ø` | Axon diameter | $\geq 0.5\,\mu\text{m}$ | Below optical resolution limit |
| `sol` | Solidity | $\geq 0.4$ | Non-convex / fragmented blob — likely noise or artefact |
| `ecc` | Eccentricity | $\leq 0.99$ | Near-linear shape — likely a cross-section artefact |
| `off` | Centroid offset | $\leq 0.65$ | Severely off-center axon within its fiber |
| `brd` | Image border | excluded | Incomplete fiber at image edge |

> **Note:** All thresholds are intentionally permissive — the clinician
> reviews the color-coded overlay and adjusts `config.py` to fit the study
> protocol and tissue preparation.

---

### Step 6 — Visualizations

**File:** `visualization.py`

#### Overlay (`*_overlay.png`)

Full-resolution colour-coded image with every fiber annotated:

- Each QC-passed fiber shows a **green axon** and **blue myelin ring**.
- Each rejected fiber shows an **orange axon** with the rejection code (`G`,
  `Ø`, `sol`, `ecc`, `off`, `brd`) printed in yellow at its centroid.
- Fibers with no detected axon are **red**.
- A semi-transparent legend box is embedded in the bottom-left corner.

#### Numbered Image (`*_numbered.png`)

Same overlay with sequential **yellow numbers** on each QC-passed axon and
a top banner displaying: count, mean axon diameter, mean fiber diameter, mean
g-ratio.

#### G-ratio Heatmap (`*_gratio_map.png`)

Each axon pixel is coloured by its g-ratio on the **RdYlGn** colormap:
red = low g-ratio (thin myelin relative to axon), green = high g-ratio
(thick myelin), yellow = ~0.6 (optimal). A colorbar is included.

#### Dashboard (`*_dashboard.png`)

Six-panel summary figure:

| Panel | Content |
|---|---|
| Top-left | Axon diameter distribution (histogram) |
| Top-center | G-ratio distribution (histogram) |
| Top-right | Myelin thickness distribution (histogram) |
| Bottom-left | G-ratio vs. axon diameter (scatter + reference line at 0.6) |
| Bottom-center | Myelin thickness vs. axon diameter (scatter) |
| Bottom-right | QC & aggregate metrics table |

---

## Mathematics

Complete formula reference:

$$d_{\text{ECD}} = \sqrt{\frac{4A}{\pi}} \cdot s \qquad (s = \texttt{PIXEL\_SIZE},\ \text{µm/px})$$

$$g = \frac{d_{\text{axon}}}{d_{\text{fiber}}}$$

$$t_{\text{myelin}} = \frac{d_{\text{fiber}} - d_{\text{axon}}}{2}$$

$$\delta_{\text{offset}} = \frac{\|\mathbf{c}_{\text{axon}} - \mathbf{c}_{\text{fiber}}\|_2}{\sqrt{A_{\text{fiber}}/\pi}}$$

$$\text{AVF} = \frac{\displaystyle\sum_i A_{\text{axon},i}}{H \cdot W \cdot s^2}$$

$$\text{MVF} = \frac{\displaystyle\sum_i \bigl(A_{\text{fiber},i} - A_{\text{axon},i}\bigr)}{H \cdot W \cdot s^2}$$

$$\text{density} = \frac{N}{H \cdot W \cdot s^2 \cdot 10^{-6}} \quad [\text{axons/mm}^2]$$

$$\text{solidity} = \frac{A_{\text{region}}}{A_{\text{convex hull}}}$$

$$\varepsilon = \sqrt{1 - \frac{b^2}{a^2}} \quad (a \geq b = \text{ellipse semi-axes})$$

$$t^*_{\text{Otsu}} = \arg\max_t\; \omega_0(t)\,\omega_1(t)\,\bigl[\mu_0(t)-\mu_1(t)\bigr]^2$$

---

## Configuration Reference

All parameters live in `config.py`. Edit there — no code changes needed.

```python
# I/O
INPUT_DIR  = Path("edited")   # source TIFF directory
OUTPUT_DIR = Path("output")   # results directory
PIXEL_SIZE = 0.09             # µm per pixel at acquisition resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL    = "cyto3"   # Cellpose model name
CP_DIAM_UM  = 8.0       # expected fiber diameter (µm)
CP_FLOW_THR = 0.4       # flow error threshold (lower = more permissive)
CP_CELLPROB = 0.0       # cell probability threshold

# Inversion / preprocessing
MASK_ERODE_PX = 4   # boundary erosion depth (px)
FADE_PX       = 8   # inward fade length (px)
MIN_AXON_SIZE = 40  # minimum axon blob area (px²)

# QC filters  (permissive by default — clinician adjusts)
MIN_GRATIO          = 0.215
MAX_GRATIO          = 0.9
MIN_AXON_DIAM_UM    = 0.5
MIN_SOLIDITY        = 0.4
MAX_CENTROID_OFFSET = 0.65
MAX_AXON_ECCEN      = 0.99
EXCLUDE_BORDER      = True
```

**Tuning guide:**

| Symptom | Fix |
|---|---|
| Cellpose over-segments large fibers | Increase `CP_DIAM_UM` |
| Cellpose misses small fibers | Decrease `CP_DIAM_UM` or `CP_CELLPROB` |
| Dark rings visible in `*_axon_input.png` | Increase `MASK_ERODE_PX` |
| Hard edge visible at fiber boundary | Increase `FADE_PX` |
| Too many orange fibers | Loosen QC thresholds in `config.py` |
| Want to re-run axon detection only | Delete `*_axon_inner.npy` |
| Want to re-run everything | Delete `*_cellpose_outer.npy` |

---

## Output Files

For each input image `Foo.tif`, results are saved in `output/Foo/`:

| File | Description |
|---|---|
| `Foo_cellpose_outer.npy` | Cellpose pass-1 label array **(cache)** |
| `Foo_axon_inner.npy` | Axon detection label array **(cache)** |
| `Foo_axon_input.png` | Normalized-inverted debug image (axon=dark, myelin=bright) |
| `Foo_overlay.png` | Full-resolution colour-coded overlay with legend |
| `Foo_numbered.png` | Overlay with numbered QC-passed axons + stats banner |
| `Foo_gratio_map.png` | Per-axon g-ratio heatmap (RdYlGn) |
| `Foo_dashboard.png` | 6-panel morphometry dashboard |
| `Foo_morphometrics.csv` | Per-axon measurements (QC-passed only) |
| `Foo_morphometrics.xlsx` | Same as CSV in Excel format |
| `Foo_aggregate.csv` | Image-level aggregates (AVF, MVF, density, mean g-ratio) |

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
└── test_one.py         # Single-image test runner
```

**Data flow:**

```
img (TIFF)
  │
  ├── [detection.py]      run_cellpose_fibers()  →  outer_labels
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

```
  █  Green   #00D23C  Axon interior     — passed all QC filters
  █  Blue    #3232F0  Myelin ring       — passed all QC filters
  █  Orange  #FF8C00  Axon detected     — failed QC (reason code printed on fiber)
  █  Red     #DC3232  No axon detected  — Otsu found no dark blob in this fiber

Rejection codes (printed in yellow on each orange fiber):
  G    g-ratio outside [MIN_GRATIO, MAX_GRATIO]
  Ø    axon diameter < MIN_AXON_DIAM_UM
  sol  solidity < MIN_SOLIDITY
  ecc  eccentricity > MAX_AXON_ECCEN
  off  centroid offset > MAX_CENTROID_OFFSET
  brd  fiber touches image border
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
