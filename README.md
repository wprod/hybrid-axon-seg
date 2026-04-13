# hybrid-axon-seg for Marie 🩵

**Automated nerve morphometry pipeline.**

Detects myelinated axons in cross-sectional nerve images, segments the axon
and myelin compartments, and computes g-ratio, myelin thickness, axon/fiber
diameters, axon volume fraction (AVF), myelin volume fraction (MVF), N-ratio,
and axon density — with full QC overlay output.

---

## Table of Contents

1. [Overview](#overview)
2. [Recommended Workflow](#recommended-workflow)
3. [Web Validation UI](#web-validation-ui)
   - [Running the App](#running-the-app)
   - [Remote Access (Cloudflare Tunnel)](#remote-access-cloudflare-tunnel)
4. [Pipeline Architecture](#pipeline-architecture)
   - [Step 0 — Preprocessing: Contrast Stretch](#step-0--preprocessing-contrast-stretch)
   - [Step 1 — Cellpose Pass 1: Outer Fibers](#step-1--cellpose-pass-1-outer-fibers)
   - [Step 2 — Normalized Inversion: Building `axon_input`](#step-2--normalized-inversion-building-axon_input)
   - [Step 3 — Global Otsu Axon Detection](#step-3--global-otsu-axon-detection)
   - [Step 4 — Morphometrics](#step-4--morphometrics)
   - [Step 5 — QC Filtering](#step-5--qc-filtering)
   - [Step 6 — Visualizations](#step-6--visualizations)
5. [Mathematics](#mathematics)
6. [Configuration Reference](#configuration-reference)
7. [Output Files](#output-files)
8. [Module Structure](#module-structure)
9. [Overlay Color Scheme](#overlay-color-scheme)
10. [References](#references)

---

## Overview

Toluidine-blue staining of semi-thin peripheral nerve cross-sections produces
images where **myelin sheaths appear dark** and **axon interiors appear bright**.
This pipeline automates the full morphometric analysis:

```
edited TIFF  →  contrast stretch  →  Cellpose (cpsam, outer fibers)
             →  normalized inversion  →  Otsu threshold (axon blobs)
             →  morphometrics  →  QC  →  outputs
```

Key design decisions:

| Decision | Rationale |
|---|---|
| **Cellpose cpsam is the source of truth for fiber count** | Deep-learning segmentation handles touching fibers, irregular shapes, and staining variability better than any threshold-based approach |
| **Percentile stretch before Cellpose** | Global p2–p98 stretch removes the gray veil, making myelin rings and axon centers pop; improves Cellpose detection without introducing CLAHE artefacts |
| **Per-fiber normalization before inversion** | Each fiber is contrast-stretched independently, making the axon/myelin contrast invariant to global staining intensity gradients |
| **Global Otsu on the full `axon_input` image** | All normalized fiber pixels together form a clean bimodal distribution — one threshold for the entire image, no per-fiber fitting |
| **Manual fascicle mask (web UI)** | The clinician draws the nerve fascicle boundary in the web UI; this mask is used as the denominator for AVF, MVF, N-ratio, and density, and constrains Cellpose to the fascicle area |
| **Hull-constrained closing** | Axon blob gap-filling uses closing intersected with the convex hull — fills C-shapes without pushing the blob toward the myelin edge |
| **Dual outer-labels (edited + gt)** | Clinician corrections are split: `_outer_edited.npy` tracks corrected predictions, `_outer_gt.npy` is the complete ground truth (includes additions). The diff reveals model false-negatives for fine-tuning |
| **Two-level cache** | Cellpose results and axon detections are cached separately so detection parameters can be tuned without re-running Cellpose |
| **QC overrides** | Clinician can manually accept QC-rejected fibers (mode 9) — stored in `*_qc_overrides.json`, applied on recompute |
| **Per-stem write lock** | Concurrent edits to the same image are serialized via threading locks, so multi-user sessions can't corrupt `.npy` files |

> **Image preparation:** contrast and brightness must be adjusted manually
> (with a medical eye) in ImageJ/Fiji before running the pipeline, as optimal
> settings vary per image and staining batch. Place corrected images in `edited/`.

---

## Recommended Workflow

```
1. Place images in edited/<Group Xw>/<stem>.tif
   (e.g. edited/ALLO A 12w/alloA12w1L.tif)

2. Start the web app
   python app.py  →  http://127.0.0.1:8000

3. For each image, draw the fascicle boundary
   Select image → mode 7 (Fascicle) → click polygon points → close near first point
   (or double-click to close)

4. Run segment.py once all fascicle masks are drawn
   python segment.py

5. Review results in the web app (works on desktop + iPad with Apple Pencil)
   - Delete false-positive axons (mode 2)
   - Add missing fibers (mode 6) — saved to ground truth only for fine-tuning
   - Draw exclusion zones (mode 8) — artefacts/tears subtracted from nerve area
   - Accept QC-rejected fibers you disagree with (mode 9)
   - Click "Recompute" to apply edits to a single image
   - Images with pending edits show a ⚠ badge in the sidebar
   - Undo last edit with Ctrl+Z / Cmd+Z

6. Export: output/summary.csv + per-image CSVs in output/<stem>/
```

> **Cache behaviour:** Cellpose results are cached in `*_cellpose_outer.npy`.
> If the fascicle mask is updated after the last Cellpose run, segment.py
> automatically invalidates the cache and reruns Cellpose. To force a full
> rerun on all images, delete the `*_cellpose_outer.npy` files and rerun
> `segment.py`.

---

## Web Validation UI

`app.py` is a FastAPI-based browser UI for manual review and correction of
segmentation results. It lets a clinician inspect every image, draw fascicle
boundaries, delete false-positive axons, add missing ones, and trigger
recomputation — without touching the command line.

### Running the App

```bash
python app.py
# → http://127.0.0.1:8000
#   Login  →  user: axon  |  password: <randomly generated>
```

The password is printed in the terminal at startup and changes every run. To fix a permanent password:

```bash
APP_PASSWORD=mypassword python app.py
```

**Features:**

| Action | How |
|---|---|
| Browse all images | Sidebar — shows axon count, edit status, ⚠ if recompute pending |
| Blend overlay with raw image | Opacity slider in the header |
| View overlay / numbered / dashboard / g-ratio map | Buttons per image |
| Navigate (pan / zoom) | Mode 1 |
| Delete a false-positive fiber | Mode 2 — click on it |
| Paint axon inside existing fiber | Mode 3 — freehand lasso |
| Paint new outer (myelin only) | Mode 4 — freehand lasso (saved to GT only) |
| Erase region | Mode 5 — freehand lasso |
| Add a complete fiber (outer + axon) | Mode 6 — two-step lasso (saved to GT only) |
| Draw fascicle boundary | Mode 7 — click polygon points, close near first point or double-click |
| Mark exclusion zone | Mode 8 — polygon (subtracted from nerve area) |
| Accept QC-rejected fiber | Mode 9 — click to toggle; recompute to apply |
| Clear fascicle | "Clear fascicle" in right panel |
| Undo last edit | "Undo" button or Ctrl+Z / Cmd+Z |
| Recompute morphometrics | "Recompute" button (single image) |
| Recompute all images | "Recompute All" button (background — app stays usable) |
| Open multi-image comparison | "Compare" button |

**Multi-user awareness:**

When multiple people are connected (e.g. via a Cloudflare tunnel), the sidebar
shows presence badges indicating how many others are currently viewing each image.

**URL hash routing:**

Selecting an image updates the browser URL to `#stem_name`. Bookmarks and
shared links jump directly to the right image.

**iPad / Apple Pencil support:**

The app uses Pointer Events for unified mouse/touch/stylus input:

| Gesture | Action |
|---|---|
| **Pencil** tap/drag | Active tool (delete, lasso, polygon) |
| **Finger** drag | Pan |
| **Finger** pinch | Zoom |
| Mouse left-click | Active tool |
| Mouse right-click / middle-click / space+drag | Pan |
| Mouse wheel | Zoom |

Keyboard shortcuts (1–9, Esc, arrows, F) work with an external keyboard.

**Dual label maps for fine-tuning:**

| File | Content | Purpose |
|---|---|---|
| `_outer_edited.npy` | Cellpose + deletions + modifications | Corrected prediction (no additions) |
| `_outer_gt.npy` | Cellpose + deletions + modifications + additions | Ground truth (complete) |

Deletions and modifications update both files. Manual additions (new fibers the model missed) go only to `_outer_gt.npy`. This lets you compute false-negatives (`gt − edited`) and false-positives (deleted labels) for fine-tuning Cellpose.

All edits are saved to `*_edits.json` and original `.npy` files are kept as
`*_original.npy` backups. QC overrides (mode 9) are stored in
`*_qc_overrides.json`. A "Reset" button restores any image to its pre-edit state.

### Remote Access (Cloudflare Tunnel)

To let a collaborator access the app from another computer without any VPN or port-forwarding setup:

```bash
# 1. Install cloudflared (one-time)
brew install cloudflare/cloudflare/cloudflared

# 2. Start the app and the tunnel in two terminals
python app.py
cloudflared tunnel --url http://localhost:8000
```

Cloudflare prints a public URL (e.g. `https://xyz-abc.trycloudflare.com`). Send it along with the login credentials shown in the app terminal. The tunnel is active only while both processes are running — close them when the session is done.

> The app requires HTTP Basic Auth (username + password) for every request, so the tunnel cannot be accessed without credentials.

---

## Pipeline Architecture

### Step 0 — Preprocessing: Contrast Stretch

**File:** `detection.py → _clahe_preprocess()`
**Enabled by:** `CP_CLAHE = True`

Before Cellpose, the raw image is **globally stretched** from p2–p98 to 0–255.
This removes the gray veil present in some toluidine-blue images, making the
myelin ring / axon center pattern more prominent and improving Cellpose detection.

Optionally, CLAHE can be applied after the stretch by setting
`CLAHE_CLIP_LIMIT > 0`, but in practice the stretch alone performs best.

---

### Step 1 — Cellpose Pass 1: Outer Fibers

**File:** `detection.py → run_cellpose_fibers()`

Cellpose ([Stringer et al., 2021](https://doi.org/10.1038/s41592-020-01018-x))
is a deep-learning instance segmentation model. The `cpsam` model (Cellpose 4)
is used with `augment=True` (4-orientation averaging) for robust detection of
myelinated fiber cross-sections (axon + myelin sheath together).

**How Cellpose works:**
Rather than detecting explicit boundaries, Cellpose uses a gradient-flow
formulation. It predicts, for every pixel, the 2-D vector pointing toward
the center of its containing cell. Pixels that converge to the same attractor
belong to the same cell. This approach is robust to touching and overlapping
objects.

**Parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `CP_MODEL` | `cpsam` | Cellpose 4 model |
| `CP_DIAM_UM` | `5.0 µm` | Expected fiber diameter (sets internal rescaling) |
| `CP_FLOW_THR` | `0.3` | Flow error threshold — lower is more permissive |
| `CP_CELLPROB` | `-1.0` | Cell probability threshold — lower detects more |

**Post-processing:**
- If a **manual fascicle mask** exists, Cellpose runs only inside the fascicle boundary (background pixels are set to median intensity before passing to Cellpose).
- Each fiber mask is **eroded** by `OUTER_ERODE_PX` pixels to shrink the fiber boundary and reduce myelin bleed.
- **Satellite fiber removal**: isolated fibers with fewer than `MIN_SATELLITE_NEIGHBORS` neighbours within 5× fiber diameter are removed (skipped when a fascicle mask is present).
- **Low-QC cluster removal**: clusters of fibers with a QC pass rate below `MIN_CLUSTER_QC_RATE` or fewer than `MIN_CLUSTER_FRACTION` × largest-cluster fiber count are removed.

**Output:** An integer label array cached as `*_cellpose_outer.npy`.

---

### Step 2 — Normalized Inversion: Building `axon_input`

**File:** `preprocessing.py → _invert_crop(), build_axon_input()`

This step transforms each fiber crop so that:

- **Axon interior** (originally bright) → **dark blob ≈ 0**
- **Myelin sheath** (originally dark) → **bright ring ≈ 255**
- **Background / boundary** → **white = 255** (excluded)

> Note: this step always uses the **original raw image**, not the
> contrast-stretched version used by Cellpose.

#### 2a — Distance-Transform Boundary Erosion

A Euclidean distance transform on the fiber mask excludes the boundary zone
where endoneurium contamination is likely. Erosion depth is capped at 25% of
the fiber equivalent radius to preserve signal in small fibers.

#### 2b — Per-Fiber Percentile Stretch

Contrast is normalized using the 5th–95th percentile of clean interior pixels:

$$I_{\text{stretched}}(x,y) = \text{clip}\!\left(\frac{I(x,y) - p_5}{p_{95} - p_5} \cdot 255,\ 0,\ 255\right)$$

#### 2c — Inversion

$$I_{\text{inverted}}(x,y) = 255 - I_{\text{stretched}}(x,y)$$

#### 2d — Inward Fade

A smooth fade from white at the eroded edge to fully inverted at `FADE_PX`
pixels inward prevents hard black ring artefacts at the boundary.

---

### Step 3 — Global Otsu Axon Detection

**File:** `detection.py → find_axons()`

Otsu's method finds the threshold that maximises inter-class variance over
all fiber pixels:

$$t^* = \arg\max_t \; \omega_0(t)\,\omega_1(t)\,\bigl[\mu_0(t) - \mu_1(t)\bigr]^2$$

**Why global?** Per-fiber normalization makes all axon pixels cluster around
~0–80 and all myelin pixels around ~180–255 across the entire image, giving a
clean bimodal histogram. Global Otsu is simpler and more stable than per-fiber
thresholding.

After thresholding, the correct axon blob is selected by centroid (falls back
to largest blob). Multicore fibers (multiple distinct blobs) are flagged.

**Post-processing per blob:**

1. **Hull-constrained closing** — `binary_closing(disk(6)) & convex_hull_image()`.
   Closing fills C-shape gaps; intersecting with the convex hull prevents the
   blob from being pushed toward the nearest myelin edge (the original
   `closing(disk(8))` caused axons to "stick" to the myelin on one side).
2. **Adaptive myelin margin** — the axon must be at least
   `max(AXON_MIN_MYELIN_PX, AXON_MIN_MYELIN_FRAC × fiber_radius)` pixels from
   the fiber edge. Large fibers get a proportionally larger gap.
3. **Gaussian smoothing** (`AXON_SMOOTH_SIGMA`) — smooths the perimeter.
4. **Adaptive dilation** (`AXON_DILATE_PX`) — compensates for Otsu
   under-segmentation. Reduced for large fibers.
5. **Morphological opening** (`AXON_OPEN_PX`) — removes thin protrusions.
6. All operations are clipped to the margin mask (hard guarantee).

---

### Step 4 — Morphometrics

**File:** `morphometrics.py → process_fibers()`

All measurements use physical units (µm) via `PIXEL_SIZE`. Operations run on
bounding-box crops for efficiency.

| Metric | Formula |
|---|---|
| ECD | $\sqrt{4A/\pi} \cdot \texttt{PIXEL\_SIZE}$ |
| G-ratio | $d_{\text{axon}} / d_{\text{fiber}}$ |
| Myelin thickness | $(d_{\text{fiber}} - d_{\text{axon}}) / 2$ |
| Centroid offset | $\|\mathbf{c}_{\text{axon}} - \mathbf{c}_{\text{fiber}}\| / \sqrt{A_{\text{fiber}}/\pi}$ |

**Aggregate stats** (`compute_aggregate()`):

- **N-ratio** counts ALL detected fibers (pass + rejected + no-axon) because
  every fibre occupies nerve cross-section regardless of health.
- **AVF / MVF / G-ratio / density** use QC-passed fibers only (functional population).
- **Area-weighted g-ratio** $\sqrt{\sum A_\text{axon} / \sum A_\text{fiber}}$ is reported alongside the arithmetic mean.
- **Exclusion zones** (artefacts, tears drawn in the web UI) are subtracted
  from the fascicle area denominator so they don't dilute the ratios.
- The fascicle area (manual or auto-computed) is the base denominator.

---

### Step 5 — QC Filtering

**File:** `qc.py → apply_qc()`

| Code | Filter | Default threshold |
|---|---|---|
| `G` | G-ratio | $[0.3,\ 0.9]$ |
| `lgG` | Large-fiber g-ratio (≥ p85) | g < 0.5 |
| `shp` | Shape discordance | fiber_solidity − axon_solidity > 0.2 |
| `sol` | Solidity | ≥ 0.1 |
| `off` | Centroid offset | ≤ 0.95 |
| `Ø` | Axon diameter | ≥ 0.5 µm |
| `brd` | Image border | excluded |

All thresholds are intentionally permissive — adjust in `config.py`.

---

### Step 6 — Visualizations

**File:** `visualization.py`

- **`*_overlay.png`** — Full-resolution color-coded overlay (green axon, blue myelin, rejection colors, white fascicle boundary)
- **`*_numbered.png`** — Overlay with sequential numbers on QC-passed axons + stats banner
- **`*_gratio_map.png`** — Per-axon g-ratio heatmap (RdYlGn colormap) — only generated when `GRATIO_MAP = True`
- **`*_dashboard.png`** — 2×4 layout: histograms (axon diam, fiber diam, g-ratio, myelin thickness), scatter plots, rejection breakdown, and aggregate metrics table

---

## Mathematics

$$d_{\text{ECD}} = \sqrt{\frac{4A}{\pi}} \cdot s \qquad (s = \texttt{PIXEL\_SIZE},\ \text{µm/px})$$

$$g = \frac{d_{\text{axon}}}{d_{\text{fiber}}}$$

$$t_{\text{myelin}} = \frac{d_{\text{fiber}} - d_{\text{axon}}}{2}$$

$$\delta_{\text{offset}} = \frac{\|\mathbf{c}_{\text{axon}} - \mathbf{c}_{\text{fiber}}\|_2}{\sqrt{A_{\text{fiber}}/\pi}}$$

$$\text{AVF} = \frac{\displaystyle\sum_i A_{\text{axon},i}}{A_{\text{fascicle}}}$$

$$\text{MVF} = \frac{\displaystyle\sum_i \bigl(A_{\text{fiber},i} - A_{\text{axon},i}\bigr)}{A_{\text{fascicle}}}$$

$$\text{N-ratio} = \frac{\displaystyle\sum_{\text{all fibers}} A_{\text{fiber},i}}{A_{\text{fascicle}} - A_{\text{exclusion}}}$$

$$\text{density} = \frac{N}{A_{\text{fascicle}} \cdot 10^{-6}} \quad [\text{axons/mm}^2]$$

$$g_{\text{area-weighted}} = \sqrt{\frac{\sum_i A_{\text{axon},i}}{\sum_i A_{\text{fiber},i}}}$$

---

## Configuration Reference

All parameters live in `config.py`. Edit there — no code changes needed.

```python
# I/O
INPUT_DIR  = Path("edited")   # source TIFF directory (subfolders = group/timepoint)
OUTPUT_DIR = Path("output")   # results directory
PIXEL_SIZE = 0.09             # µm per pixel at acquisition resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL    = "cpsam"   # Cellpose 4 model
CP_DIAM_UM  = 5.0       # expected fiber diameter (µm)
CP_FLOW_THR = 0.3       # flow error threshold (lower = more permissive)
CP_CELLPROB = -1.0      # cell probability threshold (lower = more detections)

# Contrast preprocessing (applied before Cellpose only)
CP_CLAHE        = True   # enable contrast stretch before Cellpose
CLAHE_CLIP_LIMIT = 0.0   # 0 = stretch only; >0 = stretch + CLAHE
CLAHE_TILE_SIZE  = (64, 64)
STRETCH_PLOW    = 2      # low percentile for stretch
STRETCH_PHIGH   = 98     # high percentile for stretch

# Inversion / preprocessing (axon_input — uses original raw image)
MASK_ERODE_PX       = 4    # boundary erosion depth (px)
FADE_PX             = 8    # inward fade length (px)
MIN_AXON_SIZE       = 40   # minimum axon blob area (px²)
AXON_SMOOTH_SIGMA   = 1.5  # Gaussian sigma for axon perimeter smoothing (0 = off)
AXON_DILATE_PX      = 3    # expand axon mask after hull-constrained closing (0 = off)
AXON_OPEN_PX        = 2    # morphological opening to remove thin protrusions (0 = off)
AXON_MIN_MYELIN_PX  = 2    # hard floor: axon ≥ this many px from fiber edge
AXON_MIN_MYELIN_FRAC = 0.30 # adaptive floor: axon ≥ frac × fiber_radius from edge
OUTER_ERODE_PX      = 2    # erode fiber mask before morphometrics (px)
AXON_INPUT_WHITE_POINT = 255  # clip white point of axon_input (255 = off)
GRATIO_MAP          = False   # spatial g-ratio heatmap (slow — enable only if needed)

# QC filters (permissive by default — adjust per study)
MIN_GRATIO             = 0.3
MAX_GRATIO             = 0.9
LARGE_FIBER_MIN_GRATIO = 0.5   # g-ratio floor for large fibers
LARGE_FIBER_PERCENTILE = 85    # "large fiber" = top 15% by fiber area
MIN_AXON_DIAM_UM       = 0.5
MIN_SOLIDITY           = 0.1
MAX_SHAPE_DISCORDANCE  = 0.2
MAX_CENTROID_OFFSET    = 0.95
MAX_AXON_ECCEN         = 1.0   # effectively disabled
EXCLUDE_BORDER         = True

# Satellite / cluster removal (skipped when fascicle mask is present)
MIN_SATELLITE_NEIGHBORS = 15   # min neighbours within 5× fiber diam
MIN_CLUSTER_QC_RATE     = 0.50 # clusters below this pass rate are removed
MIN_CLUSTER_FRACTION    = 0.10 # clusters < 10% of largest cluster are removed
```

**Tuning guide:**

| Symptom | Fix |
|---|---|
| Cellpose detects large non-fiber structures | Add a fascicle mask in the web UI |
| Cellpose misses small fibers | Decrease `CP_DIAM_UM` or `CP_CELLPROB` |
| Cellpose over-segments | Increase `CP_DIAM_UM` or `CP_FLOW_THR` |
| Dark rings in `*_axon_input.png` | Increase `MASK_ERODE_PX` |
| Hard edge at fiber boundary | Increase `FADE_PX` |
| Too many QC rejections | Loosen thresholds in `config.py` |
| Re-run axon detection only | Delete `*_axon_inner.npy` |
| Re-run Cellpose only | Delete `*_cellpose_outer.npy` |
| Re-run everything | Delete both caches above |

---

## Output Files

For each input image `Foo.tif`, results are saved in `output/Foo/`:

| File | Description |
|---|---|
| `Foo_cellpose_outer.npy` | Cellpose fiber label array **(cache)** |
| `Foo_axon_inner.npy` | Axon detection label array **(cache)** |
| `Foo_multicore_labels.npy` | Fiber IDs flagged as multicore **(cache)** |
| `Foo_fascicle_mask.npy` | Auto-computed fascicle mask |
| `Foo_fascicle_mask_edited.npy` | Manual fascicle mask (drawn in web UI) — takes priority |
| `Foo_raw.png` | Raw input image cached as PNG |
| `Foo_axon_input.png` | Normalized-inverted debug image (axon=dark, myelin=bright) |
| `Foo_overlay.png` | Full-resolution colour-coded overlay |
| `Foo_numbered.png` | Overlay with numbered QC-passed axons + stats banner |
| `Foo_gratio_map.png` | Per-axon g-ratio heatmap (RdYlGn) — only when `GRATIO_MAP = True` |
| `Foo_dashboard.png` | Morphometry dashboard |
| `Foo_morphometrics.csv` | Per-axon measurements (QC-passed only) |
| `Foo_morphometrics.xlsx` | Same as CSV in Excel format |
| `Foo_aggregate.csv` | Image-level aggregates (AVF, MVF, N-ratio, density, mean g-ratio) |
| `Foo_exclusion_mask.npy` | User-drawn exclusion zones (artefacts, tears) |
| `Foo_edits.json` | Manual edit history (deleted/added fibers) |
| `Foo_qc_overrides.json` | Fiber labels manually accepted via mode 9 (QC Accept) |
| `Foo_outer_edited.npy` | Corrected prediction: Cellpose + deletions/modifications (no additions) |
| `Foo_outer_gt.npy` | Ground truth: Cellpose + deletions/modifications + manual additions |
| `Foo_axon_inner_original.npy` | Backup of axon labels before editing |
| `Foo_axon_version.txt` | Cache version marker — triggers axon cache rebuild on version bump |
| `Foo_*_undo.npy` | Single-level undo snapshots (axon + outer_edited + outer_gt) |

A global `output/summary.csv` collects aggregate metrics across all images.

---

## Module Structure

```
hybrid-axon-seg/
├── segment.py          # Entry point — process_image() + main()
├── app.py              # Web validation UI (FastAPI) — fascicle drawing, correction, recompute
├── config.py           # All tunable parameters
├── utils.py            # Image type conversion, font loading, satellite/cluster detection
├── preprocessing.py    # Per-fiber normalized inversion → axon_input
├── detection.py        # Cellpose pass 1 (with stretch preprocessing) + Otsu axon detection
├── morphometrics.py    # Per-fiber geometry + aggregate metrics
├── qc.py               # QC filters with rejection reason tracking
├── visualization.py    # Overlay, numbered image, g-ratio map, dashboard
├── compare.py          # Group comparison dashboard
├── clean_output.py     # Remove generated files (keep fascicle masks + raws)
├── test_one.py         # Quick single-image test runner
└── static/             # Web UI assets (app.js, style.css, index.html)
```

**Data flow:**

```
img (TIFF)
  │
  ├── [detection.py]      _clahe_preprocess()    →  img_stretched (Cellpose only)
  │                       run_cellpose_fibers()  →  outer_labels  (cached)
  │                       satellite/cluster removal
  │
  ├── [preprocessing.py]  build_axon_input(img)  →  axon_input  (uses raw img)
  │                                                 saved as *_axon_input.png
  │
  ├── [detection.py]      find_axons()           →  axon_assignments
  │
  ├── [morphometrics.py]  process_fibers()       →  inner_labels, df_all
  │                       compute_aggregate()    →  agg (nratio, avf, mvf, …)
  │
  ├── [qc.py]             apply_qc()             →  df_pass, df_rej
  │
  └── [visualization.py]  make_overlay()         →  *_overlay.png
                          make_numbered()        →  *_numbered.png
                          make_gratio_map()      →  *_gratio_map.png
                          make_dashboard()       →  *_dashboard.png
```

---

## Overlay Color Scheme

The overlay uses fill regions + 1-px contour boundaries:

**QC-passed fibers:**
```
  █  Green    (0,210,60)   Axon fill + (0,240,80) contour
  █  Blue     (50,50,240)  Myelin fill + contour
```

**Rejected fibers:**
```
  █  Orange   (255,140,0)  Axon fill + contour — detected but QC rejected
```

**Other:**
```
  █  Red          (220,50,50)   No axon detected — Otsu found no dark blob
  █  Crimson      (210,50,85)   Multi-core fiber (2+ cores → excluded)
  ─  White                      Fascicle boundary — manual or auto-computed
```

**Dashboard rejection breakdown** uses per-reason colors:
```
  █  Orange       (255,140,0)   G      g-ratio outside [MIN_GRATIO, MAX_GRATIO]
  █  Red-orange   (255,70,30)   lgG    large fiber with g-ratio below floor
  █  Amber        (220,200,0)   shp    shape discordance (round fiber, irregular axon)
  █  Purple       (180,60,210)  sol    solidity below threshold
  █  Sky blue     (30,170,230)  off    centroid offset above threshold
  █  Pink         (240,60,140)  ecc    eccentricity above threshold
  █  Teal         (20,200,160)  Ø      axon diameter below resolution limit
  █  Grey         (160,160,160) brd    fiber touches image border
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
