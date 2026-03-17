# hybrid-axon-seg

Automated morphometry pipeline for myelinated axons in toluidine-blue bright-field histology images.

---

## Overview

This pipeline quantifies myelinated nerve fiber morphology from light-microscopy images of osmium-fixed, toluidine-blue-stained sections. It produces per-axon measurements (diameter, g-ratio, myelin thickness) and aggregate tissue metrics (AVF, MVF, axon density) along with annotated overlays.

**Design principle:** Cellpose is the single source of truth for fiber count. No other step can create, split, or inflate fibers — only accept or reject them.

---

## Method

### Step 1 — Outer fiber segmentation (Cellpose)

[Cellpose](https://github.com/MouseLand/cellpose) (`cyto3` model) detects the outer boundary of each myelinated fiber (axon + myelin sheath together). It runs on the raw image with a target diameter set in physical units (`CP_DIAM_UM / PIXEL_SIZE` pixels).

- Each detected instance becomes one labeled region in `outer_labels`.
- The fiber count `n_outer` from this step is final and never modified downstream.
- Hardware: MPS (Apple Silicon) → CUDA → CPU, in that order of preference.

### Step 2 — Inner boundary detection (progressive erosion)

Identifying the axon within each fiber is the core challenge. Toluidine blue stains myelin dark blue, but axon interiors are non-uniform (varying brightness, occasional debris). Global thresholding (Otsu) fails because the axon/myelin contrast is locally consistent but globally variable across a field of view.

**The approach: erosion intensity profiling**

For each fiber crop (extracted from `outer_labels` via `regionprops.bbox`):

1. **Progressive erosion**: erode the fiber binary mask one pixel at a time (disk(1) structuring element), up to `max_erosion = 40` steps.
2. **Ring sampling**: at each step, measure the mean grayscale intensity of the removed ring (the pixels between current and previous mask).
3. **Profile analysis**: the resulting 1-D intensity profile has a characteristic U-shape:
   - The profile drops as erosion enters the dark myelin annulus.
   - It reaches a minimum at the core of the myelin ring.
   - It rises again as erosion enters the lighter axon interior.

```
Intensity
   │
   │  ●                              ●●●
   │   ●                          ●●●
   │    ●●                     ●●●
   │      ●●               ●●●
   │        ●●●●●●●●●●●●●●
   └──────────────────────────────────▶ erosion step
                   ↑
               myelin min  ← axon = everything still inside here
              (red line)
```

4. **Axon detection**:
   - Find `myelin_idx` = the step with minimum intensity (deepest myelin ring).
   - The axon mask = the eroded binary mask **at that exact step**. No gradient, no threshold.
   - Physical rationale: the darkest ring is the myelin core. What remains inside it is the axon interior.

5. **Expansion**: the detected inner mask is optionally dilated by `AXON_EXPAND_UM` µm and clipped to the fiber boundary, compensating for the slight underestimation inherent to erosion-based detection.

6. **Parallelisation**: all fibers are processed in parallel using `joblib.Parallel` with thread-based workers (safe with NumPy/skimage).

### Step 3 — Quality control

Detected axons passing all of the following are included in the final output:

| Filter | Parameter | Default |
|--------|-----------|---------|
| G-ratio in valid range | `MIN_GRATIO` / `MAX_GRATIO` | 0.2 – 0.8 |
| Axon diameter ≥ minimum | `MIN_AXON_DIAM_UM` | 1.0 µm |
| Shape convexity (solidity) | `MIN_SOLIDITY` | 0.60 |
| Not touching image border | `EXCLUDE_BORDER` | True |

---

## Morphometrics

All measurements use equivalent-circle diameter: `d = √(4A/π) × pixel_size`.

| Metric | Definition |
|--------|-----------|
| `axon_diam` | Diameter of axon (inner boundary), µm |
| `fiber_diam` | Diameter of fiber (axon + myelin), µm |
| `gratio` | `axon_diam / fiber_diam` — 0.6 is typical for healthy peripheral nerve |
| `myelin_thickness` | `(fiber_diam − axon_diam) / 2`, µm |
| `axon_area` | Axon cross-sectional area, µm² |
| `fiber_area` | Fiber cross-sectional area, µm² |
| `solidity` | Convexity of axon mask (1 = perfectly convex) |
| `image_border_touching` | Whether fiber bbox touches image edge |

**Aggregate (per-image):**

| Metric | Definition |
|--------|-----------|
| `AVF` | Axon volume fraction = total axon area / total image area |
| `MVF` | Myelin volume fraction = total myelin area / total image area |
| `gratio_aggr` | Mean g-ratio across all passing axons |
| `axon_density_mm²` | Number of passing axons per mm² |

---

## Outputs

For each input image `<stem>`, the following files are written to `output/<stem>/`:

| File | Content |
|------|---------|
| `<stem>_overlay.png` | Color-coded segmentation overlay |
| `<stem>_numbered.png` | Same overlay with axon index labels and summary banner |
| `<stem>_gratio_map.png` | Spatial heatmap of g-ratio (red=low, green=high) |
| `<stem>_dashboard.png` | Distribution histograms + scatter plots + aggregate table |
| `<stem>_morphometrics.csv` | Per-axon measurements (QC-passing only) |
| `<stem>_morphometrics.xlsx` | Same, Excel format |
| `<stem>_aggregate.csv` | Single-row aggregate stats for this image |

### Overlay color code

| Color | Meaning |
|-------|---------|
| **Green** | Axon interior (inner boundary detected, QC candidate) |
| **Blue** | Myelin (fiber interior minus axon) |
| **Red** | Fiber with no detected inner boundary (no clear myelin/axon transition) |

---

## Configuration

All parameters are at the top of `segment.py`:

```python
INPUT_DIR   = Path("edited")   # folder containing input images
OUTPUT_DIR  = Path("output")
PIXEL_SIZE  = 0.09             # µm/pixel — MUST match your acquisition

# Cellpose
CP_MODEL    = "cyto3"          # pretrained model name
CP_DIAM_UM  = 8.0              # expected fiber outer diameter in µm
CP_FLOW_THR = 0.4              # flow threshold (lower = more detections)
CP_CELLPROB = 0.0              # cell probability threshold

# Inner boundary
MIN_AXON_SIZE  = 40            # minimum axon area in pixels²
AXON_EXPAND_UM = 0.18          # post-detection dilation in µm (0 to disable)

# QC filters
MIN_GRATIO       = 0.2
MAX_GRATIO       = 0.8
MIN_AXON_DIAM_UM = 1.0         # µm
MIN_SOLIDITY     = 0.60
EXCLUDE_BORDER   = True
```

> **Critical**: set `PIXEL_SIZE` correctly before running. It is used to convert Cellpose diameter, compute all physical measurements, and scale `AXON_EXPAND_UM`. All other pixel-based parameters (like `MIN_AXON_SIZE`) are secondary and rarely need changing.

---

## Usage

```bash
# Full pipeline — all images in edited/
python segment.py

# Single image (for testing)
python test_one.py

# Validate inner boundary detection on N random fibers BEFORE running the full pipeline
python diagnose_profile.py           # 12 random fibers
python diagnose_profile.py 24        # 24 fibers
python diagnose_profile.py 12 99     # 12 fibers, random seed 99
```

### Expected folder layout

```
hybrid-axon-seg/
├── edited/                  # input images (.tif, .tiff, .png)
│   ├── sample_A.tif
│   └── sample_B.tif
├── output/                  # created automatically
│   └── sample_A/
│       ├── sample_A_overlay.png
│       ├── sample_A_numbered.png
│       ├── sample_A_gratio_map.png
│       ├── sample_A_dashboard.png
│       ├── sample_A_morphometrics.csv
│       ├── sample_A_morphometrics.xlsx
│       └── sample_A_aggregate.csv
├── segment.py
├── diagnose_profile.py
├── test_one.py
└── README.md
```

---

## Diagnostic tool

`diagnose_profile.py` visualises the erosion intensity profile for N randomly sampled fibers **without** running the full pipeline. Use it to validate that the myelin/axon transition is correctly detected on your images before processing a whole dataset.

Each fiber is shown as two panels:

- **Left**: fiber crop with cyan = outer Cellpose boundary, lime = detected inner/axon boundary.
- **Right**: intensity profile with orange dotted line = myelin minimum (deepest myelin ring), red dashed line = detected axon edge (first rise after minimum).

**What to look for:**
- Orange and red lines should be separated — orange at the bottom of the U, red on the right rising slope.
- Lime contour should sit visibly inside the dark myelin ring.
- "no transition detected" (red text) = fiber will be rejected. Acceptable for a minority of fibers; if widespread, check `PIXEL_SIZE` and image quality.

---

## Dependencies

```
cellpose
torch
scikit-image
numpy
pandas
matplotlib
Pillow
joblib
openpyxl        # optional, for .xlsx export
```

Install:

```bash
pip install cellpose torch scikit-image numpy pandas matplotlib Pillow joblib openpyxl
```
