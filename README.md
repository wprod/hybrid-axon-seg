# hybrid-axon-seg

Automated quantitative morphometry pipeline for myelinated nerve fibers in
toluidine-blue stained semi-thin cross-sections. Developed for a rat peripheral
nerve regeneration study comparing allograft and autograft repair strategies.

---

## Table of Contents

1. [Scientific Context](#scientific-context)
2. [Pipeline Overview](#pipeline-overview)
3. [Recommended Workflow](#recommended-workflow)
4. [Web Validation UI](#web-validation-ui)
5. [U-Net Architecture and Training](#u-net-architecture-and-training)
6. [Morphometrics](#morphometrics)
7. [QC Filtering](#qc-filtering)
8. [Statistical Analysis](#statistical-analysis)
9. [Configuration Reference](#configuration-reference)
10. [Output Files](#output-files)
11. [Module Structure](#module-structure)

---

## Scientific Context

### Experimental Design

Peripheral nerve regeneration study in rats. Nerves are imaged after surgical
repair at two timepoints:

| Code | Meaning |
|---|---|
| `allo` | Allograft — nerve from a donor animal |
| `auto` | Autograft — nerve from the same animal (gold standard) |
| `X` | Graft alone, no cell supplementation |
| `A` | Graft + adipose-derived cells |
| `B` | Graft + bone marrow cells |
| `O` | Graft + olfactory mucosa cells |
| `12w` / `16w` | Weeks post-surgery |
| `L` | Operated paw (grafted) |
| `R` | Contralateral healthy paw (control) |

Filename convention: `{allo|auto}{A|B|O|X}{timepoint}{animal}{L|R}.tif`
Example: `alloB12w3L.tif` — allograft + bone marrow cells, 12 weeks, animal 3, operated paw.

### Key Morphometric Outcomes

- **G-ratio** — axon diameter / fiber diameter. Optimal myelination ≈ 0.6.
  Measures the efficiency of myelin deposition relative to axon size.
- **N-ratio** — total fiber area / nerve cross-section area. Measures how
  densely packed myelinated fibers are within the nerve.
- **AVF / MVF** — axon and myelin volume fractions (proportion of nerve area
  occupied by axon or myelin compartments).
- **Axon density** — number of myelinated axons per mm².

---

## Pipeline Overview

```
TIFF image
  │
  ├─ U-Net inference (ResNet34 encoder, 3 classes: bg / myelin / axon)
  │    Sliding window (512 px tiles, 50% overlap), predictions averaged
  │    → _cellpose_outer.npy  (fiber instance labels)
  │    → _axon_inner.npy      (axon instance labels, matched to fiber IDs)
  │
  ├─ Fascicle mask  (drawn by clinician in web UI, or auto-estimated)
  │
  ├─ QC filtering   (g-ratio, axon position, border, size)
  │
  ├─ Morphometrics  (per-fiber ECD, g-ratio, myelin thickness, AVF, MVF…)
  │
  └─ Outputs
       _overlay.png         — color-coded segmentation
       _morphometrics.csv   — per-axon measurements
       _aggregate.csv       — image-level statistics
       _dashboard.png       — histograms + aggregate table
```

**Design decisions:**

| Decision | Rationale |
|---|---|
| Custom-trained U-Net | Off-the-shelf segmentation tools (Cellpose) do not discriminate axon from myelin; a pixel-level model trained on manual annotations is required |
| Tile-based inference with 50% overlap | Large TIFFs (>6000 px) cannot be processed in one pass; tile overlap eliminates boundary artifacts via prediction averaging |
| Manual fascicle mask | The nerve boundary varies in shape and is not reliably estimated from pixel intensity; a clinician-drawn mask ensures a correct anatomical denominator for all area fractions |
| Multi-core Voronoi partition | When the model merges two adjacent fibers into one label, the fiber area is partitioned among axon blobs by Voronoi tessellation so each axon still contributes a valid individual measurement without discarding the pair |
| Image-level statistics for group comparisons | Fiber-level pooling introduces pseudo-replication; all group comparisons use one aggregate value per image (≈ one animal) as the statistical unit |

---

## Recommended Workflow

```
1. Prepare images
   - Adjust contrast/brightness in ImageJ/Fiji if needed
   - Place TIFFs in edited/

2. Run U-Net inference
   python -m train.predict --image edited/GROUP/stem.tif
   # or batch:
   ./reinfer.sh

3. Start the web app
   python app.py  →  http://127.0.0.1:8000

4. Draw fascicle boundary for each image
   Mode 7 (Fascicle) → click polygon points → close near first point

5. Review segmentation
   - Delete false-positive fibers (mode 2)
   - Correct axon boundaries (mode 3) or fiber boundaries (mode 4)
   - Erase artifacts (mode 5)
   - Add missing fibers (mode 6)
   - Draw exclusion zones for tears/folds (mode 8)
   - Manually accept a QC-rejected fiber (mode 9)

6. Recompute morphometrics
   ↻ Recompute (single image)  or  ↻ All (batch, background)

7. Run comparative analysis
   python compare_all.py
   → output/comparison/rigorous/
```

---

## Web Validation UI

`app.py` is a FastAPI browser application for segmentation review and manual
correction. Supports desktop and iPad with Apple Pencil.

```bash
python app.py
# → http://127.0.0.1:8000  (no password on localhost)
# → password required via Cloudflare tunnel

APP_PASSWORD=mypassword python app.py  # fixed password for remote
```

### Editing Modes

| Key | Mode | Action |
|---|---|---|
| 1 | Navigate | Pan / zoom |
| 2 | Delete | Click a fiber to remove it |
| 3 | Axon | Lasso → paint axon inside an existing fiber |
| 4 | Myelin | Lasso → extend or create a fiber boundary |
| 5 | Erase | Lasso → erase myelin + axon inside zone |
| 6 | Fiber | Two-step lasso → add a complete new fiber (outer then axon) |
| 7 | Fascicle | Click polygon → draw fascicle boundary |
| 8 | Exclude | Polygon → mark exclusion zone (subtracted from nerve area) |
| 9 | Accept | Click a QC-rejected fiber to manually accept it |

### Overlay Color Scheme

| Color | Meaning |
|---|---|
| Green | Axon — QC passed |
| Blue | Myelin ring — QC passed |
| Red | Fiber with no axon detected |
| Crimson | Multi-core fiber (resolved by Voronoi, shown split) |
| White line | Fascicle boundary |

Rejected fibers are not shown (hidden to reduce visual clutter; count reported
in the dashboard).

### GT Annotation Mode

Switch with the **GT** button. Used to produce ground-truth labels for model
training:
- Mode 6 (+Fiber) draws fiber and axon outlines from scratch
- Mode 0 (Vessel) marks blood vessels
- **Mark Validated** locks the annotation for training

---

## U-Net Architecture and Training

### Architecture

- **Encoder:** ResNet34 pretrained on ImageNet
- **Decoder:** U-Net with skip connections (segmentation-models-pytorch)
- **Output:** 3-class pixel classification — background (0), myelin (1), axon (2)
- **Input tile:** 512 × 512 px, RGB

### Loss Function

Combined Dice + Cross-Entropy loss:

$$\mathcal{L} = w_{\text{dice}} \cdot \mathcal{L}_{\text{Dice}} + w_{\text{ce}} \cdot \mathcal{L}_{\text{CE}}$$

The Dice component handles class imbalance at the object level (rewards overlap
between predicted and ground-truth regions). The cross-entropy component
stabilizes gradient flow at the pixel level.

**Class weights** are set proportional to the inverse square root of class
frequency to counteract the severe imbalance between background (dominant),
myelin (moderate), and axon (rare):

$$w_c = \frac{1}{\sqrt{f_c}} \quad \text{normalized so } \sum_c w_c = 1$$

Empirically derived from pixel-level class frequencies in the annotated dataset:
`[0.48, 1.11, 1.42]` for background / myelin / axon. Upweighting the axon
class relative to background prevents the model from learning a trivial
solution that ignores small axon blobs.

### Training Procedure

```bash
# Train from scratch
python -m train.train --epochs 120 --batch 8

# Fine-tune from existing checkpoint
python -m train.train --resume train/checkpoints/best.pt --epochs 30

# Resume interrupted run
python -m train.train --resume train/checkpoints/last.pt --epochs 20

# With parallel data loading (recommended on macOS)
python -m train.train --workers 2
```

**Optimizer:** AdamW, lr=1e-3, weight decay=1e-4  
**Scheduler:** Cosine annealing (T_max = num_epochs)  
**Augmentation:** Random flips, rotations, color jitter, elastic deformation  
**Validation:** Random hold-out split (VAL_SPLIT fraction of annotated stems)  
**Metric:** mIoU on foreground classes (myelin + axon), ignoring background

Checkpoints saved to `train/checkpoints/`:
- `best.pt` — best validation mIoU
- `last.pt` — last completed epoch
- `epoch_NNN.pt` — snapshot every 10 epochs

### Re-inference

After retraining or bug fixes, update the segmentation caches:

```bash
./reinfer.sh          # all images
./reinfer.sh --dry-run  # preview only
./reinfer.sh stem1 stem2  # specific images
```

Then use **↻ All** in the web app to regenerate morphometrics.

### Ground Truth Data

`ground_truth/` is **read-only**. It contains manually annotated reference
images by the clinician and is the sole source of training labels.
Never modify, overwrite, or delete any file in this directory.

```
ground_truth/
├── images/       # raw images (lossless)
└── masks/
    ├── <stem>_outer_gt.npy     # fiber instance labels
    ├── <stem>_axon_gt.npy      # axon instance labels
    └── <stem>_vessels_gt.npy   # blood vessel labels
```

Validation against GT: `python compare_gt.py`
Current performance: mIoU(fg) = 0.886, Fiber F1 = 0.986 (n=7 annotated images)

---

## Morphometrics

**File:** `morphometrics.py`

All measurements are computed on bounding-box crops (not full images) to keep
memory and runtime linear in the number of fibers.

| Metric | Formula | Notes |
|---|---|---|
| Equivalent circular diameter | $d = \sqrt{4A/\pi} \cdot s$ | $s$ = pixel size in µm/px |
| G-ratio | $g = d_\text{axon} / d_\text{fiber}$ | Per fiber |
| Myelin thickness | $(d_\text{fiber} - d_\text{axon}) / 2$ | |
| Axon area | $A_\text{axon} \cdot s^2$ | µm² |
| Fiber area | $A_\text{fiber} \cdot s^2$ | µm² |
| Centroid offset | $\|\mathbf{c}_\text{axon} - \mathbf{c}_\text{fiber}\| / r_\text{fiber}$ | Normalized by fiber radius |

**Aggregate (image-level):**

$$g_\text{area-weighted} = \sqrt{\frac{\sum_i A_{\text{axon},i}}{\sum_i A_{\text{fiber},i}}}$$

$$\text{AVF} = \frac{\sum_i A_{\text{axon},i}}{A_\text{nerve}} \qquad \text{MVF} = \frac{\sum_i (A_{\text{fiber},i} - A_{\text{axon},i})}{A_\text{nerve}}$$

$$\text{N-ratio} = \frac{\sum_{\text{all}} A_{\text{fiber},i}}{A_\text{nerve} - A_\text{exclusion}}$$

N-ratio counts ALL detected fibers (QC-passed and rejected) because every
myelinated fiber physically occupies nerve cross-section regardless of
measurement quality. AVF and MVF use QC-passed fibers only.

### Multi-Core Fibers

When the U-Net assigns a single fiber label to a region containing two
distinct axon blobs (adjacent fibers not separated by the model), the fiber
pixels are partitioned by Voronoi tessellation: each axon blob inherits the
fiber pixels geometrically closest to it. This yields two virtual fiber labels,
each with its own axon area and fiber area, enabling a valid individual g-ratio
for each without discarding the pair.

---

## QC Filtering

**File:** `qc.py`

Each fiber is tested sequentially. The first failing test determines its
rejection code (shown on the overlay and counted in the dashboard).

| Code | Filter | Threshold |
|---|---|---|
| `out` | Axon pixels outside fiber boundary | > 5% |
| `G` | G-ratio out of physiological range | outside [0.1, 0.9] |
| `lgG` | Large fiber (≥ p85) with low g-ratio | g < 0.15 |
| `Ø` | Axon diameter too small | < 0.1 µm |
| `brd` | Fiber touches image border | always excluded |

Thresholds are permissive by default; adjust in `config.py`. A clinician can
override individual rejections using mode 9 (Accept) in the web UI.

---

## Statistical Analysis

**Script:** `python compare_all.py`

Produces four figures and a complete statistics CSV in
`output/comparison/rigorous/`:

| Output | Description |
|---|---|
| `01_overview.png` | 9 groups × 6 metrics, bars = mean ± 95% CI, stars vs healthy |
| `02_recovery_heatmap.png` | % recovery toward healthy control per group × metric |
| `03_timepoint.png` | 12w vs 16w comparison within each group |
| `04_violins.png` | Fiber-level distributions for g-ratio, axon diameter, myelin thickness |
| `stats_complets.csv` | Full table: n, mean, SD, CI95, p (raw), p (Bonferroni × 8), Cohen's d, % recovery |

**Statistical unit:** one image ≈ one animal. Fiber-level values are not used
directly in group tests to avoid pseudo-replication.

**Tests:** Mann-Whitney U (non-parametric, no normality assumption).
Multiple comparisons corrected by Bonferroni (× 8 graft groups vs healthy).

**Stem parsing:** group membership is inferred from the filename via regex
`(allo|auto)([ABOX])(\d+w)\d*(L|R)`. Files ending in `R` are pooled as
healthy controls; files ending in `L` are grafted.

---

## Configuration Reference

All tunable parameters in `config.py`:

```python
PIXEL_SIZE  = 0.09   # µm/px at acquisition resolution
INPUT_DIR   = Path("edited")
OUTPUT_DIR  = Path("output")

FIBER_DIAM_UM = 5.0  # expected fiber diameter — used for fascicle masking

MIN_AXON_SIZE  = 40  # minimum axon blob area (px²) — smaller blobs ignored
OUTER_ERODE_PX = 2   # erode fiber mask before morphometrics (px)
GRATIO_MAP     = False  # spatial g-ratio heatmap (slow, enable only if needed)

# QC thresholds
AXON_OUT_MAX_FRAC      = 0.05   # max fraction of axon pixels outside fiber
MIN_GRATIO             = 0.1
MAX_GRATIO             = 0.9
LARGE_FIBER_MIN_GRATIO = 0.15
LARGE_FIBER_PERCENTILE = 85
MIN_AXON_DIAM_UM       = 0.1
EXCLUDE_BORDER         = True

# Satellite / low-QC cluster removal (applied when no fascicle mask)
MIN_SATELLITE_NEIGHBORS = 15
MIN_CLUSTER_QC_RATE     = 0.50
MIN_CLUSTER_FRACTION    = 0.10
```

---

## Output Files

For each input image `stem.tif`, results are written to `output/stem/`:

| File | Description |
|---|---|
| `stem_cellpose_outer.npy` | U-Net fiber instance labels **(cache)** |
| `stem_axon_inner.npy` | U-Net axon instance labels **(cache)** |
| `stem_fascicle_mask.npy` | Auto-computed fascicle mask |
| `stem_fascicle_mask_edited.npy` | Clinician-drawn fascicle mask (takes priority) |
| `stem_exclusion_mask.npy` | User-drawn exclusion zones |
| `stem_overlay.png` | Color-coded segmentation overlay |
| `stem_numbered.png` | Overlay with numbered QC-passed axons + stats banner |
| `stem_dashboard.png` | Histograms + aggregate metrics table |
| `stem_morphometrics.csv` | Per-axon measurements (QC-passed only) |
| `stem_morphometrics.xlsx` | Same in Excel format |
| `stem_aggregate.csv` | Image-level aggregates (AVF, MVF, N-ratio, g-ratio, density) |
| `stem_outer_edited.npy` | U-Net labels after clinician deletions/corrections |
| `stem_outer_gt.npy` | Complete GT: U-Net + all manual additions |
| `stem_qc_overrides.json` | Fiber IDs manually accepted via mode 9 |

Global: `output/summary.csv` — one aggregate row per processed image.

---

## Module Structure

```
hybrid-axon-seg/
├── segment.py          # CLI entry — orchestrates full pipeline
├── app.py              # Web validation UI (FastAPI)
├── config.py           # All tunable parameters
├── morphometrics.py    # Per-fiber geometry + image-level aggregates
├── qc.py               # QC filters with rejection reason codes
├── visualization.py    # Overlay, numbered image, dashboard
├── compare_all.py      # Full rigorous statistical analysis (9 groups)
├── compare_gt.py       # Pipeline validation vs ground-truth annotations
├── compare.py          # Generic cross-sample comparison dashboard
├── utils.py            # Satellite/cluster detection, shared I/O helpers
├── reinfer.sh          # Re-run U-Net inference on all or specific images
├── backup_gt.sh        # Backup ground_truth/ + model checkpoints
├── static/             # Web UI (app.js, style.css, index.html)
└── train/
    ├── train.py        # Training loop (AdamW + cosine LR, DiceCE loss)
    ├── predict.py      # Sliding-window inference → instance labels
    ├── dataset.py      # GTDataset — tile extraction from ground_truth/
    ├── model.py        # build_model() — ResNet34 U-Net via smp
    ├── losses.py       # DiceCE loss with per-class weighting
    ├── config.py       # Training hyperparameters
    └── checkpoints/    # best.pt, last.pt, epoch_NNN.pt
```
