"""config.py — All pipeline constants. Edit here to tune the pipeline."""

from pathlib import Path

# I/O
INPUT_DIR = Path("edited")
OUTPUT_DIR = Path("output")
GT_DIR = Path("ground_truth")
PIXEL_SIZE = 0.09  # µm/px at source resolution

# Fiber geometry — used for fascicle masking + satellite removal
FIBER_DIAM_UM = 5.0

# Axon
MIN_AXON_SIZE = 40  # min axon blob area (px²)
OUTER_ERODE_PX = 2  # erode fiber mask before morphometrics (shrinks fiber → less myelin)
GRATIO_MAP = False  # spatial g-ratio heatmap (slow — enable only if needed)

# Axon shrink — applied before morphometrics to correct over-segmented axons
# shrink_px = AXON_SHRINK_K * sqrt(d_px)  (non-linear: large axons shrink more)
# At d=24px (≈2µm axon): shrink≈0.5px.  At d=100px (≈9µm): shrink≈1px.
AXON_SHRINK_PX = 2  # erode every axon mask by this many pixels before morphometrics

# QC: axon-outside-fiber — reject if axon pixels outside fiber > this fraction
AXON_OUT_MAX_FRAC = 0.05  # 5% tolerance for segmentation imprecision

# QC filters  (permissive by default — clinician adjusts)
MIN_GRATIO = 0.1
MAX_GRATIO = 0.9
LARGE_FIBER_MIN_GRATIO = 0.15  # large fibers (≥ p85) with g-ratio below this → rejected
LARGE_FIBER_PERCENTILE = 85
MIN_AXON_DIAM_UM = 0.1
EXCLUDE_BORDER = True

# Satellite detection — fibers with fewer than this many neighbours
# within 5× fiber diameter are considered satellites and removed
MIN_SATELLITE_NEIGHBORS = 15
MIN_CLUSTER_QC_RATE = 0.50  # clusters with QC pass rate below this are removed
MIN_CLUSTER_FRACTION = 0.10  # clusters with < 10% of the largest cluster's fiber count are removed
