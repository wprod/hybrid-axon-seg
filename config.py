"""config.py — All pipeline constants. Edit here to tune the pipeline."""

from pathlib import Path

# I/O
INPUT_DIR = Path("edited")
OUTPUT_DIR = Path("output")
PIXEL_SIZE = 0.09  # µm/px at source resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL = "cpsam"
CP_DIAM_UM = 5.0
CP_FLOW_THR = 0.3
CP_CELLPROB = -1.0
CP_CLAHE = True  # apply percentile stretch before Cellpose to boost contrast
CLAHE_CLIP_LIMIT = 0.0  # 0 = stretch only (no CLAHE); >0 = stretch + CLAHE
CLAHE_TILE_SIZE = (64, 64)  # local tile size for CLAHE kernel (used only if clip_limit > 0)
STRETCH_PLOW = 2  # low percentile for stretch
STRETCH_PHIGH = 98  # high percentile for stretch

# Inversion / preprocessing
MASK_ERODE_PX = 4  # px stripped from fiber boundary before normalization
FADE_PX = 8  # px of inward fade from the eroded edge
MIN_AXON_SIZE = 40  # min axon blob area (px²)
AXON_SMOOTH_SIGMA = 1.5  # Gaussian sigma for perimeter smoothing (0 = off)
AXON_DILATE_PX = 3  # expand axon mask after convex hull (0 = off; convex hull already fills)
AXON_OPEN_PX = (
    2  # morphological opening radius after dilation: removes thin notches/protrusions (0 = off)
)
AXON_MIN_MYELIN_PX = 2  # hard floor: axon must be ≥ this many px from fiber edge
AXON_MIN_MYELIN_FRAC = 0.30  # adaptive floor: axon must also be ≥ frac×outer_radius from edge
# (larger fibers → thicker myelin → larger minimum gap)
OUTER_ERODE_PX = 2  # erode fiber mask before morphometrics (shrinks fiber → less myelin)
AXON_INPUT_WHITE_POINT = 255  # clip white point of axon_input (pixels above → 255); 255 = off
GRATIO_MAP = False  # spatial g-ratio heatmap (slow — enable only if needed)

# QC filters  (permissive by default — clinician adjusts)
MIN_GRATIO = 0.3
MAX_GRATIO = 0.9
LARGE_FIBER_MIN_GRATIO = 0.5  # large fibers with g-ratio below this → likely fail (axon trop grand)
LARGE_FIBER_PERCENTILE = 85  # only apply to fibers ≥ this size percentile (top 15%)
MIN_AXON_DIAM_UM = 0.5
MIN_SOLIDITY = 0.1
MAX_SHAPE_DISCORDANCE = (
    0.2  # max (fiber_solidity - axon_solidity); high = fibre ronde mais axone irrégulier
)
MAX_CENTROID_OFFSET = 0.95
MAX_AXON_ECCEN = 1.0  # effectively disabled — set < 1.0 to re-enable
EXCLUDE_BORDER = True

# Satellite detection — fibers with fewer than this many neighbours
# within 5× fiber diameter are considered satellites and removed
MIN_SATELLITE_NEIGHBORS = 15
MIN_CLUSTER_QC_RATE = 0.50  # clusters with QC pass rate below this are removed
MIN_CLUSTER_FRACTION = 0.10  # clusters with < 10% of the largest cluster's fiber count are removed
