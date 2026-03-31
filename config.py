"""config.py — All pipeline constants. Edit here to tune the pipeline."""

from pathlib import Path

# I/O
INPUT_DIR = Path("edited")
OUTPUT_DIR = Path("output")
PIXEL_SIZE = 0.09  # µm/px at source resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL = "cyto3"
CP_DIAM_UM = 7.0  # expected fiber diameter (µm)
CP_FLOW_THR = 0.4
CP_CELLPROB = 0.0
CP_CLAHE = False  # apply CLAHE before Cellpose to boost local contrast
CLAHE_CLIP_LIMIT = 0.02  # CLAHE clip limit (0–1); higher = more aggressive
CLAHE_TILE_SIZE = (64, 64)  # local tile size for CLAHE kernel

# Inversion / preprocessing
MASK_ERODE_PX = 4  # px stripped from fiber boundary before normalization
FADE_PX = 8  # px of inward fade from the eroded edge
MIN_AXON_SIZE = 40  # min axon blob area (px²)
AXON_SMOOTH_SIGMA = 1.5  # Gaussian sigma for perimeter smoothing (0 = off)
AXON_DILATE_PX = 3  # expand axon mask after convex hull (0 = off; convex hull already fills)
AXON_MIN_MYELIN_PX = 2  # minimum myelin ring thickness (clips axon from fiber edge)
OUTER_ERODE_PX = 2  # erode fiber mask before morphometrics (shrinks fiber → less myelin)
AXON_INPUT_WHITE_POINT = 160  # clip white point of axon_input (pixels above → 255); 255 = off

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
