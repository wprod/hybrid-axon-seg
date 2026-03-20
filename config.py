"""config.py — All pipeline constants. Edit here to tune the pipeline."""

from pathlib import Path

# I/O
INPUT_DIR = Path("edited")
OUTPUT_DIR = Path("output")
PIXEL_SIZE = 0.09  # µm/px at source resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL = "cyto3"
CP_DIAM_UM = 8.0  # expected fiber diameter (µm)
CP_FLOW_THR = 0.4
CP_CELLPROB = 0.0

# Inversion / preprocessing
MASK_ERODE_PX = 4  # px stripped from fiber boundary before normalization
FADE_PX = 8  # px of inward fade from the eroded edge
MIN_AXON_SIZE = 40  # min axon blob area (px²)
AXON_SMOOTH_SIGMA = 1.5  # Gaussian sigma for perimeter smoothing (0 = off)
AXON_DILATE_PX = 0  # expand axon mask after convex hull (0 = off; convex hull already fills)
AXON_MIN_MYELIN_PX = (
    5  # minimum myelin ring thickness after convex hull (clips axon from fiber edge)
)
AXON_INPUT_WHITE_POINT = 160  # clip white point of axon_input (pixels above → 255); 255 = off

# QC filters  (permissive by default — clinician adjusts)
MIN_GRATIO = 0.25
MAX_GRATIO = 0.9
LARGE_FIBER_MIN_GRATIO = (
    0.55  # large fibers with g-ratio below this → likely fail (axon trop grand)
)
LARGE_FIBER_PERCENTILE = 85  # only apply to fibers ≥ this size percentile (top 15%)
MIN_AXON_DIAM_UM = 0.5
MIN_SOLIDITY = 0.1
MAX_CENTROID_OFFSET = 0.95
MAX_AXON_ECCEN = 1.0  # effectively disabled — set < 1.0 to re-enable
EXCLUDE_BORDER = True
