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

# QC filters  (permissive by default — clinician adjusts)
MIN_GRATIO = 0.215
MAX_GRATIO = 0.9
MIN_AXON_DIAM_UM = 0.5
MIN_SOLIDITY = 0.4
MAX_CENTROID_OFFSET = 0.65
MAX_AXON_ECCEN = 0.99
EXCLUDE_BORDER = True
