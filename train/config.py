"""train/config.py — All hyperparameters for the U-Net axon/myelin segmentation model."""

from pathlib import Path

import torch

# ── Paths ────────────────────────────────────────────────────────────────────
GT_DIR = Path("ground_truth")
GT_IMAGES = GT_DIR / "images"
GT_MASKS = GT_DIR / "masks"
CHECKPOINT_DIR = Path("train/checkpoints")

# ── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE = 512       # px — square tiles fed to the network
TILE_STRIDE = 128     # 75% overlap → each pixel covered by centre of multiple tiles
MIN_FG_FRAC = 0.05    # skip tiles with < 5% annotated pixels (mostly background)

# ── Model ────────────────────────────────────────────────────────────────────
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"  # ImageNet pre-training for the ResNet34 encoder
NUM_CLASSES = 3               # 0 = background, 1 = myelin sheath, 2 = axon

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_EPOCHS = 120
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
VAL_SPLIT = 0.0   # 0 = train on all stems (no held-out val); real validation is visual inspection

# ── Loss ─────────────────────────────────────────────────────────────────────
DICE_WEIGHT = 0.5
CE_WEIGHT = 0.5
# Upweight axon class — it's spatially rare relative to background
CLASS_WEIGHTS = [0.48, 1.11, 1.42]  # [bg, myelin, axon] — sqrt-inverse-frequency

# ── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# AMP (mixed precision) — only stable on CUDA; MPS and CPU use fp32
USE_AMP = DEVICE == "cuda"
