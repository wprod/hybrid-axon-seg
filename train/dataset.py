"""train/dataset.py — Ground-truth dataset: load, tile, augment."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train import config as C


# ── Helpers ──────────────────────────────────────────────────────────────────

def annotated_stems() -> list[str]:
    """Return stems that have both outer_gt.npy and axon_gt.npy masks."""
    stems = []
    seen: set[str] = set()
    for img_file in sorted(C.GT_IMAGES.iterdir()):
        if img_file.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg"}:
            continue
        stem = img_file.stem
        if stem in seen:
            continue
        seen.add(stem)
        outer = C.GT_MASKS / f"{stem}_outer_gt.npy"
        axon = C.GT_MASKS / f"{stem}_axon_gt.npy"
        if outer.exists() and axon.exists():
            stems.append(stem)
    return stems


def load_image_rgb(stem: str) -> np.ndarray:
    """Load TIF as uint8 RGB (H, W, 3) — normalises 16-bit inputs."""
    for ext in (".tiff", ".tif", ".png", ".jpg"):
        path = C.GT_IMAGES / f"{stem}{ext}"
        if path.exists():
            break
    img = np.array(Image.open(path))

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Normalise to uint8 if 16-bit
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = img.min(), img.max()
        img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)

    return img


def build_semantic_mask(stem: str) -> np.ndarray:
    """Merge instance masks into 3-class semantic mask (uint8).

    Classes:
        0 — background
        1 — myelin sheath  (pixels in outer_gt but not axon_gt)
        2 — axon           (pixels in axon_gt)
    """
    outer = np.load(str(C.GT_MASKS / f"{stem}_outer_gt.npy"))
    axon = np.load(str(C.GT_MASKS / f"{stem}_axon_gt.npy"))
    semantic = np.zeros_like(outer, dtype=np.uint8)
    semantic[outer > 0] = 1
    semantic[axon > 0] = 2  # axon overrides myelin
    return semantic


def extract_tiles(
    img: np.ndarray,
    mask: np.ndarray,
    tile_size: int = C.TILE_SIZE,
    stride: int = C.TILE_STRIDE,
    min_fg: float = C.MIN_FG_FRAC,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slide a window over the image and return non-trivial (img, mask) tile pairs."""
    H, W = img.shape[:2]
    tiles = []
    for y in range(0, max(H - tile_size + 1, 1), stride):
        for x in range(0, max(W - tile_size + 1, 1), stride):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            img_tile = img[y1:y2, x1:x2]
            mask_tile = mask[y1:y2, x1:x2]

            # Pad if tile is smaller than tile_size (edges of large images)
            if img_tile.shape[0] < tile_size or img_tile.shape[1] < tile_size:
                pad_h = tile_size - img_tile.shape[0]
                pad_w = tile_size - img_tile.shape[1]
                img_tile = np.pad(img_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                mask_tile = np.pad(mask_tile, ((0, pad_h), (0, pad_w)), mode="constant")

            # Skip tiles that are almost entirely background
            if (mask_tile > 0).mean() < min_fg:
                continue

            tiles.append((img_tile, mask_tile))
    return tiles


# ── Augmentation ─────────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(augment: bool) -> A.Compose:
    if not augment:
        return A.Compose([
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ])

    return A.Compose([
        # Geometry
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5,
                           border_mode=0),
        # Photometric — toluidine blue staining can vary slightly between batches
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.3),
        # Noise / blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.12), p=0.3),
        # Elastic deformation — mimics section cutting artefacts
        A.ElasticTransform(alpha=80, sigma=8, p=0.3),
        # Normalise + convert to tensor
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Dataset ──────────────────────────────────────────────────────────────────

class GTDataset(Dataset):
    """Tile-based dataset built from ground-truth annotations.

    Args:
        stems:   list of image stems to include (each must have masks).
        augment: whether to apply training augmentations.
    """

    def __init__(self, stems: list[str], augment: bool = True) -> None:
        self.transforms = get_transforms(augment)
        self.tiles: list[tuple[np.ndarray, np.ndarray]] = []

        for stem in stems:
            img = load_image_rgb(stem)
            mask = build_semantic_mask(stem)
            new_tiles = extract_tiles(img, mask)
            self.tiles.extend(new_tiles)
            print(f"  {stem}: {img.shape[:2]} → {len(new_tiles)} tiles")

        if not self.tiles:
            raise RuntimeError(
                f"No usable tiles found for stems {stems}. "
                "Check that _outer_gt.npy and _axon_gt.npy exist in ground_truth/masks/."
            )

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.tiles[idx]
        out = self.transforms(image=img, mask=mask)
        return out["image"], out["mask"].long()
