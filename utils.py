"""utils.py — Small shared helpers (image conversion, font loading, stem cleaning)."""

import re
from pathlib import Path

import numpy as np
from PIL import ImageFont


def clean_stem(path: Path) -> str:
    """Strip trailing '(clean).tif[f]' suffix to get a clean output stem."""
    name = path.name
    name = re.sub(r"(?:\s*\(clean\))?\.tiff?$", "", name)
    return name


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any image array to uint8 RGB (H×W×3)."""
    if img.ndim == 2:
        out = np.stack([img] * 3, axis=-1)
    else:
        out = img[:, :, :3].copy()
    if out.dtype in (np.float32, np.float64):
        out = (
            (out * 255).clip(0, 255).astype(np.uint8)
            if out.max() <= 1.0
            else out.clip(0, 255).astype(np.uint8)
        )
    elif out.dtype == np.uint16:
        out = (out / 256).astype(np.uint8)
    else:
        out = out.astype(np.uint8)
    return out


def to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Convert any image array to a 2-D uint8 greyscale array."""
    if img.ndim == 3:
        return to_rgb_uint8(img).mean(axis=2).astype(np.uint8)
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    if img.dtype in (np.float32, np.float64):
        maxv = img.max()
        return ((img / maxv) * 255).astype(np.uint8) if maxv > 0 else img.astype(np.uint8)
    return img.astype(np.uint8)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Return a PIL font at *size* pt, falling back to the built-in default."""
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
