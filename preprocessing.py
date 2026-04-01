"""preprocessing.py — Per-fiber normalized-inversion to build axon_input image.

Strategy
--------
For each Cellpose pass-1 fiber:
  1. Erode the mask by MASK_ERODE_PX to exclude endoneurium contamination at
     the Cellpose boundary.
  2. Percentile-stretch (p5–p95) the eroded interior, then invert:
       axon  (bright original) → dark blob  ≈ 0
       myelin (dark original)  → bright ring ≈ 255
  3. Apply a smooth fade from white at the eroded edge to fully inverted at
     FADE_PX px inward, preventing any hard black ring artefact.

The resulting full-image `axon_input` is then thresholded by `find_axons`
in detection.py.
"""

import numpy as np
from scipy import ndimage
from skimage import measure

import config
from utils import to_uint8_gray


def _invert_crop(crop_gray: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
    """Contrast-stretch + invert one fiber crop with a smooth inward fade.

    Erosion is adaptive: capped at 25% of the fiber's equivalent radius so
    that thin-myelin or small fibers don't lose their myelin signal entirely.
    """
    dist = ndimage.distance_transform_edt(fiber_mask).astype(np.float32)
    fiber_radius = float(dist.max())  # ≈ equivalent radius of this fiber
    erode_px = min(config.MASK_ERODE_PX, max(1, fiber_radius * 0.25))
    inner = dist > erode_px
    if not inner.any():
        inner = fiber_mask

    pixels = crop_gray[inner].astype(np.float32)
    lo, hi = np.percentile(pixels, 5), np.percentile(pixels, 95)
    if hi > lo:
        stretched = np.clip((crop_gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    else:
        stretched = crop_gray.astype(np.float32)

    inverted = 255.0 - stretched
    inner_dist = ndimage.distance_transform_edt(inner).astype(np.float32)
    fade = np.clip(inner_dist / max(config.FADE_PX, 1), 0.0, 1.0)
    blended = (255.0 + (inverted - 255.0) * fade).clip(0, 255).astype(np.uint8)

    result = np.full(crop_gray.shape, 255, dtype=np.uint8)
    result[inner] = blended[inner]
    return result


def build_axon_input(img: np.ndarray, outer_labels: np.ndarray) -> np.ndarray:
    """Build the full-image normalized-inverted map (saved as *_axon_input.png).

    axon_input:  axon ≈ 0 (dark),  myelin ≈ 255 (bright),  background = 255
    """
    gray = to_uint8_gray(img)
    result = np.full_like(gray, 255, dtype=np.uint8)
    for p in measure.regionprops(outer_labels):
        minr, minc, maxr, maxc = p.bbox
        crop = _invert_crop(gray[minr:maxr, minc:maxc], p.image)
        result[minr:maxr, minc:maxc][p.image] = crop[p.image]
    wp = getattr(config, "AXON_INPUT_WHITE_POINT", 255)
    if wp < 255:
        result = np.clip(result.astype(np.uint16) * 255 // wp, 0, 255).astype(np.uint8)
    return result
