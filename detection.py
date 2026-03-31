"""detection.py — Cellpose pass 1 (outer fibers) + Otsu axon detection.

Pass 1 : Cellpose cyto3 on the raw image → outer fiber label map.
         This is the source of truth for fiber COUNT.

Axons  : Global Otsu threshold on `axon_input` (the normalized-inverted image
         produced by preprocessing.py).  All fiber pixels together give a clean
         bimodal distribution.  Per-fiber: select the CC at the centroid (or
         largest CC if centroid falls on background), then fill holes.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu

import config


def _clahe_preprocess(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the green channel (or grayscale) to boost local contrast.

    Works on uint8 or uint16 input; always returns uint8 suitable for Cellpose.
    Operates channel-by-channel if RGB, otherwise on the single channel.
    """
    from skimage import exposure
    from skimage.util import img_as_ubyte

    # Clip range (tiny percentile for robustness against hot pixels)
    clip = getattr(config, "CLAHE_CLIP_LIMIT", 0.02)
    tile = getattr(config, "CLAHE_TILE_SIZE", (64, 64))

    if img.ndim == 3:
        out = np.empty_like(img if img.dtype == np.uint8 else img.astype(np.uint8))
        for c in range(img.shape[2]):
            ch = img[:, :, c]
            ch_f = ch.astype(np.float32) / (
                np.iinfo(ch.dtype).max if np.issubdtype(ch.dtype, np.integer) else 1.0
            )
            out[:, :, c] = img_as_ubyte(
                np.clip(exposure.equalize_adapthist(ch_f, kernel_size=tile, clip_limit=clip), 0, 1)
            )
        return out
    else:
        ch_f = img.astype(np.float32) / (
            np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0
        )
        return img_as_ubyte(
            np.clip(exposure.equalize_adapthist(ch_f, kernel_size=tile, clip_limit=clip), 0, 1)
        )


def run_cellpose_fibers(img: np.ndarray) -> np.ndarray:
    """Run Cellpose (cyto3) on *img* → integer label array (0 = background)."""
    import torch
    from cellpose import models

    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    device = (
        "MPS"
        if torch.backends.mps.is_available()
        else "CUDA"
        if torch.cuda.is_available()
        else "CPU"
    )
    print(f"       device: {device}")

    if getattr(config, "CP_CLAHE", False):
        print("       applying CLAHE…")
        img = _clahe_preprocess(img)

    model = models.CellposeModel(pretrained_model=config.CP_MODEL, gpu=gpu)
    masks, _, _ = model.eval(
        img,
        diameter=config.CP_DIAM_UM / config.PIXEL_SIZE,
        flow_threshold=config.CP_FLOW_THR,
        cellprob_threshold=config.CP_CELLPROB,
        min_size=config.MIN_AXON_SIZE,
    )
    return masks


def find_axons(axon_input: np.ndarray, outer_labels: np.ndarray) -> tuple[dict, set]:
    """Global Otsu on axon_input → per-fiber centroid-based blob selection.

    Returns
    -------
    tuple:
        axon_assignments : dict  fiber_label → (minr, minc, crop_bool)
            crop_bool is a boolean mask relative to the fiber bounding-box.
        multicore_labels : set of fiber labels rejected for having 2+ dark cores.
    """
    fiber_pixels = axon_input[outer_labels > 0]
    thr = threshold_otsu(fiber_pixels)
    dark_mask = (axon_input <= thr) & (outer_labels > 0)

    result = {}
    multicore_labels = set()
    for p in measure.regionprops(outer_labels):
        minr, minc, maxr, maxc = p.bbox

        crop_dark = dark_mask[minr:maxr, minc:maxc] & p.image

        # Multi-core check on RAW blobs (before any closing that could bridge them)
        # Reject if the 2nd largest blob is ≥ 30% of the largest → true double-core
        raw_labeled = measure.label(crop_dark)
        raw_rprops = sorted(measure.regionprops(raw_labeled), key=lambda r: r.area, reverse=True)
        if len(raw_rprops) >= 2 and raw_rprops[1].area >= raw_rprops[0].area * 0.30:
            multicore_labels.add(p.label)
            continue

        # Now apply small closing to denoise before CC selection
        crop_dark = morphology.binary_closing(crop_dark, morphology.disk(1)) & p.image
        labeled = measure.label(crop_dark)
        if labeled.max() == 0:
            continue

        cy = int(np.clip(p.centroid[0] - minr, 0, maxr - minr - 1))
        cx = int(np.clip(p.centroid[1] - minc, 0, maxc - minc - 1))

        centroid_lbl = labeled[cy, cx]
        if centroid_lbl > 0:
            best = labeled == centroid_lbl
        else:
            rprops = measure.regionprops(labeled)
            best = labeled == max(rprops, key=lambda r: r.area).label

        # Large closing bridges C-shape gaps without fully simplifying the shape,
        # then clip to minimum myelin margin from the fiber boundary
        dist = ndimage.distance_transform_edt(p.image).astype(np.float32)
        margin_mask = dist > config.AXON_MIN_MYELIN_PX
        best = morphology.binary_closing(best, morphology.disk(8)) & margin_mask
        best = ndimage.binary_fill_holes(best) & margin_mask

        # Smooth perimeter: Gaussian blur on the float mask, re-threshold at 0.5
        if config.AXON_SMOOTH_SIGMA > 0:
            best = (
                ndimage.gaussian_filter(best.astype(np.float32), sigma=config.AXON_SMOOTH_SIGMA)
                >= 0.5
            ) & p.image

        # Expand axon mask to compensate for Otsu under-segmentation at axon/myelin boundary.
        # Dilation is adaptive: small fibers (L, ~18 px radius) receive the full correction
        # because the fixed-pixel Otsu error represents a larger fraction of their radius.
        # Large fibers (R, ~39 px radius) receive a reduced correction.
        # Threshold: 25 px radius ≈ 4.5 µm diameter @ 0.09 µm/px.
        if config.AXON_DILATE_PX > 0:
            fiber_radius_px = float(np.sqrt(p.area / np.pi))
            dil_px = (
                config.AXON_DILATE_PX
                if fiber_radius_px < 25
                else max(1, config.AXON_DILATE_PX // 2)
            )
            best = morphology.binary_dilation(best, morphology.disk(dil_px)) & p.image

        if best.sum() < config.MIN_AXON_SIZE:
            continue

        result[p.label] = (minr, minc, best)

    return result, multicore_labels
