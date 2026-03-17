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

    model = models.CellposeModel(pretrained_model=config.CP_MODEL, gpu=gpu)
    masks, _, _ = model.eval(
        img,
        diameter=config.CP_DIAM_UM / config.PIXEL_SIZE,
        flow_threshold=config.CP_FLOW_THR,
        cellprob_threshold=config.CP_CELLPROB,
        min_size=config.MIN_AXON_SIZE,
    )
    return masks


def find_axons(axon_input: np.ndarray, outer_labels: np.ndarray) -> dict:
    """Global Otsu on axon_input → per-fiber centroid-based blob selection.

    Returns
    -------
    dict : fiber_label → (minr, minc, crop_bool)
        crop_bool is a boolean mask relative to the fiber bounding-box.
    """
    fiber_pixels = axon_input[outer_labels > 0]
    thr = threshold_otsu(fiber_pixels)
    dark_mask = (axon_input <= thr) & (outer_labels > 0)

    result = {}
    for p in measure.regionprops(outer_labels):
        minr, minc, maxr, maxc = p.bbox

        crop_dark = dark_mask[minr:maxr, minc:maxc] & p.image
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

        best = ndimage.binary_fill_holes(best) & p.image
        if best.sum() < config.MIN_AXON_SIZE:
            continue

        result[p.label] = (minr, minc, best)

    return result
