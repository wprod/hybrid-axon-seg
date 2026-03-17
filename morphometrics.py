"""morphometrics.py — Per-fiber geometry computation.

All operations work on small bounding-box crops (not full-image masks) to
keep memory usage and runtime linear in the number of fibers.

Measurements
------------
- axon / fiber equivalent diameter  (ECD = √(4A/π))
- g-ratio  = d_axon / d_fiber
- myelin thickness  = (d_fiber − d_axon) / 2
- axon solidity and eccentricity
- centroid offset  (normalised by fiber radius)
- border-touching flag

Aggregate outputs: AVF, MVF, aggregate g-ratio, axon density (mm⁻²).
"""

import numpy as np
import pandas as pd
from skimage import measure


def process_fibers(
    outer_labels: np.ndarray,
    axon_assignments: dict,  # fiber_label → (r0, c0, crop_bool)
    pixel_size: float,  # µm / px
) -> tuple[np.ndarray, dict, pd.DataFrame, np.ndarray, dict]:
    """Compute morphometrics for every detected axon.

    Returns
    -------
    inner_labels  : label image (fiber_label value at each axon pixel)
    pairs         : dict  fiber_label → fiber_label  (matched fibers)
    df            : DataFrame with one row per axon
    index_image   : int32 array mapping pixels to sequential axon IDs
    agg           : dict of aggregate metrics (AVF, MVF, …)
    """
    fiber_rprops = {p.label: p for p in measure.regionprops(outer_labels)}
    img_h, img_w = outer_labels.shape

    inner_labels = np.zeros_like(outer_labels)
    index_image = np.zeros(outer_labels.shape, dtype=np.int32)
    pairs = {}
    rows = []

    for axon_id, (fiber_label, (r0, c0, crop_axon)) in enumerate(axon_assignments.items(), start=1):
        fp = fiber_rprops.get(fiber_label)
        if fp is None:
            continue

        area_outer = int(fp.area)
        area_inner = int(crop_axon.sum())
        if area_inner == 0:
            continue

        d_outer = np.sqrt(4 * area_outer / np.pi) * pixel_size
        d_inner = np.sqrt(4 * area_inner / np.pi) * pixel_size
        gratio = d_inner / d_outer
        myelin_thickness = (d_outer - d_inner) / 2

        axon_coords = np.argwhere(crop_axon)
        y0 = float(axon_coords[:, 0].mean()) + r0
        x0 = float(axon_coords[:, 1].mean()) + c0

        fy, fx = fp.centroid
        offset_px = np.sqrt((y0 - fy) ** 2 + (x0 - fx) ** 2)
        fiber_radius_px = np.sqrt(area_outer / np.pi)
        centroid_offset = float(offset_px / fiber_radius_px) if fiber_radius_px > 0 else 1.0

        axon_rprops = measure.regionprops(measure.label(crop_axon.astype(np.uint8)))
        best_region = max(axon_rprops, key=lambda r: r.area) if axon_rprops else None
        solidity = best_region.solidity if best_region else 1.0
        eccentricity = best_region.eccentricity if best_region else 0.0

        minr, minc, maxr, maxc = fp.bbox
        border = minr == 0 or minc == 0 or maxr == img_h or maxc == img_w

        h, w = crop_axon.shape
        inner_labels[r0 : r0 + h, c0 : c0 + w][crop_axon] = fiber_label
        index_image[r0 : r0 + h, c0 : c0 + w][crop_axon] = axon_id
        pairs[fiber_label] = fiber_label

        rows.append(
            {
                "axon_diam": d_inner,
                "fiber_diam": d_outer,
                "gratio": gratio,
                "myelin_thickness": myelin_thickness,
                "axon_area": area_inner * pixel_size**2,
                "fiber_area": area_outer * pixel_size**2,
                "solidity": solidity,
                "eccentricity": eccentricity,
                "centroid_offset": centroid_offset,
                "x0": x0,
                "y0": y0,
                "image_border_touching": border,
                "_label": axon_id,
                "_fiber_label": fiber_label,
            }
        )

    df = pd.DataFrame(rows)
    n_ok = len(rows)
    n_fail = int(outer_labels.max()) - n_ok
    print(f"       -> {n_ok} fibers with axon, {n_fail} without (shown in red)")

    h, w = outer_labels.shape
    total_um2 = h * w * pixel_size**2
    total_axon = df["axon_area"].sum() if len(df) else 0.0
    total_fib = df["fiber_area"].sum() if len(df) else 0.0
    agg = {
        "avf": total_axon / total_um2 if total_um2 else 0,
        "mvf": (total_fib - total_axon) / total_um2 if total_um2 else 0,
        "gratio_aggr": df["gratio"].mean() if len(df) else 0,
        "axon_density_mm2": len(df) / (total_um2 * 1e-6) if total_um2 else 0,
    }

    return inner_labels, pairs, df, index_image, agg
