"""utils.py — Small shared helpers (image conversion, font loading, stem cleaning)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
from PIL import ImageFont


def clean_stem(path: Path) -> str:
    """Strip trailing '(clean).tif[f]' suffix (and any remaining .tif) to get a clean stem."""
    name = path.name
    name = re.sub(r"(?:\s*\(clean\))?\.tiff?$", "", name)  # strip (clean).tif
    name = re.sub(r"\.tiff?$", "", name)  # strip any remaining .tif
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


def find_satellite_labels(
    outer_labels: np.ndarray,
    pixel_size: float,
    fiber_diam_um: float,
) -> set[int]:
    """Identify satellite fibers by local neighbour density (KD-tree).

    A fiber is a satellite if it has fewer than MIN_SATELLITE_NEIGHBORS
    neighbours within a radius of 5× the expected fiber diameter.
    Returns the set of fiber labels to remove.
    """
    from scipy.spatial import cKDTree
    from skimage import measure as _measure

    import config as _cfg

    min_neighbors = getattr(_cfg, "MIN_SATELLITE_NEIGHBORS", 15)
    radius_um = fiber_diam_um * 5  # ~35 µm at default settings
    radius_px = radius_um / pixel_size  # ~389 px at 100×

    props = _measure.regionprops(outer_labels)
    if not props:
        return set()

    labels = [p.label for p in props]
    centroids = np.array([[p.centroid[0], p.centroid[1]] for p in props])

    tree = cKDTree(centroids)
    counts = tree.query_ball_point(centroids, r=radius_px, return_length=True)

    # counts includes self, so threshold is min_neighbors + 1
    satellites = set()
    for i, cnt in enumerate(counts):
        if cnt - 1 < min_neighbors:
            satellites.add(labels[i])
    return satellites


def find_low_qc_cluster_labels(
    outer_labels: np.ndarray,
    df_pass: pd.DataFrame,
    df_rej: pd.DataFrame,
    pixel_size: float,
    fiber_diam_um: float,
) -> set[int]:
    """Identify fibers in clusters that are too small or have a low QC pass rate.

    Groups fibers into spatial clusters using centroid-based dilation (NOT a
    closing of the full binary mask, which bridges real gaps with connective
    tissue).  Two fibers are in the same cluster when their centroid disks
    overlap, i.e. their centroids are within ~1.5 × fiber_diam of each other.

    Removes clusters where:
    - fiber count < MIN_CLUSTER_FRACTION of the largest cluster, OR
    - QC pass rate < MIN_CLUSTER_QC_RATE.
    The largest cluster is always kept regardless.
    """
    from skimage import measure as _measure
    from skimage import morphology

    import config as _cfg

    min_rate = getattr(_cfg, "MIN_CLUSTER_QC_RATE", 0.50)
    min_fraction = getattr(_cfg, "MIN_CLUSTER_FRACTION", 0.05)

    fiber_diam_px = fiber_diam_um / pixel_size
    h, w = outer_labels.shape
    scale = max(1, int(round(fiber_diam_px / 4)))
    sh, sw = h // scale, w // scale

    props = _measure.regionprops(outer_labels)
    if not props:
        return set()

    # ── Centroid-based clustering ─────────────────────────────────────────
    # Place a point at each fiber centroid (downscaled), dilate by a disk
    # whose radius ≈ 0.75 × fiber_diam in downscaled space.
    # Two centroids merge iff their disks overlap (gap < ~1.5 × fiber_diam).
    # This avoids bridging visible gaps in connective tissue.
    centroid_map = np.zeros((sh, sw), dtype=bool)
    for p in props:
        cy = min(int(p.centroid[0]) // scale, sh - 1)
        cx = min(int(p.centroid[1]) // scale, sw - 1)
        centroid_map[cy, cx] = True

    # Gap tolerance ≈ 1.5 × fiber_diam: bridges thin internal septa while
    # leaving real satellite gaps (typically >> 3 × fiber_diam) separate.
    link_r = max(1, int(round(fiber_diam_px * 1.5 / scale)))
    dilated = morphology.binary_dilation(centroid_map, morphology.disk(link_r))
    labeled_cc = _measure.label(dilated)

    # Map each fiber label → cluster id via its downscaled centroid
    label_to_cc = {}
    for p in props:
        cy = min(int(p.centroid[0]) // scale, sh - 1)
        cx = min(int(p.centroid[1]) // scale, sw - 1)
        label_to_cc[p.label] = labeled_cc[cy, cx]

    # Count total fibers and QC-passed fibers per cluster
    pass_labels = set(df_pass["_fiber_label"].tolist()) if len(df_pass) else set()
    cluster_total: dict[int, int] = {}
    cluster_pass: dict[int, int] = {}
    for lbl, cc in label_to_cc.items():
        if cc == 0:
            continue
        cluster_total[cc] = cluster_total.get(cc, 0) + 1
        if lbl in pass_labels:
            cluster_pass[cc] = cluster_pass.get(cc, 0) + 1

    # Always keep the largest cluster
    largest_cc = max(cluster_total, key=cluster_total.get) if cluster_total else -1
    largest_count = cluster_total.get(largest_cc, 0)

    # Identify bad clusters (too small OR low QC rate)
    bad_clusters = set()
    for cc, total in cluster_total.items():
        if cc == largest_cc:
            continue
        if total < largest_count * min_fraction:
            bad_clusters.add(cc)
            continue
        rate = cluster_pass.get(cc, 0) / total
        if rate < min_rate:
            bad_clusters.add(cc)

    # Collect all fiber labels in bad clusters
    bad_labels = set()
    for lbl, cc in label_to_cc.items():
        if cc in bad_clusters:
            bad_labels.add(lbl)
    return bad_labels


def build_fascicle_mask(
    outer_labels: np.ndarray,
    pixel_size: float,
    fiber_diam_um: float,
) -> np.ndarray:
    """Build a smooth fascicle mask from the current (already filtered) labels.

    Used for nerve area computation and visualization boundary.
    """
    from scipy.ndimage import binary_fill_holes, gaussian_filter, zoom
    from skimage import morphology
    from skimage.transform import downscale_local_mean

    fiber_diam_px = fiber_diam_um / pixel_size
    h, w = outer_labels.shape

    scale = max(1, int(round(fiber_diam_px / 4)))
    sh, sw = h // scale, w // scale

    fiber_full = (outer_labels > 0).astype(np.float32)
    fiber_small = downscale_local_mean(fiber_full, (scale, scale))[:sh, :sw] > 0

    close_r = max(2, int(round(fiber_diam_px * 1.0 / scale)))
    closed = binary_fill_holes(morphology.closing(fiber_small, morphology.disk(close_r)))

    smooth_sigma = max(2, fiber_diam_px * 0.5 / scale)
    smooth = gaussian_filter(closed.astype(np.float32), sigma=smooth_sigma)
    return zoom(smooth, (h / sh, w / sw), order=1) > 0.40


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
