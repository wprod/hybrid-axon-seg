#!/usr/bin/env python3
"""
segment.py — Nerve morphometry pipeline (entry point + orchestration).

Pipeline
--------
1. Cellpose cyto3      → outer fiber label map   (cached *_cellpose_outer.npy)
2. Normalized inversion → axon_input image        (cached *_axon_inner.npy)
   Global Otsu + centroid CC selection → axon blobs
3. process_fibers      → per-fiber measurements
4. apply_qc            → pass / reject split
5. Visualizations      → overlay, numbered, g-ratio map, dashboard
"""

import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, measure

import config
from detection import find_axons, run_cellpose_fibers
from morphometrics import process_fibers
from preprocessing import build_axon_input
from qc import apply_qc
from utils import clean_stem
from visualization import make_dashboard, make_gratio_map, make_numbered, make_overlay

warnings.filterwarnings("ignore", category=FutureWarning)


def _keep_main_cluster(outer_labels: np.ndarray, dilation_px: int) -> np.ndarray:
    """Zero out all fibers not belonging to the largest spatial cluster.

    Dilates the union of all fiber masks to merge nearby fibers into groups,
    then discards every fiber whose centroid falls outside the largest component.
    """
    from skimage import morphology as morph

    fiber_mask = outer_labels > 0
    merged = morph.binary_dilation(fiber_mask, morph.disk(dilation_px))
    cluster_map = measure.label(merged)

    props = measure.regionprops(cluster_map)
    if not props:
        return outer_labels
    main_label = max(props, key=lambda p: p.area).label
    keep = cluster_map == main_label

    # Zero out any fiber whose centroid is outside the main cluster
    cleaned = outer_labels.copy()
    for p in measure.regionprops(outer_labels):
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        if not keep[cy, cx]:
            cleaned[outer_labels == p.label] = 0
    return cleaned


# ── Single-image pipeline ────────────────────────────────────────────────────


def process_image(img_path: Path) -> tuple[str, int, dict]:
    stem = clean_stem(img_path)
    print(f"\n{'=' * 60}")
    print(f"  {img_path.name}  →  {stem}")
    print(f"{'=' * 60}")

    out_dir = config.OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Reading image…")
    img = io.imread(str(img_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # ── Step 1: Cellpose — outer fibers ──────────────────────────────────
    cache_outer = out_dir / f"{stem}_cellpose_outer.npy"
    old_cache = out_dir / f"{stem}_cellpose.npy"
    if not cache_outer.exists() and old_cache.exists():
        cache_outer = old_cache  # backwards-compat

    if cache_outer.exists():
        print("  [1/2] Cellpose (fibers) — loading from cache…")
        outer_labels = np.load(str(cache_outer))
    else:
        print("  [1/2] Cellpose (fibers)…")
        outer_labels = run_cellpose_fibers(img)
        np.save(str(out_dir / f"{stem}_cellpose_outer.npy"), outer_labels)

    # Erode each fiber mask: remove pixels within OUTER_ERODE_PX of any border
    # (internal between cells OR external). Vectorized via distance transform.
    if config.OUTER_ERODE_PX > 0:
        from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

        nz_max = maximum_filter(outer_labels, size=3)
        nz_min = minimum_filter(outer_labels, size=3)
        border = (outer_labels != 0) & (nz_max != nz_min)
        dist = distance_transform_edt(~border)
        outer_labels = (outer_labels * (dist > config.OUTER_ERODE_PX)).astype(outer_labels.dtype)

    if config.MAIN_CLUSTER_DILATION_PX > 0:
        outer_labels = _keep_main_cluster(outer_labels, config.MAIN_CLUSTER_DILATION_PX)

    n_outer = int(outer_labels.max())
    print(f"       → {n_outer} fibers")

    # ── Step 2: Axon detection ────────────────────────────────────────────
    cache_axon = out_dir / f"{stem}_axon_inner.npy"

    cache_multicore = out_dir / f"{stem}_multicore_labels.npy"

    if cache_axon.exists():
        print("  [2/2] Axon detection — loading from cache…")
        inner_labels_raw = np.load(str(cache_axon))
        fiber_bboxes = {p.label: p.bbox for p in measure.regionprops(outer_labels)}
        axon_assignments = {}
        for lbl in np.unique(inner_labels_raw):
            if lbl == 0:
                continue
            bbox = fiber_bboxes.get(int(lbl))
            if bbox is None:
                continue
            minr, minc, maxr, maxc = bbox
            axon_assignments[int(lbl)] = (minr, minc, inner_labels_raw[minr:maxr, minc:maxc] == lbl)
        multicore_labels = (
            set(np.load(str(cache_multicore)).tolist()) if cache_multicore.exists() else set()
        )
    else:
        print("  [2/2] Building axon input image…")
        axon_input = build_axon_input(img, outer_labels)
        io.imsave(str(out_dir / f"{stem}_axon_input.png"), axon_input, check_contrast=False)

        print("       detecting axons (global Otsu)…")
        axon_assignments, multicore_labels = find_axons(axon_input, outer_labels)

        inner_labels_raw = np.zeros(outer_labels.shape, dtype=outer_labels.dtype)
        for fiber_label, (r0, c0, crop) in axon_assignments.items():
            inner_labels_raw[r0 : r0 + crop.shape[0], c0 : c0 + crop.shape[1]][crop] = fiber_label
        np.save(str(cache_axon), inner_labels_raw)
        np.save(str(cache_multicore), np.array(sorted(multicore_labels), dtype=np.int32))

    # ── Step 3: Morphometrics + QC ────────────────────────────────────────
    print("  Computing morphometrics…")
    inner_labels, pairs, df_all, index_image, agg = process_fibers(
        outer_labels,
        axon_assignments,
        config.PIXEL_SIZE,
    )
    n_matched = len(pairs)
    print(f"       → {len(df_all)} axons measured")

    df_pass, df_rej = apply_qc(df_all)
    print(f"       → QC: {len(df_pass)} pass / {len(df_rej)} reject")

    # Recompute aggregate stats from QC-passed fibers only
    # Use fascicle area (morphologically closed fiber mask) as denominator
    from scipy.ndimage import binary_fill_holes as _fill_holes
    from scipy.ndimage import gaussian_filter as _gauss
    from skimage import morphology as _morph

    _closed = _fill_holes(_morph.binary_closing(outer_labels > 0, _morph.disk(30)))
    fascicle_mask = _gauss(_closed.astype(np.float32), sigma=40) > 0.35
    nerve_um2 = int(fascicle_mask.sum()) * config.PIXEL_SIZE**2
    if len(df_pass) and nerve_um2:
        avf = df_pass["axon_area"].sum() / nerve_um2
        mvf = (df_pass["fiber_area"].sum() - df_pass["axon_area"].sum()) / nerve_um2
        agg = {
            "avf": avf,
            "mvf": mvf,
            "nratio": avf + mvf,
            "gratio_aggr": df_pass["gratio"].mean(),
            "axon_density_mm2": len(df_pass) / (nerve_um2 * 1e-6),
            "nerve_area_mm2": nerve_um2 * 1e-6,
        }
    else:
        agg = {
            "avf": 0.0,
            "mvf": 0.0,
            "nratio": 0.0,
            "gratio_aggr": 0.0,
            "axon_density_mm2": 0.0,
            "nerve_area_mm2": 0.0,
        }

    # ── Step 4: Save data ─────────────────────────────────────────────────
    pub_cols = [c for c in df_pass.columns if not c.startswith("_")]
    df_pass[pub_cols].to_csv(out_dir / f"{stem}_morphometrics.csv", index=False)
    try:
        df_pass[pub_cols].to_excel(out_dir / f"{stem}_morphometrics.xlsx", index=False)
    except ImportError:
        print("       ⚠ openpyxl not installed — .xlsx skipped")
    pd.DataFrame([agg]).to_csv(out_dir / f"{stem}_aggregate.csv", index=False)

    # ── Step 5: Visualizations ────────────────────────────────────────────
    print("  Generating visualizations…")
    overlay = make_overlay(img, outer_labels, inner_labels, df_pass, df_rej, multicore_labels)
    io.imsave(str(out_dir / f"{stem}_overlay.png"), overlay, check_contrast=False)

    numbered = make_numbered(
        overlay, df_pass, n_outer, stem, nerve_area_mm2=agg.get("nerve_area_mm2", 0.0)
    )
    io.imsave(str(out_dir / f"{stem}_numbered.png"), numbered, check_contrast=False)

    make_gratio_map(img, df_pass, index_image, out_dir / f"{stem}_gratio_map.png")
    make_dashboard(
        df_pass,
        df_rej,
        agg,
        n_outer,
        n_matched,
        stem,
        out_dir / f"{stem}_dashboard.png",
        n_multicore=len(multicore_labels),
    )

    print(f"  ✓ Done — {out_dir}")
    return stem, len(df_pass), agg


# ── Batch entry point ────────────────────────────────────────────────────────


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        images = [Path(p) for p in sys.argv[1:]]
    else:
        images = sorted(
            p for p in config.INPUT_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png")
        )
    if not images:
        sys.exit(f"No images found in {config.INPUT_DIR}")

    print(f"Found {len(images)} image(s) in {config.INPUT_DIR}\n")
    results = []
    for p in images:
        try:
            stem, n, agg = process_image(p)
            results.append({"image": stem, "n_axons": n, **agg})
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")
            traceback.print_exc()

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(config.OUTPUT_DIR / "summary.csv", index=False)
        print(f"\n{'=' * 60}")
        print(f"Done — {len(results)}/{len(images)} images")
        print(summary.to_string(index=False))
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
