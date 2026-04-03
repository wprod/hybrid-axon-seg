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
from morphometrics import compute_aggregate, process_fibers
from preprocessing import build_axon_input
from qc import apply_qc
from utils import build_fascicle_mask, clean_stem, find_low_qc_cluster_labels, find_satellite_labels
from visualization import (
    make_comparison,
    make_dashboard,
    make_gratio_map,
    make_numbered,
    make_overlay,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _remove_labels(outer_labels: np.ndarray, labels_to_remove: set) -> np.ndarray:
    """Zero out the given fiber labels."""
    if not labels_to_remove:
        return outer_labels
    cleaned = outer_labels.copy()
    remove_mask = np.isin(outer_labels, list(labels_to_remove))
    cleaned[remove_mask] = 0
    return cleaned


# ── Single-image pipeline ────────────────────────────────────────────────────


def _parse_folder(name: str) -> tuple[str, str]:
    """'ALLO A 12w' → ('ALLO A', '12w').  Falls back to (name, '') if no match."""
    import re

    m = re.search(r"(\d+w)\s*$", name.strip())
    if m:
        timepoint = m.group(1)
        group = name.strip()[: m.start()].strip()
        return group, timepoint
    return name.strip(), ""


def process_image(img_path: Path, group: str = "", timepoint: str = "") -> tuple[str, int, dict]:
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

    fascicle_pre = out_dir / f"{stem}_fascicle_mask_edited.npy"
    has_fascicle = fascicle_pre.exists()

    # If fascicle mask is newer than Cellpose cache → invalidate both caches
    # so Cellpose re-runs restricted to the fascicle boundary
    if (
        has_fascicle
        and cache_outer.exists()
        and fascicle_pre.stat().st_mtime > cache_outer.stat().st_mtime
    ):
        print("       fascicle mask updated — clearing Cellpose & axon caches")
        cache_outer.unlink()
        for _stale in (
            out_dir / f"{stem}_axon_inner.npy",
            out_dir / f"{stem}_axon_input.png",
        ):
            if _stale.exists():
                _stale.unlink()

    if cache_outer.exists():
        print("  [1/2] Cellpose (fibers) — loading from cache…")
        outer_labels = np.load(str(cache_outer))
    else:
        print("  [1/2] Cellpose (fibers)…")
        img_for_cellpose = img
        if has_fascicle:
            fm = np.load(str(fascicle_pre))
            if fm.shape == img.shape[:2]:
                print("       restricting to manual fascicle mask…")
                img_for_cellpose = img.copy()
                bg = (
                    255
                    if img.dtype == np.uint8
                    else (np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 1.0)
                )
                if img_for_cellpose.ndim == 3:
                    img_for_cellpose[~fm] = bg
                else:
                    img_for_cellpose[~fm] = bg
            else:
                print(f"       ⚠ fascicle mask shape {fm.shape} ≠ image {img.shape[:2]}, ignored")
        outer_labels = run_cellpose_fibers(img_for_cellpose)
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

    # Remove satellite fibers — skipped when fascicle mask constrains Cellpose
    if not has_fascicle:
        satellites = find_satellite_labels(outer_labels, config.PIXEL_SIZE, config.CP_DIAM_UM)
        outer_labels = _remove_labels(outer_labels, satellites)

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

    # ── Step 2b: Hard myelin-gap enforcement ─────────────────────────────
    # Guarantee that no axon pixel is within the mandatory margin of its fiber
    # boundary.  Works on the final assembled inner_labels_raw using the exact
    # outer_labels that will be used for visualisation — so no mismatch is
    # possible.  Catches any stray pixels that slip through per-crop processing
    # (fill_holes edge cases, sub-pixel crop-boundary artefacts, etc.).
    if config.AXON_MIN_MYELIN_FRAC > 0 or config.AXON_MIN_MYELIN_PX > 0:
        from scipy.ndimage import distance_transform_edt as _edt

        inner_labels_raw = np.zeros(outer_labels.shape, dtype=outer_labels.dtype)
        for fiber_label, (r0, c0, crop) in axon_assignments.items():
            inner_labels_raw[r0 : r0 + crop.shape[0], c0 : c0 + crop.shape[1]][crop] = fiber_label
        # Precompute bboxes once — avoids repeated full-image scans inside the loop
        _bboxes = {p.label: (p.bbox, p.area) for p in measure.regionprops(outer_labels)}
        for lbl in np.unique(inner_labels_raw):
            if lbl == 0:
                continue
            info = _bboxes.get(int(lbl))
            if info is None:
                continue
            (minr, minc, maxr, maxc), area = info
            fiber_crop = outer_labels[minr:maxr, minc:maxc] == lbl
            fiber_radius_px = float(np.sqrt(area / np.pi))
            min_px = max(
                float(config.AXON_MIN_MYELIN_PX), config.AXON_MIN_MYELIN_FRAC * fiber_radius_px
            )
            dist_crop = _edt(fiber_crop)  # EDT on small crop, not full image
            axon_crop = inner_labels_raw[minr:maxr, minc:maxc] == lbl
            inner_labels_raw[minr:maxr, minc:maxc][axon_crop & (dist_crop <= min_px)] = 0
        # Rebuild axon_assignments from the cleaned inner_labels_raw
        fiber_bboxes = {p.label: p.bbox for p in measure.regionprops(outer_labels)}
        axon_assignments = {}
        for lbl in np.unique(inner_labels_raw):
            if lbl == 0:
                continue
            bbox = fiber_bboxes.get(int(lbl))
            if bbox is None:
                continue
            minr, minc, maxr, maxc = bbox
            crop = inner_labels_raw[minr:maxr, minc:maxc] == lbl
            if crop.sum() >= config.MIN_AXON_SIZE:
                axon_assignments[int(lbl)] = (minr, minc, crop)
        np.save(str(cache_axon), inner_labels_raw)

    # ── Step 3: Morphometrics + QC ────────────────────────────────────────
    print("  Computing morphometrics…")
    inner_labels, pairs, df_all, index_image = process_fibers(
        outer_labels,
        axon_assignments,
        config.PIXEL_SIZE,
    )
    n_matched = len(pairs)
    print(f"       → {len(df_all)} axons measured")

    df_pass, df_rej = apply_qc(df_all)
    print(f"       → QC: {len(df_pass)} pass / {len(df_rej)} reject")

    # Remove low-QC clusters — skipped when fascicle mask constrains Cellpose
    if not has_fascicle:
        bad_cluster_labels = find_low_qc_cluster_labels(
            outer_labels, df_pass, df_rej, config.PIXEL_SIZE, config.CP_DIAM_UM
        )
        if bad_cluster_labels:
            outer_labels = _remove_labels(outer_labels, bad_cluster_labels)
            inner_labels = _remove_labels(inner_labels, bad_cluster_labels)

            def keep_fibers(df):
                return df[~df["_fiber_label"].isin(bad_cluster_labels)]

            df_pass = keep_fibers(df_pass)
            df_rej = keep_fibers(df_rej)
            print(f"       → removed {len(bad_cluster_labels)} fibers in low-QC clusters")

    # Fascicle mask: use manual if available, else auto-compute from labels
    if has_fascicle:
        fascicle_mask = np.load(str(fascicle_pre))
        if fascicle_mask.shape != outer_labels.shape:
            fascicle_mask = build_fascicle_mask(outer_labels, config.PIXEL_SIZE, config.CP_DIAM_UM)
    else:
        fascicle_mask = build_fascicle_mask(outer_labels, config.PIXEL_SIZE, config.CP_DIAM_UM)
    np.save(str(out_dir / f"{stem}_fascicle_mask.npy"), fascicle_mask)

    # Load exclusion zones (artefacts / tears drawn in the web UI)
    excl_path = out_dir / f"{stem}_exclusion_mask.npy"
    excl_mask = np.load(str(excl_path)) if excl_path.exists() else None
    if excl_mask is not None and excl_mask.shape != fascicle_mask.shape:
        excl_mask = None

    # Aggregate stats — nratio uses ALL fibres, avf/mvf/density use QC-pass only
    agg = compute_aggregate(
        outer_labels,
        df_pass,
        fascicle_mask,
        config.PIXEL_SIZE,
        group=group,
        timepoint=timepoint,
        excl_mask=excl_mask,
    )

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
    overlay = make_overlay(
        img,
        outer_labels,
        inner_labels,
        df_pass,
        df_rej,
        multicore_labels,
        fascicle_mask=fascicle_mask,
    )
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

    # Build list of (image_path, group, timepoint)
    if len(sys.argv) > 1:
        image_tuples = [(Path(p), "", "") for p in sys.argv[1:]]
    else:
        image_tuples = []
        _EXTS = {".tif", ".tiff", ".png"}
        for child in sorted(config.INPUT_DIR.iterdir()):
            if child.is_dir():
                group, timepoint = _parse_folder(child.name)
                for p in sorted(child.glob("*")):
                    if p.suffix.lower() in _EXTS:
                        image_tuples.append((p, group, timepoint))
            elif child.suffix.lower() in _EXTS:
                image_tuples.append((child, "", ""))

    if not image_tuples:
        sys.exit(f"No images found in {config.INPUT_DIR}")

    # Keep only images that have a manual fascicle mask
    with_fascicle = [
        (p, g, t)
        for p, g, t in image_tuples
        if (
            config.OUTPUT_DIR / clean_stem(p) / f"{clean_stem(p)}_fascicle_mask_edited.npy"
        ).exists()
    ]
    skipped = len(image_tuples) - len(with_fascicle)
    if skipped:
        print(f"  ↷ {skipped} image(s) without fascicle mask — skipped")
    if not with_fascicle:
        sys.exit("No images with a fascicle mask found. Draw fascicle boundaries in the app first.")
    image_tuples = with_fascicle

    print(f"Processing {len(image_tuples)} image(s) with fascicle mask\n")
    results = []
    for p, group, timepoint in image_tuples:
        stem = clean_stem(p)
        agg_path = config.OUTPUT_DIR / stem / f"{stem}_aggregate.csv"
        fascicle_path = config.OUTPUT_DIR / stem / f"{stem}_fascicle_mask_edited.npy"
        # Skip only if aggregate is newer than the fascicle mask
        if agg_path.exists() and agg_path.stat().st_mtime >= fascicle_path.stat().st_mtime:
            print(f"  ↷ {stem}  (already processed, skipping)")
            agg = pd.read_csv(agg_path).iloc[0].to_dict()
            results.append({"image": stem, **agg})
            continue
        try:
            stem, n, agg = process_image(p, group=group, timepoint=timepoint)
            results.append({"image": stem, "n_axons": n, **agg})
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")
            traceback.print_exc()

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(config.OUTPUT_DIR / "summary.csv", index=False)
        print(f"\n{'=' * 60}")
        print(f"Done — {len(results)}/{len(image_tuples)} images")
        print(summary.to_string(index=False))
        print(f"{'=' * 60}")

        print("  Generating comparison figure…")
        make_comparison(results, config.OUTPUT_DIR / "comparison.png")
        print(f"  ✓ comparison.png → {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
