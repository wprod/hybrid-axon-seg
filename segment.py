#!/usr/bin/env python3
"""
segment.py — Nerve morphometry pipeline (entry point + orchestration).

Pipeline
--------
1. U-Net inference         → outer fiber label map   (cached *_cellpose_outer.npy)
                           → axon label map           (cached *_axon_inner.npy)
2. process_fibers          → per-fiber measurements
3. apply_qc                → pass / reject split
4. Visualizations          → overlay, numbered, g-ratio map, dashboard
"""

import contextlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, measure

import config
from morphometrics import compute_aggregate, process_fibers
from qc import apply_qc
from utils import build_fascicle_mask, clean_stem, find_low_qc_cluster_labels, find_satellite_labels
from visualization import (
    make_dashboard,
    make_gratio_map,
    make_numbered,
    make_overlay,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def build_axon_assignments(
    outer_labels: np.ndarray,
    inner_labels_raw: np.ndarray,
) -> tuple[dict, dict[int, float], np.ndarray]:
    """Build axon_assignments dict from cached label arrays, applying shrink + outside detection.

    Multi-core fibers (one outer label with N≥2 disconnected axon blobs) are split
    via Voronoi partition: each axon blob inherits the fiber pixels closest to it.
    The returned outer_labels reflects those splits (virtual labels appended after max).

    Returns
    -------
    axon_assignments   : fiber_label → (r0, c0, crop_bool)  — shrunk axon masks
    axon_outside_frac  : fiber_label → fraction of axon pixels outside fiber
    outer_labels       : updated label array (may contain new virtual labels for splits)
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.ndimage import label as nd_label

    outer_labels = outer_labels.copy()
    fiber_bboxes = {p.label: p.bbox for p in measure.regionprops(outer_labels)}
    axon_assignments: dict = {}
    axon_outside_frac: dict[int, float] = {}
    next_label = int(outer_labels.max()) + 1

    for lbl in np.unique(inner_labels_raw):
        if lbl == 0:
            continue
        bbox = fiber_bboxes.get(int(lbl))
        if bbox is None:
            continue
        minr, minc, maxr, maxc = bbox
        axon_crop = inner_labels_raw[minr:maxr, minc:maxc] == lbl
        fiber_crop = outer_labels[minr:maxr, minc:maxc] == lbl

        if axon_crop.sum() < config.MIN_AXON_SIZE:
            continue

        # Detect connected components in the axon mask
        labeled_blobs, n_blobs = nd_label(axon_crop)
        components = [
            labeled_blobs == c
            for c in range(1, n_blobs + 1)
            if (labeled_blobs == c).sum() >= config.MIN_AXON_SIZE
        ]

        if not components:
            continue

        if len(components) == 1:
            # ── Normal single-axon path ──────────────────────────────────
            crop = components[0]
            outside_px = int((crop & ~fiber_crop).sum())
            axon_outside_frac[int(lbl)] = outside_px / int(crop.sum())
            shrink_px = int(getattr(config, "AXON_SHRINK_PX", 0))
            if shrink_px > 0:
                dist = distance_transform_edt(crop)
                shrunk = dist > shrink_px
                if shrunk.any():
                    crop = shrunk
            axon_assignments[int(lbl)] = (minr, minc, crop)

        else:
            # ── Multi-core: Voronoi partition of fiber pixels ────────────
            # Each fiber pixel goes to the nearest axon blob (by Euclidean distance)
            dists = np.stack([distance_transform_edt(~c) for c in components])
            nearest = np.argmin(dists, axis=0)  # 0-indexed

            new_labels = [int(lbl)] + [next_label + i for i in range(len(components) - 1)]
            next_label += len(components) - 1

            # Repartition outer_labels in this crop
            crop_view = outer_labels[minr:maxr, minc:maxc]
            for i, new_lbl in enumerate(new_labels):
                crop_view[fiber_crop & (nearest == i)] = new_lbl

            # Register each axon with its virtual fiber label
            for comp, new_lbl in zip(components, new_labels):
                outside_px = int((comp & ~fiber_crop).sum())
                axon_outside_frac[new_lbl] = outside_px / max(int(comp.sum()), 1)
                shrink_px = int(getattr(config, "AXON_SHRINK_PX", 0))
                crop = comp
                if shrink_px > 0:
                    dist = distance_transform_edt(crop)
                    shrunk = dist > shrink_px
                    if shrunk.any():
                        crop = shrunk
                axon_assignments[new_lbl] = (minr, minc, crop)

    return axon_assignments, axon_outside_frac, outer_labels


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


_CLINICAL_COLS = [
    "axon_diam",
    "fiber_diam",
    "gratio",
    "myelin_thickness",
    "axon_area",
    "fiber_area",
    "x0",
    "y0",
]


def finalize_image(
    img: np.ndarray,
    outer_labels: np.ndarray,
    axon_assignments: dict,
    multicore_labels: set,
    stem: str,
    out_dir: Path,
    *,
    group: str = "",
    timepoint: str = "",
    has_fascicle: bool = False,
    fascicle_mask: np.ndarray | None = None,
    excl_mask: np.ndarray | None = None,
    qc_overrides: set[int] | None = None,
    axon_outside_frac: dict[int, float] | None = None,
) -> tuple[str, int, dict]:
    """Shared pipeline tail: morphometrics → QC → aggregate → CSV → visualizations.

    Called by process_image() (CLI batch) and app.py (web recompute).
    """
    # ── Morphometrics + QC ───────────────────────────────────────────────
    print("  Computing morphometrics…")
    inner_labels, pairs, df_all, index_image = process_fibers(
        outer_labels,
        axon_assignments,
        config.PIXEL_SIZE,
    )
    n_outer = int(outer_labels.max())
    n_matched = len(pairs)
    print(f"       → {len(df_all)} axons measured")

    # Inject axon-outside-fiber fraction so QC can filter on it
    if axon_outside_frac:
        df_all["axon_outside_frac"] = df_all["_fiber_label"].map(axon_outside_frac).fillna(0.0)

    df_pass, df_rej = apply_qc(df_all)
    # Re-admit manually accepted fibers
    if qc_overrides:
        override_mask = df_rej["_fiber_label"].isin(qc_overrides)
        if override_mask.any():
            df_pass = pd.concat([df_pass, df_rej[override_mask]], ignore_index=True)
            df_rej = df_rej[~override_mask].copy()
    n_overridden = sum(1 for lbl in (qc_overrides or []) if lbl in df_all["_fiber_label"].values)
    override_note = f" ({n_overridden} manually accepted)" if n_overridden else ""
    print(f"       → QC: {len(df_pass)} pass / {len(df_rej)} reject{override_note}")

    # Remove low-QC clusters — skipped when fascicle mask constrains U-Net
    if not has_fascicle:
        bad_cluster_labels = find_low_qc_cluster_labels(
            outer_labels, df_pass, df_rej, config.PIXEL_SIZE, config.FIBER_DIAM_UM
        )
        if bad_cluster_labels:
            outer_labels = _remove_labels(outer_labels, bad_cluster_labels)
            inner_labels = _remove_labels(inner_labels, bad_cluster_labels)

            def keep_fibers(df):
                return df[~df["_fiber_label"].isin(bad_cluster_labels)]

            df_pass = keep_fibers(df_pass)
            df_rej = keep_fibers(df_rej)
            print(f"       → removed {len(bad_cluster_labels)} fibers in low-QC clusters")

    # ── Fascicle mask ────────────────────────────────────────────────────
    if fascicle_mask is None:
        fascicle_mask = build_fascicle_mask(outer_labels, config.PIXEL_SIZE, config.FIBER_DIAM_UM)
    np.save(str(out_dir / f"{stem}_fascicle_mask.npy"), fascicle_mask)

    # ── Aggregate stats ──────────────────────────────────────────────────
    agg = compute_aggregate(
        outer_labels,
        df_pass,
        fascicle_mask,
        config.PIXEL_SIZE,
        group=group,
        timepoint=timepoint,
        excl_mask=excl_mask,
    )

    # ── Save data ────────────────────────────────────────────────────────
    pub_cols = [c for c in _CLINICAL_COLS if c in df_pass.columns]
    df_pass[pub_cols].to_csv(out_dir / f"{stem}_morphometrics.csv", index=False)
    with contextlib.suppress(ImportError):
        df_pass[pub_cols].to_excel(out_dir / f"{stem}_morphometrics.xlsx", index=False)
    pd.DataFrame([agg]).to_csv(out_dir / f"{stem}_aggregate.csv", index=False)

    # ── Visualizations ───────────────────────────────────────────────────
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

    if getattr(config, "GRATIO_MAP", False):
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

    return stem, len(df_pass), agg


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

    # ── Step 1: U-Net — outer fibers + axons ────────────────────────────
    cache_outer = out_dir / f"{stem}_cellpose_outer.npy"
    cache_axon = out_dir / f"{stem}_axon_inner.npy"
    cache_multicore = out_dir / f"{stem}_multicore_labels.npy"

    fascicle_pre = out_dir / f"{stem}_fascicle_mask_edited.npy"
    has_fascicle = fascicle_pre.exists()

    if cache_outer.exists():
        print("  [1/2] U-Net (fibers) — loading from cache…")
        outer_labels = np.load(str(cache_outer))
    else:
        raise FileNotFoundError(
            f"U-Net results not found for '{stem}'. Run U-Net inference first to generate "
            f"{cache_outer.name} and {cache_axon.name}."
        )

    # Erode each fiber mask: remove pixels within OUTER_ERODE_PX of any border
    if config.OUTER_ERODE_PX > 0:
        from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

        nz_max = maximum_filter(outer_labels, size=3)
        nz_min = minimum_filter(outer_labels, size=3)
        border = (outer_labels != 0) & (nz_max != nz_min)
        dist = distance_transform_edt(~border)
        outer_labels = (outer_labels * (dist > config.OUTER_ERODE_PX)).astype(outer_labels.dtype)

    # Remove satellite fibers — skipped when fascicle mask constrains U-Net
    if not has_fascicle:
        satellites = find_satellite_labels(outer_labels, config.PIXEL_SIZE, config.FIBER_DIAM_UM)
        outer_labels = _remove_labels(outer_labels, satellites)

    n_outer = int(outer_labels.max())
    print(f"       → {n_outer} fibers")

    # ── Step 2: Load axon labels ──────────────────────────────────────────
    if cache_axon.exists():
        print("  [2/2] Axon labels — loading from cache…")
        inner_labels_raw = np.load(str(cache_axon))
    else:
        print("  [2/2] No axon cache found — using empty axon map")
        inner_labels_raw = np.zeros(outer_labels.shape, dtype=outer_labels.dtype)
        np.save(str(cache_axon), inner_labels_raw)

    if not cache_multicore.exists():
        np.save(str(cache_multicore), np.array([], dtype=np.int32))

    multicore_labels: set = set()

    # Reconstruct axon_assignments (with shrink + outside detection)
    axon_assignments, axon_outside_frac, outer_labels = build_axon_assignments(outer_labels, inner_labels_raw)

    # ── Steps 3–5: shared pipeline (morphometrics → QC → viz) ──────────
    fascicle_mask = None
    if has_fascicle:
        fm = np.load(str(fascicle_pre))
        if fm.shape == outer_labels.shape:
            fascicle_mask = fm

    excl_path = out_dir / f"{stem}_exclusion_mask.npy"
    excl_mask = np.load(str(excl_path)) if excl_path.exists() else None
    if excl_mask is not None and excl_mask.shape != outer_labels.shape:
        excl_mask = None

    stem_out, n, agg = finalize_image(
        img,
        outer_labels,
        axon_assignments,
        multicore_labels,
        stem,
        out_dir,
        group=group,
        timepoint=timepoint,
        has_fascicle=has_fascicle,
        fascicle_mask=fascicle_mask,
        excl_mask=excl_mask,
        axon_outside_frac=axon_outside_frac,
    )

    print(f"  ✓ Done — {out_dir}")
    return stem_out, n, agg


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

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(config.OUTPUT_DIR / "summary.csv", index=False)
        print(f"\n{'=' * 60}")
        print(f"Done — {len(results)}/{len(image_tuples)} images")
        print(summary.to_string(index=False))
        print(f"{'=' * 60}")

        print("  ✓ Use compare.py for cross-sample comparison dashboard")


if __name__ == "__main__":
    main()
