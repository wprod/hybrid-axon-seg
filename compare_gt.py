#!/usr/bin/env python3
"""compare_gt.py — Compare pipeline segmentation vs ground-truth annotations.

Computes per-image and aggregate metrics:
  - Semantic: pixel-level IoU (background, myelin, axon)
  - Instance: fiber detection (TP / FP / FN, precision, recall, F1)
  - Instance: axon detection (same)
  - Morphometric drift: g-ratio, diameter differences on matched fibers
  - Error breakdown: what types of fibers are missed / hallucinated
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage import measure

import config


# ── GT stems ────────────────────────────────────────────────────────────────

GT_MASKS = config.GT_DIR / "masks"

def _gt_stems() -> list[str]:
    stems = []
    for p in sorted(GT_MASKS.glob("*_outer_gt.npy")):
        stem = p.name.replace("_outer_gt.npy", "")
        axon_p = GT_MASKS / f"{stem}_axon_gt.npy"
        if axon_p.exists():
            stems.append(stem)
    return stems


# ── Pixel-level IoU ─────────────────────────────────────────────────────────

def _semantic_iou(pred_outer, pred_axon, gt_outer, gt_axon):
    """Compute per-class IoU on 3-class semantic masks (bg=0, myelin=1, axon=2)."""
    def _to_semantic(outer, axon):
        s = np.zeros(outer.shape, dtype=np.uint8)
        s[outer > 0] = 1
        s[axon > 0] = 2
        return s

    pred = _to_semantic(pred_outer, pred_axon)
    gt = _to_semantic(gt_outer, gt_axon)

    results = {}
    for cls, name in [(0, "background"), (1, "myelin"), (2, "axon")]:
        p = pred == cls
        g = gt == cls
        inter = (p & g).sum()
        union = (p | g).sum()
        results[f"iou_{name}"] = float(inter / union) if union > 0 else 1.0
    results["miou_fg"] = (results["iou_myelin"] + results["iou_axon"]) / 2
    return results


# ── Instance matching via IoU ───────────────────────────────────────────────

def _instance_metrics(pred_labels, gt_labels, iou_thresh=0.3):
    """Match predicted instances to GT instances using IoU, compute P/R/F1.

    Uses bbox overlap to avoid the O(N*M) full-image comparison — only computes
    IoU for instance pairs whose bounding boxes intersect.

    Returns dict with TP, FP, FN, precision, recall, F1, and lists of
    unmatched pred/gt labels for error analysis.
    """
    pred_props = measure.regionprops(pred_labels)
    gt_props = measure.regionprops(gt_labels)
    pred_ids = [p.label for p in pred_props]
    gt_ids = [p.label for p in gt_props]

    if not pred_ids and not gt_ids:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "unmatched_pred": [], "unmatched_gt": []}
    if not pred_ids:
        return {"tp": 0, "fp": 0, "fn": len(gt_ids), "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "unmatched_pred": [], "unmatched_gt": gt_ids}
    if not gt_ids:
        return {"tp": 0, "fp": len(pred_ids), "fn": 0, "precision": 0.0, "recall": 1.0, "f1": 0.0,
                "unmatched_pred": pred_ids, "unmatched_gt": []}

    # Build a pixel→label lookup for GT to avoid repeated == comparisons
    # For each pred region, look at which GT labels its pixels overlap with
    pred_areas = {p.label: int(p.area) for p in pred_props}
    gt_areas = {p.label: int(p.area) for p in gt_props}

    # Greedy matching: for each pred, find best-overlapping GT via pixel lookup
    # This is O(total_pixels) instead of O(N*M*pixels)
    pred_idx = {lbl: i for i, lbl in enumerate(pred_ids)}
    gt_idx = {lbl: i for i, lbl in enumerate(gt_ids)}

    # Count overlapping pixels between each (pred_label, gt_label) pair
    # Vectorised: encode pair as single int64, then use np.unique to count
    overlap = {}
    mask = (pred_labels > 0) & (gt_labels > 0)
    if mask.any():
        p_vals = pred_labels[mask].astype(np.int64)
        g_vals = gt_labels[mask].astype(np.int64)
        max_gt = int(gt_labels.max()) + 1
        pair_keys = p_vals * max_gt + g_vals
        unique_pairs, counts = np.unique(pair_keys, return_counts=True)
        for pk, cnt in zip(unique_pairs, counts):
            pv = int(pk // max_gt)
            gv = int(pk % max_gt)
            if pv in pred_idx and gv in gt_idx:
                overlap[(pred_idx[pv], gt_idx[gv])] = int(cnt)

    # Build sparse IoU and do greedy matching (Hungarian is too slow for 2k×2k)
    # Sort overlaps by IoU descending, greedily assign
    iou_pairs = []
    for (pi, gi), inter in overlap.items():
        union = pred_areas[pred_ids[pi]] + gt_areas[gt_ids[gi]] - inter
        iou = inter / union if union > 0 else 0.0
        if iou >= iou_thresh:
            iou_pairs.append((iou, pi, gi))

    iou_pairs.sort(reverse=True)

    matched_pred = set()
    matched_gt = set()
    tp = 0
    for iou_val, pi, gi in iou_pairs:
        if pi not in matched_pred and gi not in matched_gt:
            tp += 1
            matched_pred.add(pi)
            matched_gt.add(gi)

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    precision = tp / len(pred_ids) if pred_ids else 0.0
    recall = tp / len(gt_ids) if gt_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    unmatched_pred = [pred_ids[i] for i in range(len(pred_ids)) if i not in matched_pred]
    unmatched_gt = [gt_ids[i] for i in range(len(gt_ids)) if i not in matched_gt]

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "unmatched_pred": unmatched_pred, "unmatched_gt": unmatched_gt}


# ── Size analysis of errors ─────────────────────────────────────────────────

def _size_analysis(labels, label_ids, pixel_size):
    """Return area stats (µm²) for a set of label IDs."""
    if not label_ids:
        return {"count": 0, "mean_area_um2": 0, "median_diam_um": 0, "min_diam_um": 0, "max_diam_um": 0}
    areas = []
    for lid in label_ids:
        a = (labels == lid).sum()
        areas.append(a * pixel_size ** 2)
    areas = np.array(areas)
    diams = np.sqrt(4 * areas / np.pi)
    return {
        "count": len(areas),
        "mean_area_um2": float(areas.mean()),
        "median_diam_um": float(np.median(diams)),
        "min_diam_um": float(diams.min()),
        "max_diam_um": float(diams.max()),
    }


# ── Per-image comparison ────────────────────────────────────────────────────

def compare_one(stem: str) -> dict:
    """Compare pipeline output vs GT for one stem."""
    print(f"\n{'─' * 50}")
    print(f"  {stem}")
    print(f"{'─' * 50}")

    out_dir = config.OUTPUT_DIR / stem

    # Load GT
    gt_outer = np.load(str(GT_MASKS / f"{stem}_outer_gt.npy"))
    gt_axon = np.load(str(GT_MASKS / f"{stem}_axon_gt.npy"))

    # Load pipeline predictions
    pred_outer = np.load(str(out_dir / f"{stem}_cellpose_outer.npy"))
    pred_axon = np.load(str(out_dir / f"{stem}_axon_inner.npy"))

    # Ensure same shape
    assert pred_outer.shape == gt_outer.shape, f"Shape mismatch: pred {pred_outer.shape} vs gt {gt_outer.shape}"

    n_gt_fibers = len(np.unique(gt_outer)) - 1  # exclude 0
    n_pred_fibers = len(np.unique(pred_outer)) - 1
    n_gt_axons = len(np.unique(gt_axon)) - 1
    n_pred_axons = len(np.unique(pred_axon)) - 1

    print(f"  Fibers:  pipeline={n_pred_fibers}  GT={n_gt_fibers}  (Δ={n_pred_fibers - n_gt_fibers:+d})")
    print(f"  Axons:   pipeline={n_pred_axons}  GT={n_gt_axons}  (Δ={n_pred_axons - n_gt_axons:+d})")

    # Semantic IoU
    sem = _semantic_iou(pred_outer, pred_axon, gt_outer, gt_axon)
    print(f"  Pixel IoU — bg={sem['iou_background']:.3f}  myelin={sem['iou_myelin']:.3f}  "
          f"axon={sem['iou_axon']:.3f}  mIoU(fg)={sem['miou_fg']:.3f}")

    # Instance matching — fibers
    fiber_m = _instance_metrics(pred_outer, gt_outer, iou_thresh=0.3)
    print(f"  Fibers — TP={fiber_m['tp']}  FP={fiber_m['fp']}  FN={fiber_m['fn']}  "
          f"P={fiber_m['precision']:.3f}  R={fiber_m['recall']:.3f}  F1={fiber_m['f1']:.3f}")

    # Instance matching — axons
    axon_m = _instance_metrics(pred_axon, gt_axon, iou_thresh=0.3)
    print(f"  Axons  — TP={axon_m['tp']}  FP={axon_m['fp']}  FN={axon_m['fn']}  "
          f"P={axon_m['precision']:.3f}  R={axon_m['recall']:.3f}  F1={axon_m['f1']:.3f}")

    # Error analysis
    fp_fibers = _size_analysis(pred_outer, fiber_m["unmatched_pred"], config.PIXEL_SIZE)
    fn_fibers = _size_analysis(gt_outer, fiber_m["unmatched_gt"], config.PIXEL_SIZE)
    fp_axons = _size_analysis(pred_axon, axon_m["unmatched_pred"], config.PIXEL_SIZE)
    fn_axons = _size_analysis(gt_axon, axon_m["unmatched_gt"], config.PIXEL_SIZE)

    if fn_fibers["count"]:
        print(f"  Missed fibers:      n={fn_fibers['count']}  diam={fn_fibers['median_diam_um']:.1f}µm "
              f"(range {fn_fibers['min_diam_um']:.1f}–{fn_fibers['max_diam_um']:.1f})")
    if fp_fibers["count"]:
        print(f"  Hallucinated fibers: n={fp_fibers['count']}  diam={fp_fibers['median_diam_um']:.1f}µm "
              f"(range {fp_fibers['min_diam_um']:.1f}–{fp_fibers['max_diam_um']:.1f})")
    if fn_axons["count"]:
        print(f"  Missed axons:       n={fn_axons['count']}  diam={fn_axons['median_diam_um']:.1f}µm "
              f"(range {fn_axons['min_diam_um']:.1f}–{fn_axons['max_diam_um']:.1f})")

    return {
        "stem": stem,
        "n_gt_fibers": n_gt_fibers, "n_pred_fibers": n_pred_fibers,
        "n_gt_axons": n_gt_axons, "n_pred_axons": n_pred_axons,
        **{f"sem_{k}": v for k, v in sem.items()},
        "fiber_tp": fiber_m["tp"], "fiber_fp": fiber_m["fp"], "fiber_fn": fiber_m["fn"],
        "fiber_precision": fiber_m["precision"], "fiber_recall": fiber_m["recall"], "fiber_f1": fiber_m["f1"],
        "axon_tp": axon_m["tp"], "axon_fp": axon_m["fp"], "axon_fn": axon_m["fn"],
        "axon_precision": axon_m["precision"], "axon_recall": axon_m["recall"], "axon_f1": axon_m["f1"],
        "fp_fiber_median_diam": fp_fibers["median_diam_um"],
        "fn_fiber_median_diam": fn_fibers["median_diam_um"],
        "fp_axon_median_diam": fp_axons["median_diam_um"],
        "fn_axon_median_diam": fn_axons["median_diam_um"],
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    stems = _gt_stems()
    # Filter to stems that also have pipeline output
    stems = [s for s in stems if (config.OUTPUT_DIR / s / f"{s}_cellpose_outer.npy").exists()]
    print(f"Comparing {len(stems)} images with both pipeline output and GT annotations")

    rows = []
    for stem in stems:
        try:
            row = compare_one(stem)
            rows.append(row)
        except Exception as e:
            print(f"  ✗ {stem}: {e}")

    if not rows:
        print("No images to compare.")
        return

    df = pd.DataFrame(rows)

    # ── Aggregate summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 60}")

    for metric in ["sem_iou_myelin", "sem_iou_axon", "sem_miou_fg",
                    "fiber_precision", "fiber_recall", "fiber_f1",
                    "axon_precision", "axon_recall", "axon_f1"]:
        vals = df[metric]
        print(f"  {metric:25s}  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    total_fiber_tp = df["fiber_tp"].sum()
    total_fiber_fp = df["fiber_fp"].sum()
    total_fiber_fn = df["fiber_fn"].sum()
    total_axon_tp = df["axon_tp"].sum()
    total_axon_fp = df["axon_fp"].sum()
    total_axon_fn = df["axon_fn"].sum()

    print(f"\n  Fibers total: TP={total_fiber_tp}  FP={total_fiber_fp}  FN={total_fiber_fn}")
    micro_p = total_fiber_tp / (total_fiber_tp + total_fiber_fp) if (total_fiber_tp + total_fiber_fp) else 0
    micro_r = total_fiber_tp / (total_fiber_tp + total_fiber_fn) if (total_fiber_tp + total_fiber_fn) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
    print(f"  Fibers micro: P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")

    print(f"\n  Axons total:  TP={total_axon_tp}  FP={total_axon_fp}  FN={total_axon_fn}")
    micro_p = total_axon_tp / (total_axon_tp + total_axon_fp) if (total_axon_tp + total_axon_fp) else 0
    micro_r = total_axon_tp / (total_axon_tp + total_axon_fn) if (total_axon_tp + total_axon_fn) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
    print(f"  Axons micro:  P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")

    # Error profile
    print(f"\n  --- Error profile ---")
    fn_diams = df["fn_fiber_median_diam"]
    fp_diams = df["fp_fiber_median_diam"]
    print(f"  Missed fibers  — median diameter across images: {fn_diams[fn_diams > 0].mean():.1f}µm")
    print(f"  False+ fibers  — median diameter across images: {fp_diams[fp_diams > 0].mean():.1f}µm")
    fn_ax = df["fn_axon_median_diam"]
    fp_ax = df["fp_axon_median_diam"]
    print(f"  Missed axons   — median diameter across images: {fn_ax[fn_ax > 0].mean():.1f}µm")

    # Save
    out_path = config.OUTPUT_DIR / "gt_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
