#!/usr/bin/env python3
"""
segment.py — Cellpose + per-fiber normalized-inversion + Otsu morphometry pipeline

Pass 1 : Cellpose (cyto3) → outer fiber masks  (source of truth for count)
Pass 2 : Per-fiber crops:
           • erode mask + distance-based fade to strip endoneurium contamination
           • per-fiber percentile stretch → invert (axon=dark blob, myelin=bright ring)
           • global Otsu on full axon_input → axon/myelin separation
           • centroid-based CC selection + fill_holes

Overlay : green=axon  blue=myelin  red=no axon detected / QC fail
"""

import re
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import io, measure, morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import find_boundaries

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────── CONFIG ────────────────────────────────
INPUT_DIR   = Path("edited")
OUTPUT_DIR  = Path("output")
PIXEL_SIZE  = 0.09      # µm/px at source resolution

# Cellpose — pass 1 (outer fibers)
CP_MODEL    = "cyto3"
CP_DIAM_UM  = 8.0
CP_FLOW_THR = 0.4
CP_CELLPROB = 0.0

# Inversion / preprocessing
MASK_ERODE_PX = 4   # px stripped from fiber boundary before inversion
_FADE_PX      = 8   # px of soft fade inside the eroded boundary
MIN_AXON_SIZE = 40  # min axon area (px²)

# QC filters  (permissive — clinician adjusts)
MIN_GRATIO          = 0.215
MAX_GRATIO          = 0.9
MIN_AXON_DIAM_UM    = 0.5
MIN_SOLIDITY        = 0.4
MAX_CENTROID_OFFSET = 0.65
MAX_AXON_ECCEN      = 0.99
EXCLUDE_BORDER      = True
# ────────────────────────────────────────────────────────────────────────


# ──────────────────────────────── Helpers ───────────────────────────────

def clean_stem(path: Path) -> str:
    name = path.name
    name = re.sub(r"(?:\s*\(clean\))?\.tiff?", "", name)
    return name


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        out = np.stack([img] * 3, axis=-1)
    else:
        out = img[:, :, :3].copy()
    if out.dtype in (np.float32, np.float64):
        out = (out * 255).clip(0, 255).astype(np.uint8) if out.max() <= 1.0 \
              else out.clip(0, 255).astype(np.uint8)
    elif out.dtype == np.uint16:
        out = (out / 256).astype(np.uint8)
    else:
        out = out.astype(np.uint8)
    return out


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return to_rgb_uint8(img).mean(axis=2).astype(np.uint8)
    elif img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        maxv = img.max()
        return ((img / maxv) * 255).astype(np.uint8) if maxv > 0 else img.astype(np.uint8)
    return img.astype(np.uint8)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ──────────────── Normalized-inversion crop ────────────────────────────

def _invert_crop(crop_gray: np.ndarray, fiber_mask: np.ndarray) -> np.ndarray:
    """Per-fiber contrast-stretch + invert with smooth inward fade.

    - Pixels within MASK_ERODE_PX of the boundary → white (strips endoneurium).
    - Normalization uses only the clean eroded interior (percentile 5–95).
    - Fade goes from white at the eroded edge to fully inverted over _FADE_PX px.
      darkest pixels (myelin)  → 255 (bright ring)
      brightest pixels (axon)  → 0   (dark blob)
    """
    dist = ndimage.distance_transform_edt(fiber_mask).astype(np.float32)
    inner = dist > MASK_ERODE_PX
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
    fade = np.clip(inner_dist / max(_FADE_PX, 1), 0.0, 1.0)
    blended = (255.0 + (inverted - 255.0) * fade).clip(0, 255).astype(np.uint8)

    result = np.full(crop_gray.shape, 255, dtype=np.uint8)
    result[inner] = blended[inner]
    return result


# ──────────────── Cellpose — pass 1 ────────────────────────────────────

def run_cellpose_fibers(img: np.ndarray) -> np.ndarray:
    import torch
    from cellpose import models

    gpu = torch.backends.mps.is_available() or torch.cuda.is_available()
    device = (
        "MPS" if torch.backends.mps.is_available()
        else "CUDA" if torch.cuda.is_available()
        else "CPU"
    )
    print(f"       device: {device}")
    model = models.CellposeModel(pretrained_model=CP_MODEL, gpu=gpu)
    masks, _, _ = model.eval(
        img,
        diameter=CP_DIAM_UM / PIXEL_SIZE,
        flow_threshold=CP_FLOW_THR,
        cellprob_threshold=CP_CELLPROB,
        min_size=MIN_AXON_SIZE,
    )
    return masks


# ──────────────── Axon detection ───────────────────────────────────────

def build_axon_input(img: np.ndarray, outer_labels: np.ndarray) -> np.ndarray:
    """Build the full-image normalized-inverted map (also saved as debug PNG)."""
    gray = _to_uint8_gray(img)
    result = np.full_like(gray, 255, dtype=np.uint8)
    for p in measure.regionprops(outer_labels):
        minr, minc, maxr, maxc = p.bbox
        crop = _invert_crop(gray[minr:maxr, minc:maxc], p.image)
        result[minr:maxr, minc:maxc][p.image] = crop[p.image]
    return result


def find_axons(axon_input: np.ndarray, outer_labels: np.ndarray) -> dict:
    """Global Otsu on axon_input → per-fiber centroid-based blob selection.

    axon_input already has per-fiber normalization applied:
      axon pixels  ≈ 0   (dark)
      myelin       ≈ 255 (bright)
      background   = 255 (white)
    One global Otsu on all fiber pixels cleanly separates them.

    Returns dict: fiber_label → (minr, minc, crop_bool)
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
        if best.sum() < MIN_AXON_SIZE:
            continue

        result[p.label] = (minr, minc, best)

    return result


# ──────────────── Morphometrics ────────────────────────────────────────

def process_fibers(
    outer_labels: np.ndarray,
    axon_assignments: dict,   # fiber_label → (minr, minc, crop_bool)
    pixel_size: float,
) -> tuple[np.ndarray, dict, pd.DataFrame, np.ndarray, dict]:

    fiber_rprops = {p.label: p for p in measure.regionprops(outer_labels)}
    img_h, img_w = outer_labels.shape

    inner_labels = np.zeros_like(outer_labels)
    index_image  = np.zeros(outer_labels.shape, dtype=np.int32)
    pairs = {}
    rows  = []

    for axon_id, (fiber_label, (r0, c0, crop_axon)) in enumerate(axon_assignments.items(), 1):
        fp = fiber_rprops.get(fiber_label)
        if fp is None:
            continue

        area_outer = int(fp.area)
        area_inner = int(crop_axon.sum())
        if area_inner == 0:
            continue

        d_outer = np.sqrt(4 * area_outer / np.pi) * pixel_size
        d_inner = np.sqrt(4 * area_inner / np.pi) * pixel_size
        gratio  = d_inner / d_outer
        myelin_thickness = (d_outer - d_inner) / 2

        axon_coords = np.argwhere(crop_axon)
        y0 = float(axon_coords[:, 0].mean()) + r0
        x0 = float(axon_coords[:, 1].mean()) + c0

        fy, fx = fp.centroid
        offset_px = np.sqrt((y0 - fy) ** 2 + (x0 - fx) ** 2)
        fiber_radius_px = np.sqrt(area_outer / np.pi)
        centroid_offset = float(offset_px / fiber_radius_px) if fiber_radius_px > 0 else 1.0

        axon_rprops = measure.regionprops(measure.label(crop_axon.astype(np.uint8)))
        best = max(axon_rprops, key=lambda r: r.area) if axon_rprops else None
        solidity     = best.solidity     if best else 1.0
        eccentricity = best.eccentricity if best else 0.0

        minr, minc, maxr, maxc = fp.bbox
        border = (minr == 0 or minc == 0 or maxr == img_h or maxc == img_w)

        inner_labels[r0:r0 + crop_axon.shape[0], c0:c0 + crop_axon.shape[1]][crop_axon] = fiber_label
        index_image [r0:r0 + crop_axon.shape[0], c0:c0 + crop_axon.shape[1]][crop_axon] = axon_id
        pairs[fiber_label] = fiber_label
        rows.append({
            "axon_diam":        d_inner,
            "fiber_diam":       d_outer,
            "gratio":           gratio,
            "myelin_thickness": myelin_thickness,
            "axon_area":        area_inner * pixel_size ** 2,
            "fiber_area":       area_outer * pixel_size ** 2,
            "solidity":         solidity,
            "eccentricity":     eccentricity,
            "centroid_offset":  centroid_offset,
            "x0": x0, "y0": y0,
            "image_border_touching": border,
            "_label":       axon_id,
            "_fiber_label": fiber_label,
        })

    df = pd.DataFrame(rows)
    n_ok   = len(rows)
    n_fail = int(outer_labels.max()) - n_ok
    print(f"       -> {n_ok} fibers with axon, {n_fail} without (shown in red)")

    h, w = outer_labels.shape
    total_um2  = h * w * pixel_size ** 2
    total_axon = df["axon_area"].sum() if len(df) else 0.0
    total_fib  = df["fiber_area"].sum() if len(df) else 0.0
    agg = {
        "avf":              total_axon / total_um2 if total_um2 else 0,
        "mvf":              (total_fib - total_axon) / total_um2 if total_um2 else 0,
        "gratio_aggr":      df["gratio"].mean() if len(df) else 0,
        "axon_density_mm2": len(df) / (total_um2 * 1e-6) if total_um2 else 0,
    }

    return inner_labels, pairs, df, index_image, agg


# ──────────────── QC ───────────────────────────────────────────────────

_QC_REASON_LABEL = {
    "gratio":    "G",
    "axon_diam": "Ø",
    "solidity":  "sol",
    "eccen":     "ecc",
    "offset":    "off",
    "border":    "brd",
}

def apply_qc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filters = {}
    if "gratio" in df.columns:
        filters["gratio"]   = ~(df["gratio"].notna() & df["gratio"].between(MIN_GRATIO, MAX_GRATIO))
    filters["axon_diam"]    = df["axon_diam"] < MIN_AXON_DIAM_UM
    if "solidity" in df.columns:
        filters["solidity"] = df["solidity"] < MIN_SOLIDITY
    if "eccentricity" in df.columns:
        filters["eccen"]    = df["eccentricity"] > MAX_AXON_ECCEN
    if "centroid_offset" in df.columns:
        filters["offset"]   = df["centroid_offset"] > MAX_CENTROID_OFFSET
    if EXCLUDE_BORDER and "image_border_touching" in df.columns:
        filters["border"]   = df["image_border_touching"].fillna(False).astype(bool)

    reject = pd.Series(False, index=df.index)
    reason = pd.Series("",    index=df.index, dtype=str)
    for name, mask in filters.items():
        n = int(mask.sum())
        if n:
            print(f"         QC [{name}]: {n} rejected")
        reason.loc[(reason == "") & mask] = _QC_REASON_LABEL.get(name, name[:3])
        reject |= mask

    df_rej = df[reject].copy()
    df_rej["reject_reason"] = reason[reject].values
    return df[~reject].copy(), df_rej


# ──────────────── Visualizations ───────────────────────────────────────

def _make_overlay(img, outer_labels, inner_labels, df_pass, df_rej):
    """
    RED    = no axon detected
    ORANGE = axon detected, rejected by QC  (reason code printed on fiber)
    BLUE   = myelin ring — QC passed
    GREEN  = axon — QC passed
    """
    rgb = to_rgb_uint8(img)
    overlay = (rgb.astype(np.float32) * 0.4).astype(np.uint8)

    pass_fibers = set(df_pass["_fiber_label"].tolist()) if len(df_pass) else set()
    rej_fibers  = set(df_rej["_fiber_label"].tolist())  if len(df_rej)  else set()

    def _blend(mask, color, alpha=0.6):
        c = np.array(color, dtype=np.float32)
        overlay[mask] = np.clip(
            overlay[mask].astype(np.float32) * (1 - alpha) + c * alpha, 0, 255,
        ).astype(np.uint8)

    no_axon    = (outer_labels > 0) & (inner_labels == 0)
    rej_axon   = np.isin(inner_labels, list(rej_fibers))  & (inner_labels > 0)
    rej_myelin = np.isin(outer_labels, list(rej_fibers))  & ~rej_axon & (inner_labels == 0)
    pass_axon  = np.isin(inner_labels, list(pass_fibers)) & (inner_labels > 0)
    pass_myel  = np.isin(outer_labels, list(pass_fibers)) & ~pass_axon

    _blend(no_axon,    [220,  50,  50])
    _blend(rej_myelin, [200, 120,  30], alpha=0.35)
    _blend(rej_axon,   [255, 140,   0])
    _blend(pass_myel,  [ 50,  50, 240])
    _blend(pass_axon,  [  0, 210,  60])

    outer_c = morphology.binary_dilation(find_boundaries(outer_labels, mode="thick"), morphology.disk(1))
    inner_c = morphology.binary_dilation(find_boundaries(inner_labels, mode="thick"), morphology.disk(1))
    overlay[outer_c] = [70, 70, 220]
    overlay[inner_c] = [0, 240, 80]

    # ── PIL layer: rejection codes + legend ──────────────────────────
    pil  = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    font_code = _load_font(11)
    font_leg  = _load_font(13)
    font_title = _load_font(14)

    # Rejection reason label on each orange fiber
    if len(df_rej) and "reject_reason" in df_rej.columns:
        for _, row in df_rej.iterrows():
            reason = str(row.get("reject_reason", ""))
            if not reason:
                continue
            x, y = int(row["x0"]), int(row["y0"])
            draw.text((x - 5, y - 6), reason, fill=(0, 0, 0),       font=font_code)  # shadow
            draw.text((x - 6, y - 7), reason, fill=(255, 220, 60),   font=font_code)

    # Legend (bottom-left, semi-transparent dark box)
    W, H    = pil.width, pil.height
    lx, ly  = 18, H - 205
    lw, lh  = 278, 185
    arr = np.array(pil)
    arr[ly:ly+lh, lx:lx+lw] = np.clip(
        arr[ly:ly+lh, lx:lx+lw].astype(float) * 0.2 + np.array([18, 18, 18]) * 0.8,
        0, 255,
    ).astype(np.uint8)
    pil  = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)

    # border around legend
    draw.rectangle([lx, ly, lx+lw, ly+lh], outline=(90, 90, 90), width=1)

    legend_entries = [
        ([  0, 210,  60], "Axon — QC passed"),
        ([ 50,  50, 240], "Myelin — QC passed"),
        ([255, 140,   0], "Detected — QC rejected"),
        ([220,  50,  50], "No axon detected"),
    ]
    codes_line = [
        "Rejection codes:",
        "G=g-ratio   Ø=diameter   sol=solidity",
        "ecc=eccen.  off=offset   brd=border",
    ]

    y_cur = ly + 10
    for color, label in legend_entries:
        draw.rectangle([lx+10, y_cur+2, lx+23, y_cur+14], fill=tuple(color))
        draw.text((lx+32, y_cur), label, fill=(230, 230, 230), font=font_leg)
        y_cur += 23
    draw.line([lx+8, y_cur+2, lx+lw-8, y_cur+2], fill=(70, 70, 70), width=1)
    y_cur += 10
    for line in codes_line:
        clr = (200, 200, 200) if line == "Rejection codes:" else (155, 155, 155)
        draw.text((lx+10, y_cur), line, fill=clr, font=font_code)
        y_cur += 16

    return np.array(pil)


def _make_numbered(overlay, df, n_outer, stem):
    pil  = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    font = _load_font(14)
    font_banner = _load_font(20)
    for num, (_, row) in enumerate(df.iterrows(), 1):
        draw.text((int(row["x0"]) - 6, int(row["y0"]) - 8), str(num),
                  fill=(255, 255, 0), font=font)
    n       = len(df)
    mean_ax = df["axon_diam"].mean()  if n else 0
    mean_fi = df["fiber_diam"].mean() if n else 0
    mean_g  = df["gratio"].mean()     if n else 0
    banner_h = 40
    canvas = Image.new("RGB", (pil.width, pil.height + banner_h), (30, 30, 30))
    canvas.paste(pil, (0, banner_h))
    d = ImageDraw.Draw(canvas)
    d.text((10, 10),
           f"  n={n}/{n_outer}   axon={mean_ax:.2f}\u00b5m   "
           f"fiber={mean_fi:.2f}\u00b5m   G={mean_g:.3f}  \u2014  {stem}",
           fill=(255, 255, 255), font=font_banner)
    return np.array(canvas)


def _make_gratio_map(img, df, index_image, out_path):
    rgb = to_rgb_uint8(img)
    bg  = (rgb.astype(np.float32) * 0.3).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(bg)
    if "gratio" in df.columns and len(df) > 0:
        norm = Normalize(vmin=MIN_GRATIO, vmax=MAX_GRATIO)
        cmap = matplotlib.colormaps["RdYlGn"]
        rgba = np.zeros((*index_image.shape, 4), dtype=np.float32)
        for _, row in df.iterrows():
            gr, lbl = row.get("gratio", np.nan), int(row["_label"])
            if np.isnan(gr) or lbl == 0:
                continue
            rgba[index_image == lbl] = cmap(norm(np.clip(gr, MIN_GRATIO, MAX_GRATIO)))
        ax.imshow(rgba)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="G-ratio", shrink=0.6, pad=0.02)
    ax.axis("off")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_dashboard(df, df_rej, agg, n_outer, n_matched, stem, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Morphometry Dashboard \u2014 {stem}", fontsize=14, weight="bold")
    has_g = "gratio" in df.columns and len(df) > 0
    axes[0, 0].hist(df["axon_diam"], bins=30, color="seagreen",    edgecolor="k", alpha=0.8) if len(df) else None
    axes[0, 0].set(xlabel="Axon diameter (\u00b5m)", ylabel="Count", title="Axon diameter")
    axes[0, 1].hist(df["gratio"], bins=30, color="steelblue",      edgecolor="k", alpha=0.8) if has_g else None
    axes[0, 1].set(xlabel="G-ratio", ylabel="Count", title="G-ratio")
    axes[0, 2].hist(df["myelin_thickness"], bins=30, color="mediumpurple", edgecolor="k", alpha=0.8) if (len(df) and "myelin_thickness" in df.columns) else None
    axes[0, 2].set(xlabel="Myelin thickness (\u00b5m)", ylabel="Count", title="Myelin thickness")
    if has_g:
        axes[1, 0].scatter(df["axon_diam"], df["gratio"], s=15, alpha=0.6, c="steelblue", edgecolors="none")
        axes[1, 0].axhline(0.6, ls="--", color="gray", lw=0.8, label="healthy \u2248 0.6")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="Axon diameter (\u00b5m)", ylabel="G-ratio", title="G-ratio vs. diameter")
    if "myelin_thickness" in df.columns and len(df):
        axes[1, 1].scatter(df["axon_diam"], df["myelin_thickness"], s=15, alpha=0.6, c="mediumpurple", edgecolors="none")
    axes[1, 1].set(xlabel="Axon diameter (\u00b5m)", ylabel="Myelin thickness (\u00b5m)", title="Myelin vs. diameter")
    ax = axes[1, 2]
    ax.axis("off")
    tbl_rows = [
        ["Fibers detected",   str(n_outer)],
        ["With axon",         str(n_matched)],
        ["Passed QC",         str(len(df))],
        ["Rejected QC",       str(len(df_rej))],
        ["", ""],
        ["AVF",               f"{agg.get('avf', 0):.4f}"],
        ["MVF",               f"{agg.get('mvf', 0):.4f}"],
        ["Aggr. G-ratio",    f"{agg.get('gratio_aggr', 0):.4f}"],
        ["Density (mm\u207b\u00b2)", f"{agg.get('axon_density_mm2', 0):.0f}"],
        ["Mean axon diam",    f"{df['axon_diam'].mean():.2f} \u00b5m" if len(df) else "\u2014"],
        ["Mean fiber diam",   f"{df['fiber_diam'].mean():.2f} \u00b5m" if len(df) else "\u2014"],
        ["Mean G-ratio",      f"{df['gratio'].mean():.3f}" if has_g else "\u2014"],
    ]
    tbl = ax.table(cellText=tbl_rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    ax.set_title("QC & Aggregate Metrics")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────── Pipeline ─────────────────────────────────────────────

def process_image(img_path: Path) -> tuple[str, int, dict]:
    stem = clean_stem(img_path)
    print(f"\n{'=' * 60}")
    print(f"  {img_path.name}  \u2192  {stem}")
    print(f"{'=' * 60}")

    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("  Reading image\u2026")
    img = io.imread(str(img_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # ── Step 1: Cellpose — outer fibers ────────────────────────────────
    cache_outer = out_dir / f"{stem}_cellpose_outer.npy"
    old_cache   = out_dir / f"{stem}_cellpose.npy"
    if not cache_outer.exists() and old_cache.exists():
        cache_outer = old_cache

    if cache_outer.exists():
        print("  [1/2] Cellpose (fibers) \u2014 loading from cache\u2026")
        outer_labels = np.load(str(cache_outer))
    else:
        print("  [1/2] Cellpose (fibers)\u2026")
        outer_labels = run_cellpose_fibers(img)
        np.save(str(out_dir / f"{stem}_cellpose_outer.npy"), outer_labels)
    n_outer = int(outer_labels.max())
    print(f"       \u2192 {n_outer} fibers")

    # ── Step 2: Build axon_input + Cellpose pass 2 ────────────────────
    cache_axon = out_dir / f"{stem}_axon_inner.npy"

    if cache_axon.exists():
        print("  [2/2] Axon detection \u2014 loading from cache\u2026")
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
            crop = inner_labels_raw[minr:maxr, minc:maxc] == lbl
            axon_assignments[int(lbl)] = (minr, minc, crop)
    else:
        print("  [2/2] Building axon input image\u2026")
        axon_input = build_axon_input(img, outer_labels)
        io.imsave(str(out_dir / f"{stem}_axon_input.png"), axon_input, check_contrast=False)

        print("       detecting axons (global Otsu)\u2026")
        axon_assignments = find_axons(axon_input, outer_labels)

        inner_labels_raw = np.zeros(outer_labels.shape, dtype=outer_labels.dtype)
        for fiber_label, (r0, c0, crop) in axon_assignments.items():
            inner_labels_raw[r0:r0 + crop.shape[0], c0:c0 + crop.shape[1]][crop] = fiber_label
        np.save(str(cache_axon), inner_labels_raw)

    # ── Morphometrics, QC, outputs ─────────────────────────────────────
    print("  Computing morphometrics\u2026")
    inner_labels, pairs, df_all, index_image, agg = process_fibers(
        outer_labels, axon_assignments, PIXEL_SIZE,
    )
    n_matched = len(pairs)
    print(f"       \u2192 {len(df_all)} axons measured")

    df_pass, df_rej = apply_qc(df_all)
    print(f"       \u2192 QC: {len(df_pass)} pass / {len(df_rej)} reject")

    cols = [c for c in df_pass.columns if not c.startswith("_")]
    df_pass[cols].to_csv(out_dir / f"{stem}_morphometrics.csv", index=False)
    try:
        df_pass[cols].to_excel(out_dir / f"{stem}_morphometrics.xlsx", index=False)
    except ImportError:
        print("       \u26a0 openpyxl not installed \u2014 .xlsx skipped")
    pd.DataFrame([agg]).to_csv(out_dir / f"{stem}_aggregate.csv", index=False)

    print("  Generating visualizations\u2026")
    overlay = _make_overlay(img, outer_labels, inner_labels, df_pass, df_rej)
    io.imsave(str(out_dir / f"{stem}_overlay.png"), overlay, check_contrast=False)

    numbered = _make_numbered(overlay, df_pass, n_outer, stem)
    io.imsave(str(out_dir / f"{stem}_numbered.png"), numbered, check_contrast=False)

    _make_gratio_map(img, df_pass, index_image, out_dir / f"{stem}_gratio_map.png")
    _make_dashboard(df_pass, df_rej, agg, n_outer, n_matched, stem,
                    out_dir / f"{stem}_dashboard.png")

    print(f"  \u2713 Done \u2014 {out_dir}")
    return stem, len(df_pass), agg


# ──────────────── Main ───────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in (".tif", ".tiff", ".png"))
    if not images:
        sys.exit(f"No images found in {INPUT_DIR}")

    print(f"Found {len(images)} image(s) in {INPUT_DIR}\n")
    results = []
    for p in images:
        try:
            stem, n, agg = process_image(p)
            results.append({"image": stem, "n_axons": n, **agg})
        except Exception as e:
            print(f"  \u2717 {p.name}: {e}")
            traceback.print_exc()

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
        print(f"\n{'=' * 60}")
        print(f"Done \u2014 {len(results)}/{len(images)} images")
        print(summary.to_string(index=False))
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
