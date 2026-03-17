"""visualization.py — Overlay, numbered image, g-ratio map, and dashboard.

Color scheme
------------
  GREEN  / teal-green boundary  = axon (QC passed)
  BLUE   / blue boundary        = myelin ring (QC passed)
  ORANGE + rejection code label = detected axon, rejected by QC
  RED                           = fiber with no axon detected
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw
from skimage import morphology
from skimage.segmentation import find_boundaries

import config
from utils import load_font, to_rgb_uint8

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Overlay ──────────────────────────────────────────────────────────────────


def make_overlay(
    img: np.ndarray,
    outer_labels: np.ndarray,
    inner_labels: np.ndarray,
    df_pass: pd.DataFrame,
    df_rej: pd.DataFrame,
) -> np.ndarray:
    """Build the colour-coded overlay image (returned as H×W×3 uint8 array).

    Layer order (back → front):
      dim background → red (no axon) → orange myelin → orange axon
      → blue myelin → green axon → boundaries → PIL text/legend
    """
    rgb = to_rgb_uint8(img)
    overlay = (rgb.astype(np.float32) * 0.4).astype(np.uint8)

    pass_fibers = set(df_pass["_fiber_label"].tolist()) if len(df_pass) else set()
    rej_fibers = set(df_rej["_fiber_label"].tolist()) if len(df_rej) else set()

    def _blend(mask: np.ndarray, color: list, alpha: float = 0.6) -> None:
        c = np.array(color, dtype=np.float32)
        overlay[mask] = np.clip(
            overlay[mask].astype(np.float32) * (1 - alpha) + c * alpha,
            0,
            255,
        ).astype(np.uint8)

    no_axon = (outer_labels > 0) & (inner_labels == 0)
    rej_axon = np.isin(inner_labels, list(rej_fibers)) & (inner_labels > 0)
    rej_myelin = np.isin(outer_labels, list(rej_fibers)) & ~rej_axon & (inner_labels == 0)
    pass_axon = np.isin(inner_labels, list(pass_fibers)) & (inner_labels > 0)
    pass_myel = np.isin(outer_labels, list(pass_fibers)) & ~pass_axon

    _blend(no_axon, [220, 50, 50])
    _blend(rej_myelin, [200, 120, 30], alpha=0.35)
    _blend(rej_axon, [255, 140, 0])
    _blend(pass_myel, [50, 50, 240])
    _blend(pass_axon, [0, 210, 60])

    outer_c = morphology.binary_dilation(
        find_boundaries(outer_labels, mode="thick"), morphology.disk(1)
    )
    inner_c = morphology.binary_dilation(
        find_boundaries(inner_labels, mode="thick"), morphology.disk(1)
    )
    overlay[outer_c] = [70, 70, 220]
    overlay[inner_c] = [0, 240, 80]

    # ── PIL layer: rejection codes + legend ──────────────────────────────
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    font_code = load_font(11)
    font_leg = load_font(13)

    # Rejection reason label on each orange fiber
    if len(df_rej) and "reject_reason" in df_rej.columns:
        for _, row in df_rej.iterrows():
            reason = str(row.get("reject_reason", ""))
            if not reason:
                continue
            x, y = int(row["x0"]), int(row["y0"])
            draw.text((x - 5, y - 6), reason, fill=(0, 0, 0), font=font_code)  # shadow
            draw.text((x - 6, y - 7), reason, fill=(255, 220, 60), font=font_code)

    # Legend: semi-transparent dark box in the bottom-left corner
    _, H = pil.width, pil.height
    lx, ly = 18, H - 205
    lw, lh = 278, 185
    arr = np.array(pil)
    arr[ly : ly + lh, lx : lx + lw] = np.clip(
        arr[ly : ly + lh, lx : lx + lw].astype(float) * 0.2 + np.array([18, 18, 18]) * 0.8,
        0,
        255,
    ).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([lx, ly, lx + lw, ly + lh], outline=(90, 90, 90), width=1)

    legend_entries = [
        ([0, 210, 60], "Axon — QC passed"),
        ([50, 50, 240], "Myelin — QC passed"),
        ([255, 140, 0], "Detected — QC rejected"),
        ([220, 50, 50], "No axon detected"),
    ]
    codes_lines = [
        "Rejection codes:",
        "G=g-ratio   Ø=diameter   sol=solidity",
        "ecc=eccen.  off=offset   brd=border",
    ]
    y_cur = ly + 10
    for color, label in legend_entries:
        draw.rectangle([lx + 10, y_cur + 2, lx + 23, y_cur + 14], fill=tuple(color))
        draw.text((lx + 32, y_cur), label, fill=(230, 230, 230), font=font_leg)
        y_cur += 23
    draw.line([lx + 8, y_cur + 2, lx + lw - 8, y_cur + 2], fill=(70, 70, 70), width=1)
    y_cur += 10
    for line in codes_lines:
        clr = (200, 200, 200) if line == "Rejection codes:" else (155, 155, 155)
        draw.text((lx + 10, y_cur), line, fill=clr, font=load_font(11))
        y_cur += 16

    return np.array(pil)


# ── Numbered image ────────────────────────────────────────────────────────────


def make_numbered(
    overlay: np.ndarray,
    df: pd.DataFrame,
    n_outer: int,
    stem: str,
) -> np.ndarray:
    """Add sequential yellow numbers on QC-passed axons + a stats banner."""
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    font = load_font(14)
    font_banner = load_font(20)

    for num, (_, row) in enumerate(df.iterrows(), start=1):
        draw.text(
            (int(row["x0"]) - 6, int(row["y0"]) - 8),
            str(num),
            fill=(255, 255, 0),
            font=font,
        )

    n = len(df)
    mean_ax = df["axon_diam"].mean() if n else 0
    mean_fi = df["fiber_diam"].mean() if n else 0
    mean_g = df["gratio"].mean() if n else 0

    banner_h = 40
    canvas = Image.new("RGB", (pil.width, pil.height + banner_h), (30, 30, 30))
    canvas.paste(pil, (0, banner_h))
    d = ImageDraw.Draw(canvas)
    d.text(
        (10, 10),
        f"  n={n}/{n_outer}   axon={mean_ax:.2f}µm   "
        f"fiber={mean_fi:.2f}µm   G={mean_g:.3f}  —  {stem}",
        fill=(255, 255, 255),
        font=font_banner,
    )
    return np.array(canvas)


# ── G-ratio heatmap ───────────────────────────────────────────────────────────


def make_gratio_map(
    img: np.ndarray,
    df: pd.DataFrame,
    index_image: np.ndarray,
    out_path,
) -> None:
    """Save a per-axon g-ratio heatmap (RdYlGn colourmap) to *out_path*."""
    bg = (to_rgb_uint8(img).astype(np.float32) * 0.3).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(bg)

    if "gratio" in df.columns and len(df) > 0:
        norm = Normalize(vmin=config.MIN_GRATIO, vmax=config.MAX_GRATIO)
        cmap = matplotlib.colormaps["RdYlGn"]
        rgba = np.zeros((*index_image.shape, 4), dtype=np.float32)
        for _, row in df.iterrows():
            gr, lbl = row.get("gratio", np.nan), int(row["_label"])
            if np.isnan(gr) or lbl == 0:
                continue
            rgba[index_image == lbl] = cmap(norm(np.clip(gr, config.MIN_GRATIO, config.MAX_GRATIO)))
        ax.imshow(rgba)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="G-ratio", shrink=0.6, pad=0.02)

    ax.axis("off")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Dashboard ─────────────────────────────────────────────────────────────────


def make_dashboard(
    df: pd.DataFrame,
    df_rej: pd.DataFrame,
    agg: dict,
    n_outer: int,
    n_matched: int,
    stem: str,
    out_path,
) -> None:
    """Save a 2×3 summary dashboard (histograms + scatter + metrics table)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Morphometry Dashboard — {stem}", fontsize=14, weight="bold")
    has_g = "gratio" in df.columns and len(df) > 0

    if len(df):
        axes[0, 0].hist(df["axon_diam"], bins=30, color="seagreen", edgecolor="k", alpha=0.8)
    axes[0, 0].set(xlabel="Axon diameter (µm)", ylabel="Count", title="Axon diameter")

    if has_g:
        axes[0, 1].hist(df["gratio"], bins=30, color="steelblue", edgecolor="k", alpha=0.8)
    axes[0, 1].set(xlabel="G-ratio", ylabel="Count", title="G-ratio")

    if len(df) and "myelin_thickness" in df.columns:
        axes[0, 2].hist(
            df["myelin_thickness"], bins=30, color="mediumpurple", edgecolor="k", alpha=0.8
        )
    axes[0, 2].set(xlabel="Myelin thickness (µm)", ylabel="Count", title="Myelin thickness")

    if has_g:
        axes[1, 0].scatter(
            df["axon_diam"], df["gratio"], s=15, alpha=0.6, c="steelblue", edgecolors="none"
        )
        axes[1, 0].axhline(0.6, ls="--", color="gray", lw=0.8, label="healthy ≈ 0.6")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set(xlabel="Axon diameter (µm)", ylabel="G-ratio", title="G-ratio vs. diameter")

    if "myelin_thickness" in df.columns and len(df):
        axes[1, 1].scatter(
            df["axon_diam"],
            df["myelin_thickness"],
            s=15,
            alpha=0.6,
            c="mediumpurple",
            edgecolors="none",
        )
    axes[1, 1].set(
        xlabel="Axon diameter (µm)", ylabel="Myelin thickness (µm)", title="Myelin vs. diameter"
    )

    ax = axes[1, 2]
    ax.axis("off")
    tbl_rows = [
        ["Fibers detected", str(n_outer)],
        ["With axon", str(n_matched)],
        ["Passed QC", str(len(df))],
        ["Rejected QC", str(len(df_rej))],
        ["", ""],
        ["AVF", f"{agg.get('avf', 0):.4f}"],
        ["MVF", f"{agg.get('mvf', 0):.4f}"],
        ["Aggr. G-ratio", f"{agg.get('gratio_aggr', 0):.4f}"],
        ["Density (mm⁻²)", f"{agg.get('axon_density_mm2', 0):.0f}"],
        ["Mean axon diam", f"{df['axon_diam'].mean():.2f} µm" if len(df) else "—"],
        ["Mean fiber diam", f"{df['fiber_diam'].mean():.2f} µm" if len(df) else "—"],
        ["Mean G-ratio", f"{df['gratio'].mean():.3f}" if has_g else "—"],
    ]
    tbl = ax.table(cellText=tbl_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    ax.set_title("QC & Aggregate Metrics")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
