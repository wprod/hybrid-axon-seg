"""visualization.py — Overlay, numbered image, g-ratio map, and dashboard.

Color scheme
------------
  GREEN  / teal-green boundary  = axon (QC passed)
  BLUE   / blue boundary        = myelin ring (QC passed)
  ORANGE + pill badge           = detected axon, rejected by QC
  PURPLE                        = multi-core fiber (2+ dark blobs → excluded)
  RED                           = fiber with no axon detected (other)
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
from skimage.segmentation import find_boundaries

import config
from utils import build_fascicle_mask, load_font, to_rgb_uint8

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Design tokens ─────────────────────────────────────────────────────────────

_CLR = {
    "axon": "#27AE60",
    "myelin": "#2980B9",
    "rej": "#E67E22",
    "noaxon": "#E74C3C",
    "multicore": "#8E44AD",
    "bg": "#F8F9FA",
    "grid": "#E9ECEF",
    "text": "#2C3E50",
    "muted": "#95A5A6",
    "header": "#2C3E50",
}

# Rejection reason → RGB colour (0-255) used in overlay + rejection map
_REJ_COLORS = {
    "G": (255, 140, 0),  # orange        — g-ratio
    "lgG": (255, 70, 30),  # red-orange    — large fiber low g-ratio
    "shp": (220, 200, 0),  # amber         — shape mismatch
    "sol": (180, 60, 210),  # purple        — low solidity
    "off": (30, 170, 230),  # sky blue      — axon off-center
    "ecc": (240, 60, 140),  # pink          — eccentricity
    "Ø": (20, 200, 160),  # teal          — too small
    "brd": (160, 160, 160),  # grey          — touches border
}

_REJ_LABELS = {
    "G": "G — G-ratio out of range",
    "lgG": "lgG — Large fiber + low G-ratio",
    "shp": "shp — Axon shape ≠ fiber shape",
    "sol": "sol — Low solidity",
    "off": "off — Axon off-center",
    "ecc": "ecc — Axon too elongated",
    "Ø": "Ø — Axon too small",
    "brd": "brd — Touches border",
}

_DASH_STYLE = {
    "figure.facecolor": _CLR["bg"],
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#DEE2E6",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": _CLR["grid"],
    "grid.linewidth": 0.8,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": _CLR["text"],
    "axes.labelsize": 10,
    "axes.labelcolor": _CLR["text"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": _CLR["muted"],
    "ytick.color": _CLR["muted"],
    "legend.framealpha": 0.9,
    "legend.edgecolor": _CLR["grid"],
    "legend.fontsize": 8,
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hist_with_stats(ax, data, color, xlabel, title):
    """Histogram + mean/median lines + stats annotation."""
    if not len(data):
        ax.set(xlabel=xlabel, title=title)
        return
    ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    mean, med, std = data.mean(), data.median(), data.std()
    ax.axvline(mean, color=_CLR["text"], lw=1.5, ls="--", label=f"mean {mean:.2f}")
    ax.axvline(med, color=_CLR["muted"], lw=1.2, ls=":", label=f"median {med:.2f}")
    ax.legend(loc="upper right")
    ax.text(
        0.97,
        0.93,
        f"σ = {std:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=_CLR["muted"],
    )
    ax.set(xlabel=xlabel, ylabel="Count", title=title)


def _scatter_colored(ax, x, y, c, cmap, norm, xlabel, ylabel, title, ref_line=None):
    """Scatter colored by a continuous value + optional horizontal reference."""
    if not len(x):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        return
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=12, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    if ref_line is not None:
        ax.axhline(ref_line, ls="--", color=_CLR["muted"], lw=1.0, label=f"ref {ref_line}")
        ax.legend(loc="upper right")
    # Pearson r
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        ax.text(
            0.03,
            0.95,
            f"r = {r:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=_CLR["muted"],
        )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)


def _styled_table(ax, rows, n_pass, n_rej):
    """Render a metrics table with a dark header and alternating row shading."""
    ax.axis("off")
    col_labels = ["Metric", "Value"]

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    n_rows = len(rows)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DEE2E6")
        cell.set_linewidth(0.5)
        if r == 0:  # header
            cell.set_facecolor(_CLR["header"])
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.085)
        elif r % 2 == 0:
            cell.set_facecolor("#F8F9FA")
        else:
            cell.set_facecolor("#FFFFFF")
        if r > 0:
            cell.set_height(0.072)

    # Colour-code a few key cells
    _GREEN = {"— valid"}
    _RED = {"— QC rejected"}
    _PURPLE = {"— multi-core"}
    _BLUE = {
        "G-ratio (mean)",
        "G-ratio (area-weighted)",
        "AVF  (axon volume fraction)",
        "MVF  (myelin volume fraction)",
        "N-ratio  (fibers / nerve)",
        "Nerve area (mm²)",
        "Total axon area (mm²)",
        "Total myelin area (mm²)",
    }
    key_rows = {
        r: rows[r - 1]
        for r in range(1, n_rows + 1)
        if rows[r - 1][0] in (_GREEN | _RED | _PURPLE | _BLUE)
    }
    for r, (label, value) in key_rows.items():
        val_cell = tbl[r, 1]
        if label in _GREEN:
            val_cell.set_facecolor("#D5F5E3")
        elif label in _RED:
            val_cell.set_facecolor("#FADBD8")
        elif label in _PURPLE:
            val_cell.set_facecolor("#E8DAEF")
        elif label in _BLUE:
            val_cell.set_facecolor("#D6EAF8")


# ── Overlay ───────────────────────────────────────────────────────────────────


def make_overlay(
    img: np.ndarray,
    outer_labels: np.ndarray,
    inner_labels: np.ndarray,
    df_pass: pd.DataFrame,
    df_rej: pd.DataFrame,
    multicore_labels: set | None = None,
    fascicle_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the colour-coded overlay image (returned as H×W×3 uint8 array)."""
    rgb = to_rgb_uint8(img)
    overlay = (rgb.astype(np.float32) * 0.4).astype(np.uint8)

    pass_fibers = set(df_pass["_fiber_label"].tolist()) if len(df_pass) else set()
    rej_fibers = set(df_rej["_fiber_label"].tolist()) if len(df_rej) else set()
    mc_fibers = multicore_labels if multicore_labels else set()

    def _blend(mask: np.ndarray, color: list, alpha: float = 0.6) -> None:
        c = np.array(color, dtype=np.float32)
        overlay[mask] = np.clip(
            overlay[mask].astype(np.float32) * (1 - alpha) + c * alpha,
            0,
            255,
        ).astype(np.uint8)

    # ── Fascicle boundary — striped overlay drawn first (behind everything) ──
    if fascicle_mask is None:
        fascicle_mask = build_fascicle_mask(outer_labels, config.PIXEL_SIZE, config.CP_DIAM_UM)

    # 2px semi-transparent white perimeter
    fascicle_boundary = find_boundaries(fascicle_mask, mode="outer")
    _blend(fascicle_boundary, [255, 255, 255], alpha=0.5)

    # Fibers with no axon: split by reason
    multicore_mask = np.isin(outer_labels, list(mc_fibers)) & (inner_labels == 0)
    no_axon = (outer_labels > 0) & (inner_labels == 0) & ~multicore_mask
    rej_axon = np.isin(inner_labels, list(rej_fibers)) & (inner_labels > 0)
    rej_myel = np.isin(outer_labels, list(rej_fibers)) & ~rej_axon & (inner_labels == 0)
    pass_axon = np.isin(inner_labels, list(pass_fibers)) & (inner_labels > 0)
    pass_myel = np.isin(outer_labels, list(pass_fibers)) & ~pass_axon

    _blend(no_axon, [220, 50, 50])
    _blend(multicore_mask, [210, 50, 85])  # crimson-red (distinct from plain red)

    _blend(rej_myel, [200, 120, 30], alpha=0.35)
    _blend(rej_axon, [255, 140, 0])

    _blend(pass_myel, [50, 50, 240])
    _blend(pass_axon, [0, 210, 60])

    # 1-px boundaries: mode="inner" sits just inside each labelled region
    outer_c = find_boundaries(outer_labels, mode="inner")
    inner_c = find_boundaries(inner_labels, mode="inner")

    overlay[outer_c] = [70, 70, 220]

    # Green contour only for passed axons
    pass_c = inner_c & np.isin(inner_labels, list(pass_fibers))
    overlay[pass_c] = [0, 240, 80]

    overlay[inner_c & np.isin(inner_labels, list(rej_fibers))] = [255, 140, 0]

    # ── PIL layer — legend only (no per-fiber badges) ─────────────────────
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    font_leg = load_font(13)

    entries = [
        ([0, 210, 60], "Axon — QC passed"),
        ([50, 50, 240], "Myelin — QC passed"),
        ([255, 140, 0], "Detected — QC rejected"),
        ([210, 50, 85], "Multi-core (excluded)"),
        ([220, 50, 50], "No axon found"),
    ]

    ENTRY_H = 24
    lx = 18
    lw = 280
    lh = 12 + len(entries) * ENTRY_H + 8
    _, H = pil.width, pil.height
    ly = H - lh - 10

    arr = np.array(pil)
    arr[ly : ly + lh, lx : lx + lw] = np.clip(
        arr[ly : ly + lh, lx : lx + lw].astype(float) * 0.15 + np.array([18, 18, 18]) * 0.85,
        0,
        255,
    ).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)

    draw.rectangle([lx, ly, lx + lw, ly + lh], outline=(100, 100, 110), width=1)

    y_cur = ly + 12
    for color, label in entries:
        cx_sw, cy_sw = lx + 18, y_cur + 8
        draw.ellipse([cx_sw - 7, cy_sw - 7, cx_sw + 7, cy_sw + 7], fill=tuple(color))
        draw.text((lx + 34, y_cur), label, fill=(230, 230, 230), font=font_leg)
        y_cur += ENTRY_H

    return np.array(pil)


# ── Numbered image ────────────────────────────────────────────────────────────


def make_numbered(
    overlay: np.ndarray,
    df: pd.DataFrame,
    n_outer: int,
    stem: str,
    nerve_area_mm2: float = 0.0,
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

    banner_h = 44
    canvas = Image.new("RGB", (pil.width, pil.height + banner_h), (22, 28, 36))
    canvas.paste(pil, (0, banner_h))
    d = ImageDraw.Draw(canvas)
    area_str = f"     nerve area = {nerve_area_mm2:.4f} mm²" if nerve_area_mm2 else ""
    d.text(
        (14, 12),
        f"n = {n} / {n_outer}     axon = {mean_ax:.2f} µm"
        f"     fiber = {mean_fi:.2f} µm     G = {mean_g:.3f}{area_str}     {stem}",
        fill=(220, 220, 230),
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
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#111111")
    ax.set_facecolor("#111111")
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
        cb = plt.colorbar(sm, ax=ax, label="G-ratio", shrink=0.55, pad=0.02)
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")

    ax.axis("off")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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
    n_multicore: int = 0,
) -> None:
    """Save a 2×4 publication-quality morphometry dashboard."""
    with plt.rc_context(_DASH_STYLE):
        fig = plt.figure(figsize=(24, 11), facecolor=_CLR["bg"])
        fig.suptitle(
            f"Nerve Morphometry  —  {stem}",
            fontsize=15,
            weight="bold",
            color=_CLR["text"],
            y=0.98,
        )

        gs = fig.add_gridspec(
            2, 4, hspace=0.42, wspace=0.35, left=0.05, right=0.97, top=0.93, bottom=0.07
        )

        has_g = "gratio" in df.columns and len(df) > 0
        norm_g = Normalize(vmin=config.MIN_GRATIO, vmax=config.MAX_GRATIO)
        cmap_g = matplotlib.colormaps["RdYlGn"]

        # ── row 0: histograms (4 panels) ─────────────────────────────────
        ax_h = [fig.add_subplot(gs[0, c]) for c in range(4)]
        _hist_with_stats(
            ax_h[0], df["axon_diam"], _CLR["axon"], "Axon diameter (µm)", "Axon diameter"
        )
        _hist_with_stats(
            ax_h[1],
            df["fiber_diam"] if "fiber_diam" in df.columns else pd.Series(dtype=float),
            _CLR["myelin"],
            "Fiber diameter (µm)",
            "Fiber diameter",
        )
        _hist_with_stats(
            ax_h[2],
            df["gratio"] if has_g else pd.Series(dtype=float),
            "#2980B9",
            "G-ratio",
            "G-ratio",
        )
        _hist_with_stats(
            ax_h[3],
            df["myelin_thickness"] if "myelin_thickness" in df.columns else pd.Series(dtype=float),
            "#8E44AD",
            "Myelin thickness (µm)",
            "Myelin thickness",
        )

        # reference line on g-ratio histogram
        if has_g:
            ax_h[2].axvline(
                0.6, color=_CLR["rej"], lw=1.2, ls="-.", label="optimal ≈ 0.6", zorder=3
            )
            ax_h[2].legend(loc="upper right")

        # ── row 1: scatter + table ────────────────────────────────────────
        ax_s0 = fig.add_subplot(gs[1, 0])
        ax_s1 = fig.add_subplot(gs[1, 1])

        if has_g:
            _scatter_colored(
                ax_s0,
                x=df["axon_diam"].values,
                y=df["gratio"].values,
                c=df["gratio"].values,
                cmap=cmap_g,
                norm=norm_g,
                xlabel="Axon diameter (µm)",
                ylabel="G-ratio",
                title="G-ratio vs. Axon diameter",
                ref_line=0.6,
            )
        else:
            ax_s0.set(
                xlabel="Axon diameter (µm)", ylabel="G-ratio", title="G-ratio vs. Axon diameter"
            )

        if "myelin_thickness" in df.columns and len(df):
            _scatter_colored(
                ax_s1,
                x=df["axon_diam"].values,
                y=df["myelin_thickness"].values,
                c=df["myelin_thickness"].values,
                cmap=matplotlib.colormaps["plasma"],
                norm=Normalize(
                    vmin=df["myelin_thickness"].quantile(0.05),
                    vmax=df["myelin_thickness"].quantile(0.95),
                ),
                xlabel="Axon diameter (µm)",
                ylabel="Myelin thickness (µm)",
                title="Myelin thickness vs. Axon diameter",
            )
        else:
            ax_s1.set(
                xlabel="Axon diameter (µm)",
                ylabel="Myelin thickness (µm)",
                title="Myelin thickness vs. Axon diameter",
            )

        # ── right panel: rejection breakdown + metrics table ──────────────
        ax_right = fig.add_subplot(gs[1, 2:])
        ax_right.axis("off")
        ax_right.set_title(
            "Segmentation quality", pad=8, fontsize=11, fontweight="bold", color=_CLR["text"]
        )

        # Rejection breakdown data
        n_no_axon = n_outer - n_matched - n_multicore
        reason_counts = (
            df_rej["reject_reason"].value_counts().to_dict()
            if len(df_rej) and "reject_reason" in df_rej.columns
            else {}
        )
        breakdown = [("✓  Valid — axon confirmed", len(df), "#27AE60")]
        for code in _REJ_LABELS:
            if code in reason_counts:
                clr = tuple(c / 255 for c in _REJ_COLORS.get(code, (255, 140, 0)))
                breakdown.append((_REJ_LABELS[code], reason_counts[code], clr))
        if n_multicore:
            breakdown.append(("MC — Multi-core (2+ cores detected)", n_multicore, "#D2355A"))
        if n_no_axon > 0:
            breakdown.append(("∅  No axon detected", n_no_axon, "#E74C3C"))

        # Draw horizontal bars
        inset_bd = ax_right.inset_axes([0.0, 0.55, 1.0, 0.43])
        inset_bd.axis("off")
        total_all = max(sum(v for _, v, _ in breakdown), 1)
        for i, (label, count, color) in enumerate(reversed(breakdown)):
            inset_bd.barh(i, count / total_all, color=color, height=0.65, alpha=0.88)
            inset_bd.text(
                count / total_all + 0.01,
                i,
                f"{count}",
                va="center",
                fontsize=7.5,
                color=_CLR["text"],
                weight="bold",
            )
            inset_bd.text(-0.01, i, label, va="center", ha="right", fontsize=7, color=_CLR["text"])
        inset_bd.set_xlim(-0.55, 1.18)
        inset_bd.set_ylim(-0.6, len(breakdown) - 0.4)

        # Metrics table (bottom half) — clinician key metrics first
        tbl_rows = [
            ["Cellpose fibers", str(n_outer)],
            ["— valid", str(len(df))],
            ["— QC rejected", str(len(df_rej))],
            ["— multi-core", str(n_multicore)],
            ["— no axon", str(n_no_axon)],
            ["Nerve area (mm²)", f"{agg.get('nerve_area_mm2', 0):.4f}"],
            *(
                [["— exclusion (mm²)", f"{agg['exclusion_area_mm2']:.4f}"]]
                if agg.get("exclusion_area_mm2", 0) > 0
                else []
            ),
            ["Total fiber area (mm²)", f"{agg.get('total_fiber_area_mm2', 0):.4f}"],
            ["— myelin (mm²)", f"{agg.get('total_myelin_area_mm2', 0):.4f}"],
            ["— axon (mm²)", f"{agg.get('total_axon_area_mm2', 0):.4f}"],
            ["N-ratio  (fibers / nerve)", f"{agg.get('nratio', 0):.4f}"],
            ["G-ratio (mean)", f"{agg.get('gratio_aggr', 0):.4f}"],
            ["G-ratio (area-weighted)", f"{agg.get('gratio_area_weighted', 0):.4f}"],
            ["AVF  (axon volume fraction)", f"{agg.get('avf', 0):.4f}"],
            ["MVF  (myelin volume fraction)", f"{agg.get('mvf', 0):.4f}"],
            ["Axon density (mm⁻²)", f"{agg.get('axon_density_mm2', 0):.0f}"],
            ["Mean axon diameter", f"{df['axon_diam'].mean():.2f} µm" if len(df) else "—"],
            ["Mean fiber diameter", f"{df['fiber_diam'].mean():.2f} µm" if len(df) else "—"],
        ]
        inset_tbl = ax_right.inset_axes([0.0, 0.0, 1.0, 0.53])
        _styled_table(inset_tbl, tbl_rows, len(df), len(df_rej))

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


# ── Multi-image comparison ────────────────────────────────────────────────────
