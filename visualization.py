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
from skimage import morphology
from skimage.segmentation import find_boundaries

import config
from utils import load_font, to_rgb_uint8

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
    key_rows = {
        r: rows[r - 1]
        for r in range(1, n_rows + 1)
        if rows[r - 1][0]
        in ("QC validé", "QC rejeté", "Multi-cœur (exclu)", "Aggr. G-ratio", "AVF", "MVF")
    }
    for r, (label, value) in key_rows.items():
        val_cell = tbl[r, 1]
        if label == "QC validé":
            val_cell.set_facecolor("#D5F5E3")
        elif label == "QC rejeté":
            val_cell.set_facecolor("#FADBD8")
        elif label == "Multi-cœur (exclu)":
            val_cell.set_facecolor("#E8DAEF")
        elif label in ("Aggr. G-ratio", "AVF", "MVF"):
            val_cell.set_facecolor("#D6EAF8")

    ax.set_title("QC & Aggregate Metrics", pad=10)


# ── Overlay ───────────────────────────────────────────────────────────────────


def make_overlay(
    img: np.ndarray,
    outer_labels: np.ndarray,
    inner_labels: np.ndarray,
    df_pass: pd.DataFrame,
    df_rej: pd.DataFrame,
    multicore_labels: set | None = None,
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

    outer_c = morphology.binary_dilation(
        find_boundaries(outer_labels, mode="thick"), morphology.disk(1)
    )
    inner_c = morphology.binary_dilation(
        find_boundaries(inner_labels, mode="thick"), morphology.disk(1)
    )
    overlay[outer_c] = [70, 70, 220]
    overlay[inner_c] = [0, 240, 80]

    # ── PIL layer ─────────────────────────────────────────────────────────
    pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil)
    font_code = load_font(12)
    font_leg = load_font(13)

    # Rejection badges: dark pill with orange border + yellow text
    PAD = 4
    if len(df_rej) and "reject_reason" in df_rej.columns:
        for _, row in df_rej.iterrows():
            reason = str(row.get("reject_reason", ""))
            if not reason:
                continue
            x, y = int(row["x0"]), int(row["y0"])
            tx, ty = x - 7, y - 8
            bb = draw.textbbox((tx, ty), reason, font=font_code)
            draw.rectangle(
                [bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD],
                fill=(25, 15, 0),
                outline=(255, 160, 30),
                width=1,
            )
            draw.text((tx, ty), reason, fill=(255, 225, 80), font=font_code)

    # Multi-core badges: crimson pill with "MC" label
    if mc_fibers:
        from skimage import measure as _measure

        for p in _measure.regionprops(outer_labels):
            if p.label not in mc_fibers:
                continue
            cy, cx = int(p.centroid[0]), int(p.centroid[1])
            tx, ty = cx - 7, cy - 8
            bb = draw.textbbox((tx, ty), "MC", font=font_code)
            draw.rectangle(
                [bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD],
                fill=(60, 10, 20),
                outline=(210, 50, 85),
                width=1,
            )
            draw.text((tx, ty), "MC", fill=(255, 180, 190), font=font_code)

    # Legend box
    _, H = pil.width, pil.height
    lx, ly = 18, H - 255
    lw, lh = 320, 237
    arr = np.array(pil)
    arr[ly : ly + lh, lx : lx + lw] = np.clip(
        arr[ly : ly + lh, lx : lx + lw].astype(float) * 0.15 + np.array([18, 18, 18]) * 0.85,
        0,
        255,
    ).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)

    # rounded border
    draw.rectangle([lx, ly, lx + lw, ly + lh], outline=(100, 100, 110), width=1)

    entries = [
        ([0, 210, 60], "Axone — QC valide"),
        ([50, 50, 240], "Myéline — QC valide"),
        ([255, 140, 0], "Détecté — rejeté QC"),
        ([210, 50, 85], "MC — Multi-cœur (exclu)"),
        ([220, 50, 50], "Aucun axone trouvé"),
    ]
    codes = [
        "Codes rejet QC :",
        "G = g-ratio    Ø = diam.    sol = solidité",
        "ecc = excentr.  off = offset  brd = bord",
        "lgG = grande fibre (P85+) + G < 0.5",
    ]

    y_cur = ly + 12
    for color, label in entries:
        # circle swatch instead of square
        cx_sw, cy_sw = lx + 18, y_cur + 8
        draw.ellipse([cx_sw - 7, cy_sw - 7, cx_sw + 7, cy_sw + 7], fill=tuple(color))
        draw.text((lx + 34, y_cur), label, fill=(230, 230, 230), font=font_leg)
        y_cur += 25

    # divider
    draw.line([lx + 10, y_cur, lx + lw - 10, y_cur], fill=(70, 70, 80), width=1)
    y_cur += 8

    for i, line in enumerate(codes):
        clr = (210, 210, 220) if i == 0 else (150, 150, 160)
        fnt = load_font(11) if i > 0 else load_font(12)
        draw.text((lx + 12, y_cur), line, fill=clr, font=fnt)
        y_cur += 17

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

    banner_h = 44
    canvas = Image.new("RGB", (pil.width, pil.height + banner_h), (22, 28, 36))
    canvas.paste(pil, (0, banner_h))
    d = ImageDraw.Draw(canvas)
    d.text(
        (14, 12),
        f"n = {n} / {n_outer}     axon = {mean_ax:.2f} µm"
        f"     fiber = {mean_fi:.2f} µm     G = {mean_g:.3f}     {stem}",
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
    """Save a 2×3 publication-quality morphometry dashboard."""
    with plt.rc_context(_DASH_STYLE):
        fig = plt.figure(figsize=(20, 11), facecolor=_CLR["bg"])
        fig.suptitle(
            f"Nerve Morphometry  —  {stem}",
            fontsize=15,
            weight="bold",
            color=_CLR["text"],
            y=0.98,
        )

        gs = fig.add_gridspec(
            2, 3, hspace=0.42, wspace=0.35, left=0.06, right=0.97, top=0.93, bottom=0.07
        )
        axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

        has_g = "gratio" in df.columns and len(df) > 0
        norm_g = Normalize(vmin=config.MIN_GRATIO, vmax=config.MAX_GRATIO)
        cmap_g = matplotlib.colormaps["RdYlGn"]

        # ── row 0: histograms ─────────────────────────────────────────────
        _hist_with_stats(
            axes[0][0], df["axon_diam"], _CLR["axon"], "Axon diameter (µm)", "Axon diameter"
        )
        _hist_with_stats(
            axes[0][1],
            df["gratio"] if has_g else pd.Series(dtype=float),
            _CLR["myelin"],
            "G-ratio",
            "G-ratio",
        )
        _hist_with_stats(
            axes[0][2],
            df["myelin_thickness"] if "myelin_thickness" in df.columns else pd.Series(dtype=float),
            "#8E44AD",
            "Myelin thickness (µm)",
            "Myelin thickness",
        )

        # reference line on g-ratio histogram
        if has_g:
            axes[0][1].axvline(
                0.6, color=_CLR["rej"], lw=1.2, ls="-.", label="optimal ≈ 0.6", zorder=3
            )
            axes[0][1].legend(loc="upper right")

        # ── row 1: scatter + table ────────────────────────────────────────
        if has_g:
            _scatter_colored(
                axes[1][0],
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
            axes[1][0].set(
                xlabel="Axon diameter (µm)", ylabel="G-ratio", title="G-ratio vs. Axon diameter"
            )

        if "myelin_thickness" in df.columns and len(df):
            _scatter_colored(
                axes[1][1],
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
            axes[1][1].set(
                xlabel="Axon diameter (µm)",
                ylabel="Myelin thickness (µm)",
                title="Myelin thickness vs. Axon diameter",
            )

        # ── metrics table ─────────────────────────────────────────────────
        tbl_rows = [
            ["Fibres détectées", str(n_outer)],
            ["Multi-cœur (exclu)", str(n_multicore)],
            ["Avec axone", str(n_matched)],
            ["QC validé", str(len(df))],
            ["QC rejeté", str(len(df_rej))],
            ["—", "—"],
            ["AVF", f"{agg.get('avf', 0):.4f}"],
            ["MVF", f"{agg.get('mvf', 0):.4f}"],
            ["N-ratio (fibres/total)", f"{agg.get('nratio', 0):.4f}"],
            ["Aggr. G-ratio", f"{agg.get('gratio_aggr', 0):.4f}"],
            ["Density (mm⁻²)", f"{agg.get('axon_density_mm2', 0):.0f}"],
            ["Mean axon diam", f"{df['axon_diam'].mean():.2f} µm" if len(df) else "—"],
            ["Mean fiber diam", f"{df['fiber_diam'].mean():.2f} µm" if len(df) else "—"],
            ["Mean G-ratio", f"{df['gratio'].mean():.3f}" if has_g else "—"],
        ]
        _styled_table(axes[1][2], tbl_rows, len(df), len(df_rej))

        # ── QC pass/reject mini-bar in table panel top ────────────────────
        total = len(df) + len(df_rej)
        if total > 0:
            inset = axes[1][2].inset_axes([0.0, 0.90, 1.0, 0.08])
            inset.barh(0, len(df) / total, color="#27AE60", height=0.6)
            inset.barh(0, len(df_rej) / total, color="#E74C3C", height=0.6, left=len(df) / total)
            inset.set_xlim(0, 1)
            inset.axis("off")
            pct = len(df) / total * 100
            inset.text(
                0.01,
                0,
                f"  {pct:.0f}% pass",
                va="center",
                fontsize=7.5,
                color="white",
                weight="bold",
            )

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
