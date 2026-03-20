"""compare.py — Cross-sample morphometry comparison dashboard.

Usage
-----
  python compare.py path/to/A_morphometrics.csv path/to/B_morphometrics.csv \\
      --labels "12w L" "12w R" --out output/comparison/

  # Auto-discover all morphometrics CSVs under output/:
  python compare.py output/**/*_morphometrics.csv --labels "L" "R"

Outputs
-------
  <out>/comparison_dashboard.png   — multi-panel figure
  <out>/comparison_summary.csv     — per-group aggregate stats + MW test
"""

import argparse
import pathlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Design tokens (mirrored from visualization.py) ────────────────────────────

_CLR = {
    "bg": "#F8F9FA",
    "grid": "#E9ECEF",
    "text": "#2C3E50",
    "muted": "#95A5A6",
    "header": "#2C3E50",
}

_GROUP_COLORS = [
    "#27AE60",  # green
    "#E67E22",  # orange
    "#2980B9",  # blue
    "#8E44AD",  # purple
    "#E74C3C",  # red
    "#16A085",  # teal
]

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
    "legend.fontsize": 9,
}


# ── Statistical helpers ────────────────────────────────────────────────────────


def _regression(x, y):
    """Linear fit. Returns (slope, intercept, r, p, x_line, y_line) or None."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    xr = np.linspace(x[mask].min(), x[mask].max(), 100)
    return slope, intercept, r, p, xr, slope * xr + intercept


def _mannwhitney(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u, p


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled else 0.0


def _pval_str(p):
    if np.isnan(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return "p < 0.01 **"
    if p < 0.05:
        return "p < 0.05 *"
    return f"p = {p:.3f} ns"


# ── Plot panels ───────────────────────────────────────────────────────────────


def _plot_scatter_regression(ax, groups, colors, labels):
    """Axon diameter vs g-ratio — points + per-group regression line."""
    for df, color, label in zip(groups, colors, labels, strict=False):
        x, y = df["axon_diam"].values, df["gratio"].values
        ax.scatter(x, y, c=color, s=8, alpha=0.35, linewidths=0, label=label)
        res = _regression(x, y)
        if res:
            slope, intercept, r, p, xr, yr = res
            ax.plot(xr, yr, color=color, lw=2.2, label=f"{label} (r={r:.2f}, {_pval_str(p)})")
    ax.set(xlabel="Axon diameter (µm)", ylabel="g-ratio", title="G-ratio vs Axon diameter")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")


def _plot_binned_gratio(ax, groups, colors, labels):
    """Mean g-ratio within axon-diameter bins — size-controlled comparison."""
    all_x = np.concatenate([df["axon_diam"].values for df in groups])
    all_x = all_x[np.isfinite(all_x)]
    bin_edges = np.percentile(all_x, np.linspace(0, 100, 7))
    bin_edges = np.unique(bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_w = (centers[1] - centers[0]) * 0.35 if len(centers) > 1 else 0.3
    offsets = np.linspace(
        -bar_w * (len(groups) - 1) / 2, bar_w * (len(groups) - 1) / 2, len(groups)
    )
    for df, color, label, offset in zip(groups, colors, labels, offsets, strict=False):
        means, sems = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
            sub = df[(df["axon_diam"] >= lo) & (df["axon_diam"] < hi)]["gratio"]
            means.append(sub.mean() if len(sub) else np.nan)
            sems.append(sub.sem() if len(sub) > 1 else np.nan)
        ax.bar(
            centers + offset,
            means,
            width=bar_w * 0.88,
            color=color,
            alpha=0.82,
            label=label,
            yerr=sems,
            error_kw={"elinewidth": 1, "capsize": 3, "ecolor": _CLR["muted"]},
        )
    ax.set(
        xlabel="Axon diameter bin (µm)",
        ylabel="Mean g-ratio ± SEM",
        title="Size-matched g-ratio comparison",
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(bin_edges)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.legend()


def _plot_violin(ax, groups, colors, labels, metric, ylabel, title):
    """Violin + jitter + median line for one metric."""
    data = [df[metric].dropna().values for df in groups]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors, strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
    parts["cmedians"].set_color(_CLR["text"])
    parts["cmedians"].set_linewidth(2)
    rng = np.random.default_rng(42)
    for i, (d, color) in enumerate(zip(data, colors, strict=False)):
        jitter = rng.uniform(-0.07, 0.07, size=len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, s=4, color=color, alpha=0.25, linewidths=0)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set(ylabel=ylabel, title=title)
    if len(groups) == 2:
        _, p = _mannwhitney(data[0], data[1])
        d = _cohens_d(data[0], data[1])
        ax.text(
            0.98,
            0.98,
            f"{_pval_str(p)}\nd = {d:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=_CLR["muted"],
        )


def _plot_stats_table(ax, groups, colors, labels):
    """Styled summary table with MW test row."""
    ax.axis("off")

    metrics = [
        ("n fibers", lambda d: str(len(d))),
        (
            "axon diam (µm)",
            lambda d: f"{d['axon_diam'].mean():.2f} ± {d['axon_diam'].std():.2f}",
        ),
        ("g-ratio", lambda d: f"{d['gratio'].mean():.3f} ± {d['gratio'].std():.3f}"),
        (
            "myelin thickness (µm)",
            lambda d: f"{d['myelin_thickness'].mean():.2f} ± {d['myelin_thickness'].std():.2f}",
        ),
        (
            "fiber diam (µm)",
            lambda d: f"{d['fiber_diam'].mean():.2f} ± {d['fiber_diam'].std():.2f}",
        ),
        ("solidity", lambda d: f"{d['solidity'].mean():.3f}"),
    ]

    col_labels = ["Metric"] + labels
    cell_data = [[m[0]] + [m[1](df) for df in groups] for m in metrics]

    if len(groups) == 2:
        _, p = _mannwhitney(groups[0]["gratio"], groups[1]["gratio"])
        d_val = _cohens_d(groups[0]["gratio"], groups[1]["gratio"])
        cell_data.append(["g-ratio MW test", _pval_str(p), f"Cohen's d = {d_val:.2f}"])

    tbl = ax.table(cellText=cell_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.65)

    n_cols = len(col_labels)
    # Header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(_CLR["header"])
        cell.set_text_props(color="white", fontweight="bold")
    # Group-colored column headers
    for j, color in enumerate(colors):
        cell = tbl[0, j + 1]
        cell.set_facecolor(color)
        cell.set_text_props(color="white", fontweight="bold")
    # Alternating rows
    for i in range(1, len(cell_data) + 1):
        bg = "#F0F4F8" if i % 2 == 0 else "#FFFFFF"
        for j in range(n_cols):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Summary Statistics (mean ± std)",
        pad=10,
        fontsize=12,
        fontweight="bold",
        color=_CLR["text"],
    )


# ── Dashboard ─────────────────────────────────────────────────────────────────


def make_comparison_dashboard(groups, labels, out_path):
    colors = _GROUP_COLORS[: len(groups)]

    with plt.rc_context(_DASH_STYLE):
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "Cross-sample Morphometry Comparison",
            fontsize=16,
            fontweight="bold",
            color=_CLR["text"],
            y=0.985,
        )
        gs = fig.add_gridspec(
            3, 3, hspace=0.44, wspace=0.35, left=0.06, right=0.97, top=0.94, bottom=0.06
        )

        ax_scat = fig.add_subplot(gs[0, :2])
        ax_bin = fig.add_subplot(gs[1, :2])
        ax_vg = fig.add_subplot(gs[0, 2])
        ax_va = fig.add_subplot(gs[1, 2])
        ax_vm = fig.add_subplot(gs[2, 2])
        ax_tbl = fig.add_subplot(gs[2, :2])

        _plot_scatter_regression(ax_scat, groups, colors, labels)
        _plot_binned_gratio(ax_bin, groups, colors, labels)
        _plot_violin(ax_vg, groups, colors, labels, "gratio", "g-ratio", "G-ratio")
        _plot_violin(ax_va, groups, colors, labels, "axon_diam", "µm", "Axon diameter")
        _plot_violin(ax_vm, groups, colors, labels, "myelin_thickness", "µm", "Myelin thickness")
        _plot_stats_table(ax_tbl, groups, colors, labels)

        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  → {out_path}")


def make_comparison_summary(groups, labels, out_path):
    rows = []
    for df, label in zip(groups, labels, strict=False):
        rows.append(
            {
                "group": label,
                "n": len(df),
                "axon_diam_mean": df["axon_diam"].mean(),
                "axon_diam_std": df["axon_diam"].std(),
                "gratio_mean": df["gratio"].mean(),
                "gratio_std": df["gratio"].std(),
                "myelin_thickness_mean": df["myelin_thickness"].mean(),
                "myelin_thickness_std": df["myelin_thickness"].std(),
                "fiber_diam_mean": df["fiber_diam"].mean(),
                "fiber_diam_std": df["fiber_diam"].std(),
                "solidity_mean": df["solidity"].mean(),
            }
        )

    # Pooled row (weighted by n)
    pooled = pd.concat(groups, ignore_index=True)
    rows.append(
        {
            "group": "POOLED (all)",
            "n": len(pooled),
            "axon_diam_mean": pooled["axon_diam"].mean(),
            "axon_diam_std": pooled["axon_diam"].std(),
            "gratio_mean": pooled["gratio"].mean(),
            "gratio_std": pooled["gratio"].std(),
            "myelin_thickness_mean": pooled["myelin_thickness"].mean(),
            "myelin_thickness_std": pooled["myelin_thickness"].std(),
            "fiber_diam_mean": pooled["fiber_diam"].mean(),
            "fiber_diam_std": pooled["fiber_diam"].std(),
            "solidity_mean": pooled["solidity"].mean(),
        }
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"  → {out_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare morphometrics across multiple samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csvs", nargs="+", help="Paths to *_morphometrics.csv files")
    parser.add_argument("--labels", nargs="+", help="Label per CSV (default: filename stem)")
    parser.add_argument("--out", default="output/comparison", help="Output directory")
    args = parser.parse_args()

    csvs = [pathlib.Path(p) for p in args.csvs]
    labels = args.labels or [p.stem.replace("_morphometrics", "") for p in csvs]

    if len(labels) != len(csvs):
        print("ERROR: --labels count must match number of CSV files", file=sys.stderr)
        sys.exit(1)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(csvs)} sample(s):")
    groups = []
    for csv, label in zip(csvs, labels, strict=False):
        df = pd.read_csv(csv)
        print(f"  [{label}]  {len(df)} fibers  ←  {csv}")
        groups.append(df)

    print("\nGenerating comparison dashboard...")
    make_comparison_dashboard(groups, labels, out_dir / "comparison_dashboard.png")

    print("Generating summary CSV...")
    summary = make_comparison_summary(groups, labels, out_dir / "comparison_summary.csv")

    print("\n── Summary ─────────────────────────────────────────────────────")
    print(summary.to_string(index=False, float_format="%.3f"))

    if len(groups) == 2:
        _, p = _mannwhitney(groups[0]["gratio"], groups[1]["gratio"])
        d = _cohens_d(groups[0]["gratio"], groups[1]["gratio"])
        print(f"\nG-ratio Mann-Whitney: {_pval_str(p)}  |  Cohen's d = {d:.3f}")


if __name__ == "__main__":
    main()
