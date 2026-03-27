"""compare.py — Cross-sample morphometry comparison dashboard.

Usage
-----
  # Group 200 images by condition extracted from filename:
  python compare.py --group-by "(auto|allo)"
  python compare.py --group-by "(BMSC|ADSC|OSC|X)"

  # Pool by L/R side:
  python compare.py --group-by LR

  # All samples individually (few samples only):
  python compare.py

  # Explicit CSVs with labels:
  python compare.py A.csv B.csv --labels "Group A" "Group B"

Outputs
-------
  <out>/comparison_dashboard.png   — multi-panel clinical figure
  <out>/comparison_summary.csv     — per-condition aggregate stats
"""

import argparse
import pathlib
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Design tokens ─────────────────────────────────────────────────────────────

_CLR = {
    "bg": "#F8F9FA",
    "grid": "#E9ECEF",
    "text": "#2C3E50",
    "muted": "#95A5A6",
    "header": "#2C3E50",
}

_PALETTE = [
    "#27AE60",
    "#E67E22",
    "#2980B9",
    "#8E44AD",
    "#E74C3C",
    "#16A085",
    "#F39C12",
    "#C0392B",
    "#1ABC9C",
    "#D35400",
    "#7F8C8D",
    "#2C3E50",
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
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": _CLR["text"],
    "axes.labelsize": 9,
    "axes.labelcolor": _CLR["text"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": _CLR["muted"],
    "ytick.color": _CLR["muted"],
    "legend.framealpha": 0.9,
    "legend.edgecolor": _CLR["grid"],
    "legend.fontsize": 8,
}

_MAX_JITTER_PTS = 1500  # max fiber dots per violin


# ── Statistical helpers ────────────────────────────────────────────────────────


def _regression(x, y):
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


def _kruskal(arrays):
    """Kruskal-Wallis across N arrays. Returns (H, p)."""
    clean = [np.asarray(g)[np.isfinite(np.asarray(g))] for g in arrays]
    clean = [g for g in clean if len(g) >= 2]
    if len(clean) < 2:
        return np.nan, np.nan
    return stats.kruskal(*clean)


def _cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled else 0.0


def _pval_stars(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _pval_full(p):
    if np.isnan(p):
        return "p = n/a"
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return "p < 0.01 **"
    if p < 0.05:
        return "p < 0.05 *"
    return f"p = {p:.3f} ns"


def _stat_label(arrays):
    """Return a short significance annotation for N arrays."""
    if len(arrays) == 2:
        _, p = _mannwhitney(arrays[0], arrays[1])
        return f"MW: {_pval_full(p)}"
    _, p = _kruskal(arrays)
    return f"KW: {_pval_full(p)}"


def _detect_side(stem):
    s = stem.upper()
    if re.search(r"(?<=[_\-])L(?=[_\-]|$)", s):
        return "L"
    if re.search(r"(?<=[_\-])R(?=[_\-]|$)", s):
        return "R"
    return None


def _detect_group(stem, pattern):
    """First capture group of `pattern` in stem, or full match if no group."""
    m = re.search(pattern, stem, re.IGNORECASE)
    if not m:
        return None
    return m.group(1) if m.lastindex else m.group(0)


# ── Plot panels ───────────────────────────────────────────────────────────────


def _get_colors(n):
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _rot_xlabels(ax, n, base_size=8):
    fs = max(6, base_size - max(0, n - 5))
    rot = 35 if n > 4 else 0
    ha = "right" if n > 4 else "center"
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rot, ha=ha, fontsize=fs)


def _plot_aggregate_bars(ax, agg_df, labels, colors, metric, ylabel, title):
    """
    One bar per condition — mean ± SEM across samples (not fibers).
    Individual sample dots overlaid so the doctor can see N and spread.
    Stats: Mann-Whitney (2 groups) or Kruskal-Wallis (N > 2).
    """
    vals_per_cond = []
    for lbl in labels:
        sub = agg_df[agg_df["_condition"] == lbl][metric].dropna().values
        vals_per_cond.append(sub)

    means = [v.mean() if len(v) else np.nan for v in vals_per_cond]
    sems = [v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in vals_per_cond]

    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        color=colors,
        alpha=0.72,
        width=0.55,
        yerr=sems,
        error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": _CLR["muted"]},
        zorder=2,
    )

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(vals_per_cond, colors, strict=False)):
        if len(vals):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                x[i] + jitter,
                vals,
                s=22,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
                alpha=0.9,
            )

    sig = _stat_label(vals_per_cond)
    ax.set_title(f"{title}\n{sig}", fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    _rot_xlabels(ax, len(labels))


def _plot_violin(ax, groups, colors, labels, metric, ylabel, title):
    """Fiber-level violin + subsampled jitter dots."""
    data = [df[metric].dropna().values for df in groups]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors, strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    parts["cmedians"].set_color(_CLR["text"])
    parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(42)
    for i, (d, color) in enumerate(zip(data, colors, strict=False)):
        d_plot = (
            rng.choice(d, size=min(len(d), _MAX_JITTER_PTS), replace=False)
            if len(d) > _MAX_JITTER_PTS
            else d
        )
        jitter = rng.uniform(-0.07, 0.07, size=len(d_plot))
        ax.scatter(
            np.full(len(d_plot), i + 1) + jitter,
            d_plot,
            s=3,
            color=color,
            alpha=0.2,
            linewidths=0,
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set(ylabel=ylabel, title=title)
    _rot_xlabels(ax, len(labels))

    sig = _stat_label(data)
    ax.text(
        0.98,
        0.98,
        sig,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=_CLR["muted"],
    )


def _plot_scatter_regression(ax, groups, colors, labels):
    # With many conditions, skip raw dots — only show regression lines
    show_dots = len(groups) <= 8
    rng = np.random.default_rng(42)
    for df, color, label in zip(groups, colors, labels, strict=False):
        x, y = df["axon_diam"].values, df["gratio"].values
        if show_dots:
            # Subsample dots for speed
            idx = rng.choice(len(x), size=min(len(x), 800), replace=False)
            ax.scatter(x[idx], y[idx], c=color, s=4, alpha=0.15, linewidths=0)
        res = _regression(x, y)
        if res:
            _, _, r, p, xr, yr = res
            ax.plot(xr, yr, color=color, lw=2.2, label=f"{label}  r={r:.2f} {_pval_stars(p)}")
    ax.set(xlabel="Axon diameter (µm)", ylabel="g-ratio", title="G-ratio vs Axon diameter")
    ax.set_ylim(0, 1)
    legend_fs = max(6, 9 - max(0, len(groups) - 6))
    ax.legend(loc="upper right", fontsize=legend_fs)


def _plot_summary_table(ax, groups, labels, colors, agg_df, nratios):
    """
    Per-condition summary:
      Condition | N images | N fibers | n-ratio | g-ratio | axon diam | myelin thick | density
    """
    ax.axis("off")

    def _agg_mean(lbl, col):
        sub = agg_df[agg_df["_condition"] == lbl][col].dropna()
        return f"{sub.mean():.3f}" if len(sub) else "—"

    def _fiber_stat(df, col, fmt):
        v = df[col].dropna()
        return f"{v.mean():{fmt}} ± {v.std():{fmt}}" if len(v) else "—"

    def _n_images(lbl):
        return str(len(agg_df[agg_df["_condition"] == lbl]))

    col_headers = [
        "Condition",
        "N images",
        "N fibers",
        "n-ratio",
        "g-ratio (mean±std)",
        "axon diam µm",
        "myelin thick µm",
        "density /mm²",
    ]

    rows = []
    for lbl, df in zip(labels, groups, strict=False):
        rows.append(
            [
                lbl,
                _n_images(lbl),
                str(len(df)),
                f"{nratios[lbl]:.3f}"
                if lbl in nratios and np.isfinite(nratios.get(lbl, np.nan))
                else _agg_mean(lbl, "nratio"),
                _fiber_stat(df, "gratio", ".3f"),
                _fiber_stat(df, "axon_diam", ".2f"),
                _fiber_stat(df, "myelin_thickness", ".2f"),
                _agg_mean(lbl, "axon_density_mm2"),
            ]
        )

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    fs = max(6, 9 - max(0, len(rows) - 8))
    tbl.set_fontsize(fs)
    row_h = max(1.2, 1.7 - max(0, len(rows) - 8) * 0.05)
    tbl.scale(1, row_h)

    # Header row styling
    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor(_CLR["header"])
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colour condition name cells
    for i, color in enumerate(colors):
        tbl[i + 1, 0].set_facecolor(color)
        tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")

    # Alternating row background
    for i in range(1, len(rows) + 1):
        bg = "#F0F4F8" if i % 2 == 0 else "#FFFFFF"
        for j in range(1, len(col_headers)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title(
        "Summary Statistics  (aggregate = mean across images  |  fiber stats = mean ± std)",
        pad=10,
        fontsize=10,
        fontweight="bold",
        color=_CLR["text"],
    )


# ── Dashboard entry point ──────────────────────────────────────────────────────


def make_comparison_dashboard(groups, labels, out_path, nratios=None, agg_df=None):
    n = len(groups)
    colors = _get_colors(n)
    nratios = nratios or {}
    if agg_df is None:
        agg_df = pd.DataFrame(columns=["_condition"])

    # Scale figure to number of conditions
    fig_w = max(18, min(32, 14 + n * 0.8))
    fig_h = max(22, min(36, 20 + n * 0.25))

    with plt.rc_context(_DASH_STYLE):
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            "Cross-sample Morphometry Comparison",
            fontsize=16,
            fontweight="bold",
            color=_CLR["text"],
            y=0.998,
        )

        gs = fig.add_gridspec(
            4,
            4,
            height_ratios=[1.5, 1.6, 1.9, 1.5],
            hspace=0.55,
            wspace=0.38,
            left=0.06,
            right=0.97,
            top=0.97,
            bottom=0.03,
        )

        # Row 0 — sample-level aggregate bars (4 key metrics)
        ax_nr = fig.add_subplot(gs[0, 0])
        ax_gr = fig.add_subplot(gs[0, 1])
        ax_den = fig.add_subplot(gs[0, 2])
        ax_are = fig.add_subplot(gs[0, 3])

        # Row 1 — fiber-level violins
        ax_vg = fig.add_subplot(gs[1, :2])
        ax_va = fig.add_subplot(gs[1, 2])
        ax_vm = fig.add_subplot(gs[1, 3])

        # Row 2 — scatter full width
        ax_sc = fig.add_subplot(gs[2, :])

        # Row 3 — summary table full width
        ax_tb = fig.add_subplot(gs[3, :])

        _plot_aggregate_bars(
            ax_nr, agg_df, labels, colors, "nratio", "n-ratio", "N-ratio  (AVF + MVF)"
        )
        _plot_aggregate_bars(
            ax_gr, agg_df, labels, colors, "gratio_aggr", "g-ratio", "Mean g-ratio"
        )
        _plot_aggregate_bars(
            ax_den, agg_df, labels, colors, "axon_density_mm2", "axons/mm²", "Axon density"
        )
        _plot_aggregate_bars(ax_are, agg_df, labels, colors, "nerve_area_mm2", "mm²", "Nerve area")

        _plot_violin(
            ax_vg,
            groups,
            colors,
            labels,
            "gratio",
            "g-ratio",
            "G-ratio distribution  (fiber level)",
        )
        _plot_violin(ax_va, groups, colors, labels, "axon_diam", "µm", "Axon diameter")
        _plot_violin(ax_vm, groups, colors, labels, "myelin_thickness", "µm", "Myelin thickness")

        _plot_scatter_regression(ax_sc, groups, colors, labels)
        _plot_summary_table(ax_tb, groups, labels, colors, agg_df, nratios)

        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"  → {out_path}")


def make_comparison_summary(groups, labels, out_path, nratios=None, agg_df=None):
    nratios = nratios or {}
    agg_df = agg_df if agg_df is not None else pd.DataFrame(columns=["_condition"])

    rows = []
    for lbl, df in zip(labels, groups, strict=False):
        sub_agg = agg_df[agg_df["_condition"] == lbl]
        rows.append(
            {
                "group": lbl,
                "n_images": len(sub_agg),
                "n_fibers": len(df),
                "nratio": nratios.get(lbl, float("nan")),
                "gratio_mean": df["gratio"].mean(),
                "gratio_std": df["gratio"].std(),
                "axon_diam_mean": df["axon_diam"].mean(),
                "axon_diam_std": df["axon_diam"].std(),
                "myelin_thickness_mean": df["myelin_thickness"].mean(),
                "myelin_thickness_std": df["myelin_thickness"].std(),
                "fiber_diam_mean": df["fiber_diam"].mean(),
                "fiber_diam_std": df["fiber_diam"].std(),
                "axon_density_mm2_mean": sub_agg["axon_density_mm2"].mean()
                if "axon_density_mm2" in sub_agg
                else float("nan"),
                "nerve_area_mm2_mean": sub_agg["nerve_area_mm2"].mean()
                if "nerve_area_mm2" in sub_agg
                else float("nan"),
            }
        )

    pooled = pd.concat(groups, ignore_index=True)
    rows.append(
        {
            "group": "POOLED (all)",
            "n_images": len(agg_df),
            "n_fibers": len(pooled),
            "nratio": float("nan"),
            "gratio_mean": pooled["gratio"].mean(),
            "gratio_std": pooled["gratio"].std(),
            "axon_diam_mean": pooled["axon_diam"].mean(),
            "axon_diam_std": pooled["axon_diam"].std(),
            "myelin_thickness_mean": pooled["myelin_thickness"].mean(),
            "myelin_thickness_std": pooled["myelin_thickness"].std(),
            "fiber_diam_mean": pooled["fiber_diam"].mean(),
            "fiber_diam_std": pooled["fiber_diam"].std(),
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
    parser.add_argument(
        "csvs",
        nargs="*",
        help="Paths to *_morphometrics.csv files (default: auto-discover under output/)",
    )
    parser.add_argument("--labels", nargs="+", help="Label per CSV (default: filename stem)")
    parser.add_argument("--out", default="output/comparison", help="Output directory")
    parser.add_argument(
        "--group-by",
        metavar="PATTERN",
        help=(
            "Pool samples into conditions. "
            "'LR' groups by L/R side detected in filename. "
            "Any other value is treated as a regex — the first capture group "
            "becomes the condition label (e.g. '(auto|allo)', '(BMSC|ADSC|OSC)')."
        ),
    )
    args = parser.parse_args()

    # ── Discover CSVs ────────────────────────────────────────────────────────
    if args.csvs:
        csvs = [pathlib.Path(p) for p in args.csvs]
    else:
        csvs = sorted(pathlib.Path("output").glob("**/*_morphometrics.csv"))
        if not csvs:
            print("No *_morphometrics.csv found under output/", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-discovered {len(csvs)} CSV(s) under output/")

    sample_labels = args.labels or [p.stem.replace("_morphometrics", "") for p in csvs]
    if len(sample_labels) != len(csvs):
        print("ERROR: --labels count must match number of CSV files", file=sys.stderr)
        sys.exit(1)

    # ── Load per-sample data ─────────────────────────────────────────────────
    print(f"\nLoading {len(csvs)} sample(s):")
    fiber_dfs = []
    agg_rows = []
    nratios_raw = {}

    for csv, label in zip(csvs, sample_labels, strict=False):
        df = pd.read_csv(csv)
        print(f"  [{label}]  {len(df)} fibers  ←  {csv}")
        fiber_dfs.append(df)

        agg_csv = pathlib.Path(str(csv).replace("_morphometrics.csv", "_aggregate.csv"))
        agg_row = {"_sample": label}
        if agg_csv.exists():
            agg = pd.read_csv(agg_csv)
            for col in [
                "nratio",
                "gratio_aggr",
                "avf",
                "mvf",
                "axon_density_mm2",
                "nerve_area_mm2",
            ]:
                if col in agg.columns:
                    agg_row[col] = float(agg[col].iloc[0])
            if "nratio" in agg_row:
                nratios_raw[label] = agg_row["nratio"]
        agg_rows.append(agg_row)

    agg_samples_df = pd.DataFrame(agg_rows)  # one row per sample

    # ── Group samples into conditions ────────────────────────────────────────
    if args.group_by == "LR":
        condition_map = {}
        unmatched = []
        for lbl in sample_labels:
            side = _detect_side(lbl)
            if side:
                condition_map[lbl] = side
            else:
                unmatched.append(lbl)
        if unmatched:
            print(f"WARNING: No L/R detected for: {', '.join(unmatched)}", file=sys.stderr)

    elif args.group_by:
        pattern = args.group_by
        condition_map = {}
        unmatched = []
        for lbl in sample_labels:
            grp = _detect_group(lbl, pattern)
            if grp:
                condition_map[lbl] = grp
            else:
                unmatched.append(lbl)
        if unmatched:
            print(
                f"WARNING: No match for pattern '{pattern}' in: {', '.join(unmatched)}",
                file=sys.stderr,
            )

    else:
        # No grouping — each sample is its own condition.
        # Guard: if too many samples to display individually, pool them all.
        if len(sample_labels) > 25:
            print(
                f"WARNING: {len(sample_labels)} samples with no --group-by. "
                "Pooling all into one group. Use --group-by PATTERN to compare conditions.",
                file=sys.stderr,
            )
            condition_map = dict.fromkeys(sample_labels, "All samples")
        else:
            condition_map = {lbl: lbl for lbl in sample_labels}

    # Build condition-level groups
    conditions = list(dict.fromkeys(condition_map.values()))  # preserve order, deduplicate
    groups, labels, nratios = [], [], {}
    agg_samples_df["_condition"] = agg_samples_df["_sample"].map(condition_map)

    for cond in conditions:
        idxs = [i for i, lbl in enumerate(sample_labels) if condition_map.get(lbl) == cond]
        merged_fibers = pd.concat([fiber_dfs[i] for i in idxs], ignore_index=True)
        groups.append(merged_fibers)
        labels.append(cond)

        cond_nratios = [
            nratios_raw[sample_labels[i]] for i in idxs if sample_labels[i] in nratios_raw
        ]
        if cond_nratios:
            nratios[cond] = float(np.mean(cond_nratios))

    print(
        f"\nConditions: {', '.join(f'{lbl} ({len(g)} fibers)' for lbl, g in zip(labels, groups, strict=False))}"
    )

    # ── Output ───────────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating comparison dashboard...")
    make_comparison_dashboard(
        groups,
        labels,
        out_dir / "comparison_dashboard.png",
        nratios=nratios,
        agg_df=agg_samples_df,
    )

    print("Generating summary CSV...")
    summary = make_comparison_summary(
        groups,
        labels,
        out_dir / "comparison_summary.csv",
        nratios=nratios,
        agg_df=agg_samples_df,
    )

    print("\n── Summary ─────────────────────────────────────────────────────")
    print(summary.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
