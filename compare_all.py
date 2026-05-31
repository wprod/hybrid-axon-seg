"""compare_all.py — Analyse rigoureuse de la régénération nerveuse.

Design expérimental
-------------------
  Healthy (R) : nerf sain controlatéral (contrôle)
  alloX / alloA / alloB / alloO : allogreffe seule / + adipeux / + moelle / + muqueuse olfactive
  autoX / autoA / autoB / autoO : autogreffe seule / + adipeux / + moelle / + muqueuse olfactive
  Timepoints : 12w, 16w

Analyses produites
------------------
  1. Vue d'ensemble  — 9 groupes × métriques clés + tests vs sain (Bonferroni)
  2. Heatmap de récupération — % du sain par groupe × métrique
  3. Effet temporel — 12w vs 16w par groupe
  4. Stats complètes — CSV avec p-values, effect sizes, intervalles de confiance

Unité statistique : image (≈ animal) — pas la fibre (pseudo-réplication).
Tests non-paramétriques (Mann-Whitney) avec correction de Bonferroni.

Usage:
    python compare_all.py
"""

import pathlib
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT   = pathlib.Path("output")
OUT_DIR  = OUTPUT / "comparison" / "rigorous"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Design ────────────────────────────────────────────────────────────────────

GROUP_ORDER = [
    "Healthy",
    "alloX", "alloA", "alloB", "alloO",
    "autoX", "autoA", "autoB", "autoO",
]

CELL_FULL = {"X": "Greffe seule", "A": "+ adipeux", "B": "+ moelle osseuse", "O": "+ muqueuse olfactive"}

GROUP_COLORS = {
    "Healthy": "#27AE60",
    "alloX":   "#922B21", "alloA": "#C0392B", "alloB": "#E74C3C", "alloO": "#F1948A",
    "autoX":   "#1A5276", "autoA": "#1F618D", "autoB": "#2980B9", "autoO": "#85C1E9",
}

# Métriques image-level (agrégats) — unité de test statistique
METRICS_AGG = [
    ("gratio_area_weighted", "G-ratio",          "G-ratio (area-weighted)"),
    ("nratio",               "N-ratio",           "N-ratio (fibres / nerf)"),
    ("axon_density_mm2",     "Densité (/mm²)",    "Densité axonale"),
    ("avf",                  "AVF",               "Axon Volume Fraction"),
    ("mvf",                  "MVF",               "Myelin Volume Fraction"),
    ("nerve_area_mm2",       "Aire nerf (mm²)",   "Aire du nerf"),
]

# Métriques fibre-level (pour violins seulement)
METRICS_FIBER = [
    ("gratio",            "G-ratio",           "G-ratio"),
    ("axon_diam",         "Diam. axone (µm)",  "Diamètre axonal"),
    ("myelin_thickness",  "Myél. (µm)",        "Épaisseur myéline"),
]

_DASH = {
    "figure.facecolor":    "#F8F9FA",
    "axes.facecolor":      "#FFFFFF",
    "axes.edgecolor":      "#DEE2E6",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#E9ECEF",
    "grid.linewidth":      0.7,
    "axes.titlesize":      10,
    "axes.titleweight":    "bold",
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
}


# ── Parsing des stems ─────────────────────────────────────────────────────────

def parse_stem(stem: str) -> dict | None:
    """Extrait : graft_type, cells, timepoint, side depuis le nom de fichier."""
    m = re.match(r"^(allo|auto)([ABOX])(\d+w)\d*(L|R)$", stem, re.IGNORECASE)
    if not m:
        return None
    graft  = m.group(1).lower()
    cells  = m.group(2).upper()
    tp     = m.group(3).lower()
    side   = m.group(4).upper()
    healthy = side == "R"
    group  = "Healthy" if healthy else f"{graft}{cells}"
    return dict(stem=stem, graft=graft, cells=cells, timepoint=tp,
                side=side, is_healthy=healthy, group=group)


# ── Chargement des données ────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (agg_df, fiber_df) enrichis avec les métadonnées du groupe."""
    agg_rows, fiber_rows = [], []

    for agg_csv in sorted(OUTPUT.glob("**/*_aggregate.csv")):
        stem = agg_csv.parent.name
        meta = parse_stem(stem)
        if meta is None:
            continue
        try:
            row = pd.read_csv(agg_csv).iloc[0].to_dict()
            row.update(meta)
            agg_rows.append(row)
        except Exception:
            continue

        morph_csv = agg_csv.parent / f"{stem}_morphometrics.csv"
        if morph_csv.exists():
            try:
                df = pd.read_csv(morph_csv)
                df["stem"]      = stem
                df["group"]     = meta["group"]
                df["graft"]     = meta["graft"]
                df["cells"]     = meta["cells"]
                df["timepoint"] = meta["timepoint"]
                df["is_healthy"]= meta["is_healthy"]
                fiber_rows.append(df)
            except Exception:
                pass

    if not agg_rows:
        sys.exit("Aucune donnée trouvée — lancez d'abord segment.py ou Recompute.")

    agg_df   = pd.DataFrame(agg_rows)
    fiber_df = pd.concat(fiber_rows, ignore_index=True) if fiber_rows else pd.DataFrame()
    return agg_df, fiber_df


# ── Stats ─────────────────────────────────────────────────────────────────────

def mannwhitney(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)];    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)

def cohens_d(a, b):
    a = np.asarray(a, float)[np.isfinite(np.asarray(a, float))]
    b = np.asarray(b, float)[np.isfinite(np.asarray(b, float))]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0

def stars(p, corrected=True):
    if np.isnan(p): return "n/a"
    if p < (0.001 / (8 if corrected else 1)): return "***"
    if p < (0.01  / (8 if corrected else 1)): return "**"
    if p < (0.05  / (8 if corrected else 1)): return "*"
    return "ns"

def ci95(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 0.0
    return float(stats.sem(arr) * stats.t.ppf(0.975, len(arr) - 1))


# ── Figure 1 : Vue d'ensemble ─────────────────────────────────────────────────

def fig_overview(agg_df: pd.DataFrame) -> None:
    """Barplot image-level pour chaque groupe × métrique + étoiles vs Healthy."""
    healthy_df = agg_df[agg_df["is_healthy"]]
    graft_groups = [g for g in GROUP_ORDER if g != "Healthy"]
    n_metrics = len(METRICS_AGG)
    n_groups  = len(GROUP_ORDER)

    with plt.rc_context(_DASH):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Morphométrie nerveuse — Vue d'ensemble des 9 groupes\n"
            "(unité = image/animal, barres = moyenne ± IC 95%,  *p<0.05  **p<0.01  ***p<0.001  Bonferroni × 8)",
            fontsize=12, fontweight="bold", color="#2C3E50", y=1.01,
        )
        axes = axes.flatten()

        for ax_i, (col, ylabel, title) in enumerate(METRICS_AGG):
            ax = axes[ax_i]
            healthy_vals = healthy_df[col].dropna().values if col in healthy_df else np.array([])

            xs, means, cis, colors_bar = [], [], [], []
            for xi, grp in enumerate(GROUP_ORDER):
                sub = agg_df[agg_df["group"] == grp][col].dropna().values
                xs.append(xi)
                means.append(sub.mean() if len(sub) else np.nan)
                cis.append(ci95(sub))
                colors_bar.append(GROUP_COLORS.get(grp, "#888"))

            ax.bar(xs, means, color=colors_bar, alpha=0.82, width=0.7,
                   yerr=cis, error_kw={"elinewidth": 1.4, "capsize": 3, "ecolor": "#7F8C8D"},
                   zorder=2)

            # Points individuels
            rng = np.random.default_rng(42)
            for xi, grp in enumerate(GROUP_ORDER):
                sub = agg_df[agg_df["group"] == grp][col].dropna().values
                if len(sub):
                    jitter = rng.uniform(-0.18, 0.18, len(sub))
                    ax.scatter(xi + jitter, sub, s=18, color=GROUP_COLORS.get(grp, "#888"),
                               edgecolors="white", linewidths=0.5, zorder=5, alpha=0.9)

            # Étoiles vs Healthy (Bonferroni × 8 comparaisons)
            if len(healthy_vals) >= 2:
                ymax = ax.get_ylim()[1]
                for xi, grp in enumerate(graft_groups, start=1):
                    sub = agg_df[agg_df["group"] == grp][col].dropna().values
                    _, p = mannwhitney(healthy_vals, sub)
                    s = stars(p, corrected=True)
                    if s != "ns":
                        bar_top = means[xi] + cis[xi] if not np.isnan(means[xi]) else 0
                        ax.text(xi, bar_top + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03,
                                s, ha="center", va="bottom", fontsize=8,
                                color="#E74C3C" if s in ("***", "**") else "#E67E22")

            ax.set_xticks(xs)
            ax.set_xticklabels(GROUP_ORDER, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            # Ligne de référence = moyenne Healthy
            if len(healthy_vals) >= 1:
                ax.axhline(healthy_vals.mean(), ls="--", lw=1.2, color=GROUP_COLORS["Healthy"],
                           alpha=0.6, zorder=1, label=f"Sain = {healthy_vals.mean():.3f}")
                ax.legend(loc="upper right", fontsize=6)

        fig.tight_layout()
        out = OUT_DIR / "01_overview.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ── Figure 2 : Heatmap de récupération ───────────────────────────────────────

def fig_recovery_heatmap(agg_df: pd.DataFrame) -> None:
    """% de récupération vers le sain pour chaque groupe × métrique."""
    healthy_df = agg_df[agg_df["is_healthy"]]
    graft_groups = [g for g in GROUP_ORDER if g != "Healthy"]

    metrics_heat = [
        ("gratio_area_weighted", "G-ratio"),
        ("nratio",               "N-ratio"),
        ("axon_density_mm2",     "Densité"),
        ("avf",                  "AVF"),
        ("mvf",                  "MVF"),
        ("nerve_area_mm2",       "Aire nerf"),
    ]

    heat = np.full((len(graft_groups), len(metrics_heat)), np.nan)
    pmat = np.full_like(heat, np.nan)

    for j, (col, _) in enumerate(metrics_heat):
        h_vals = healthy_df[col].dropna().values if col in healthy_df else np.array([])
        h_mean = h_vals.mean() if len(h_vals) else np.nan
        for i, grp in enumerate(graft_groups):
            g_vals = agg_df[agg_df["group"] == grp][col].dropna().values
            if len(g_vals) and not np.isnan(h_mean) and h_mean != 0:
                heat[i, j] = g_vals.mean() / h_mean * 100
            _, p = mannwhitney(h_vals, g_vals)
            pmat[i, j] = p

    with plt.rc_context(_DASH):
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.suptitle(
            "Récupération vers le nerf sain  (% de la valeur contrôle)\n"
            "100% = identique au sain   |   * p<0.05  ** p<0.01  *** p<0.001 (Bonferroni×8 par colonne)",
            fontsize=11, fontweight="bold", color="#2C3E50",
        )

        from matplotlib.colors import TwoSlopeNorm
        vmin = max(50, np.nanmin(heat) - 5)
        vmax = min(150, np.nanmax(heat) + 5)
        norm = TwoSlopeNorm(vcenter=100, vmin=vmin, vmax=vmax)
        im = ax.imshow(heat, cmap="RdYlGn", norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, label="% du sain", shrink=0.8)

        col_labels = [m[1] for m in metrics_heat]
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(graft_groups)))
        ax.set_yticklabels(graft_groups, fontsize=9)

        # Valeur + étoile dans chaque cellule
        for i in range(len(graft_groups)):
            for j in range(len(metrics_heat)):
                v = heat[i, j]
                p = pmat[i, j]
                s = stars(p, corrected=True)
                txt = f"{v:.0f}%\n{s}" if not np.isnan(v) else "—"
                color = "white" if abs(v - 100) > 15 else "#2C3E50"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if s != "ns" else "normal")

        # Séparateur allo / auto
        ax.axhline(3.5, color="white", lw=2)
        ax.text(-0.6, 1.5, "ALLO", va="center", ha="right", fontsize=9,
                fontweight="bold", color=GROUP_COLORS["alloX"], rotation=90)
        ax.text(-0.6, 5.5, "AUTO", va="center", ha="right", fontsize=9,
                fontweight="bold", color=GROUP_COLORS["autoX"], rotation=90)

        fig.tight_layout()
        out = OUT_DIR / "02_recovery_heatmap.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ── Figure 3 : Effet temporel ─────────────────────────────────────────────────

def fig_timepoint(agg_df: pd.DataFrame) -> None:
    """12w vs 16w par groupe pour les métriques clés."""
    graft_groups = [g for g in GROUP_ORDER if g != "Healthy"]
    metrics_tp = [
        ("gratio_area_weighted", "G-ratio"),
        ("nratio",               "N-ratio"),
        ("axon_density_mm2",     "Densité (/mm²)"),
    ]

    with plt.rc_context(_DASH):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            "Évolution temporelle 12w → 16w par groupe  (MW non corrigé, comparaison intra-groupe)",
            fontsize=11, fontweight="bold", color="#2C3E50",
        )

        for ax, (col, title) in zip(axes, metrics_tp):
            w12_means, w16_means, w12_cis, w16_cis = [], [], [], []
            for grp in graft_groups:
                sub12 = agg_df[(agg_df["group"] == grp) & (agg_df["timepoint"] == "12w")][col].dropna().values
                sub16 = agg_df[(agg_df["group"] == grp) & (agg_df["timepoint"] == "16w")][col].dropna().values
                w12_means.append(sub12.mean() if len(sub12) else np.nan)
                w16_means.append(sub16.mean() if len(sub16) else np.nan)
                w12_cis.append(ci95(sub12))
                w16_cis.append(ci95(sub16))

            x = np.arange(len(graft_groups))
            w = 0.35
            b12 = ax.bar(x - w/2, w12_means, w, color=[GROUP_COLORS[g] for g in graft_groups],
                         alpha=0.55, label="12w", yerr=w12_cis,
                         error_kw={"elinewidth": 1.2, "capsize": 2, "ecolor": "#7F8C8D"})
            b16 = ax.bar(x + w/2, w16_means, w, color=[GROUP_COLORS[g] for g in graft_groups],
                         alpha=0.95, label="16w", yerr=w16_cis,
                         error_kw={"elinewidth": 1.2, "capsize": 2, "ecolor": "#7F8C8D"})

            # Étoiles 12w vs 16w
            for xi, grp in enumerate(graft_groups):
                sub12 = agg_df[(agg_df["group"] == grp) & (agg_df["timepoint"] == "12w")][col].dropna().values
                sub16 = agg_df[(agg_df["group"] == grp) & (agg_df["timepoint"] == "16w")][col].dropna().values
                _, p = mannwhitney(sub12, sub16)
                s = stars(p, corrected=False)
                if s != "ns":
                    ymax = max(
                        (w12_means[xi] or 0) + w12_cis[xi],
                        (w16_means[xi] or 0) + w16_cis[xi],
                    )
                    ax.text(xi, ymax + ax.get_ylim()[1] * 0.02, s, ha="center",
                            fontsize=8, color="#E74C3C")

            # Ligne sain
            h_mean = agg_df[agg_df["is_healthy"]][col].dropna().mean()
            if not np.isnan(h_mean):
                ax.axhline(h_mean, ls="--", lw=1.2, color=GROUP_COLORS["Healthy"],
                           alpha=0.7, label=f"Sain = {h_mean:.3f}")

            ax.set_xticks(x)
            ax.set_xticklabels(graft_groups, rotation=35, ha="right", fontsize=7)
            ax.set_title(title)
            ax.legend(fontsize=7)

        fig.tight_layout()
        out = OUT_DIR / "03_timepoint.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ── Figure 4 : Violins fibre-level ───────────────────────────────────────────

def fig_violins(fiber_df: pd.DataFrame) -> None:
    """Distributions au niveau fibre pour les 9 groupes."""
    if fiber_df.empty:
        return

    with plt.rc_context(_DASH):
        fig, axes = plt.subplots(1, 3, figsize=(22, 8))
        fig.suptitle(
            "Distributions fibre-level  (chaque point = 1 fibre, violin = densité)",
            fontsize=11, fontweight="bold", color="#2C3E50",
        )
        rng = np.random.default_rng(42)

        for ax, (col, ylabel, title) in zip(axes, METRICS_FIBER):
            data = [
                fiber_df[fiber_df["group"] == grp][col].dropna().values
                for grp in GROUP_ORDER
            ]
            parts = ax.violinplot(data, showmedians=True, showextrema=False)
            colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c); pc.set_alpha(0.5)
            parts["cmedians"].set_color("#2C3E50"); parts["cmedians"].set_linewidth(2)

            MAX_PTS = 600
            for xi, (grp, d) in enumerate(zip(GROUP_ORDER, data)):
                if not len(d): continue
                d_plot = rng.choice(d, size=min(len(d), MAX_PTS), replace=False)
                jitter = rng.uniform(-0.08, 0.08, len(d_plot))
                ax.scatter(xi + 1 + jitter, d_plot, s=2,
                           color=GROUP_COLORS[grp], alpha=0.15, linewidths=0)

            ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
            ax.set_xticklabels(GROUP_ORDER, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel(ylabel); ax.set_title(title)

        fig.tight_layout()
        out = OUT_DIR / "04_violins.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")


# ── CSV stats complet ─────────────────────────────────────────────────────────

def export_stats(agg_df: pd.DataFrame) -> None:
    """Table complète : chaque groupe × métrique — moyenne, IC95, p vs sain, d, % récup."""
    healthy_df = agg_df[agg_df["is_healthy"]]
    graft_groups = [g for g in GROUP_ORDER if g != "Healthy"]
    rows = []

    for col, ylabel, title in METRICS_AGG:
        h_vals = healthy_df[col].dropna().values if col in healthy_df else np.array([])
        h_mean = h_vals.mean() if len(h_vals) else np.nan

        # Healthy row
        rows.append(dict(
            metric=col, metric_label=title, group="Healthy",
            n=len(h_vals), mean=h_mean, sd=h_vals.std(ddof=1) if len(h_vals)>1 else np.nan,
            ci95_half=ci95(h_vals), recovery_pct=100.0,
            p_vs_healthy=np.nan, p_bonferroni=np.nan, cohens_d=np.nan, sig="ref",
        ))

        for grp in graft_groups:
            g_vals = agg_df[agg_df["group"] == grp][col].dropna().values
            _, p = mannwhitney(h_vals, g_vals)
            p_bonf = min(p * 8, 1.0) if not np.isnan(p) else np.nan
            d = cohens_d(h_vals, g_vals)
            recovery = g_vals.mean() / h_mean * 100 if len(g_vals) and not np.isnan(h_mean) and h_mean != 0 else np.nan
            rows.append(dict(
                metric=col, metric_label=title, group=grp,
                n=len(g_vals),
                mean=g_vals.mean() if len(g_vals) else np.nan,
                sd=g_vals.std(ddof=1) if len(g_vals)>1 else np.nan,
                ci95_half=ci95(g_vals),
                recovery_pct=recovery,
                p_vs_healthy=round(p, 4) if not np.isnan(p) else np.nan,
                p_bonferroni=round(p_bonf, 4) if not np.isnan(p_bonf) else np.nan,
                cohens_d=round(d, 3) if not np.isnan(d) else np.nan,
                sig=stars(p, corrected=True),
            ))

    df_stats = pd.DataFrame(rows)
    out = OUT_DIR / "stats_complets.csv"
    df_stats.to_csv(out, index=False, float_format="%.4f")
    print(f"  → {out}")

    # Résumé console
    print("\n── Récupération G-ratio vs sain ─────────────────────────────────────")
    sub = df_stats[df_stats["metric"] == "gratio_area_weighted"][
        ["group", "n", "mean", "ci95_half", "recovery_pct", "p_bonferroni", "sig"]
    ].copy()
    sub.columns = ["Groupe", "n img", "G-ratio moy", "±IC95", "% sain", "p (Bonf)", "sig"]
    print(sub.to_string(index=False, float_format="%.3f"))

    print("\n── Récupération N-ratio vs sain ─────────────────────────────────────")
    sub = df_stats[df_stats["metric"] == "nratio"][
        ["group", "n", "mean", "ci95_half", "recovery_pct", "p_bonferroni", "sig"]
    ].copy()
    sub.columns = ["Groupe", "n img", "N-ratio moy", "±IC95", "% sain", "p (Bonf)", "sig"]
    print(sub.to_string(index=False, float_format="%.3f"))


# ── Main ──────────────────────────────────────────────────────────────────────

def export_raw_excel(agg_df: pd.DataFrame) -> None:
    """Export une ligne par animal avec métadonnées — pour Prism ou analyses externes."""
    meta_cols = ["stem", "group", "graft", "cells", "timepoint", "side", "is_healthy"]
    data_cols = [c for c in agg_df.columns if c not in meta_cols]
    df_out = agg_df[meta_cols + data_cols].sort_values(["group", "timepoint", "stem"])
    out = OUTPUT / "morphometrics_all_animals.xlsx"
    df_out.to_excel(out, index=False)
    print(f"  → {out}  ({len(df_out)} animaux, {len(df_out.columns)} colonnes)")


def main():
    print("Chargement des données…")
    agg_df, fiber_df = load_data()

    groups_found = agg_df["group"].value_counts().to_dict()
    print(f"  Groupes détectés : {groups_found}")
    print(f"  Fibres totales   : {len(fiber_df):,}")

    n_healthy = (agg_df["is_healthy"]).sum()
    if n_healthy == 0:
        print("\nATTENTION : aucun contrôle sain (R) trouvé — vérifier les noms de fichiers.", file=sys.stderr)

    print("\nGénération des figures…")
    fig_overview(agg_df)
    fig_recovery_heatmap(agg_df)
    fig_timepoint(agg_df)
    fig_violins(fiber_df)
    export_stats(agg_df)
    export_raw_excel(agg_df)

    print(f"\n✓ Tout dans {OUT_DIR}/ + {OUTPUT}/morphometrics_all_animals.xlsx")


if __name__ == "__main__":
    main()
