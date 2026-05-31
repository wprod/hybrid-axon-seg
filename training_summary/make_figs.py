"""make_figs.py — Generate all summary illustrations for training_summary/"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

OUT = Path(__file__).parent
DARK = "#1a1a2e"
BLUE = "#4f8ef7"
TEAL = "#00c9b1"
ORANGE = "#ff7043"
PINK = "#e040fb"
YELLOW = "#ffd740"
GREEN = "#69f0ae"
GRAY = "#78909c"
WHITE = "#f0f4f8"

plt.rcParams.update({
    "figure.facecolor": DARK,
    "axes.facecolor": DARK,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "axes.edgecolor": GRAY,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vue d'ensemble du pipeline d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle("Pipeline d'entraînement — vue d'ensemble", fontsize=16, fontweight="bold",
                 color=WHITE, y=0.98)

    steps = [
        ("Images GT\n(~10 images\ntoluidine blue)", TEAL, 1.0),
        ("Découpe\nen tuiles\n512×512", BLUE, 3.2),
        ("Augmentation\n(flip / rotation\n/ couleur / bruit)", ORANGE, 5.4),
        ("U-Net\nResNet34\n(ImageNet)", PINK, 7.6),
        ("Loss\nDice + CE\npondéré", YELLOW, 9.8),
        ("AdamW +\nCosine LR\n120 epochs", GREEN, 12.0),
        ("best.pt\n(meilleur\ncheckpoint)", TEAL, 14.2),
    ]

    for label, color, x in steps:
        box = FancyBboxPatch((x - 0.9, 1.2), 1.8, 2.6,
                             boxstyle="round,pad=0.15",
                             facecolor=color + "33", edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, 2.5, label, ha="center", va="center", fontsize=9.5,
                color=WHITE, fontweight="bold", linespacing=1.5)

    # Flèches
    arrow_xs = [(s[2] + 0.9, steps[i+1][2] - 0.9) for i, s in enumerate(steps[:-1])]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color=WHITE, lw=1.5))

    # Annotation nombre images
    ax.text(1.0, 0.55, "~10 images annotées\npar la clinicienne", ha="center",
            fontsize=8, color=TEAL, style="italic")
    ax.text(9.8, 0.55, "[bg=0.5 / myelin=1.0 / axon=2.5]", ha="center",
            fontsize=8, color=YELLOW, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "01_pipeline.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("01_pipeline.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transfer Learning — ImageNet → Microscopy
# ─────────────────────────────────────────────────────────────────────────────
def fig_transfer_learning():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.suptitle("Transfer Learning — ResNet34 pré-entraîné sur ImageNet", fontsize=15,
                 fontweight="bold", color=WHITE)

    # ImageNet side
    ax.add_patch(FancyBboxPatch((0.3, 0.8), 4.2, 4.4, boxstyle="round,pad=0.2",
                                facecolor="#1e3a5f", edgecolor=BLUE, linewidth=2))
    ax.text(2.4, 5.4, "ImageNet", ha="center", fontsize=13, color=BLUE, fontweight="bold")
    ax.text(2.4, 4.8, "1.2M images", ha="center", fontsize=9, color=GRAY)

    # Fake image grid
    imgs = [("chats 🐱", 0.6, 3.4), ("voitures 🚗", 2.2, 3.4), ("oiseaux 🐦", 3.8, 3.4),
            ("maisons 🏠", 0.6, 1.6), ("fleurs 🌸", 2.2, 1.6), ("arbres 🌳", 3.8, 1.6)]
    for label, x, y in imgs:
        ax.add_patch(FancyBboxPatch((x - 0.5, y - 0.5), 1.0, 0.9,
                                   boxstyle="round,pad=0.05",
                                   facecolor=BLUE + "22", edgecolor=BLUE + "66", linewidth=1))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color=WHITE)

    # Arrow
    ax.annotate("", xy=(8.5, 3.0), xytext=(5.2, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=3, mutation_scale=20))
    ax.text(6.85, 3.45, "Poids\nré-utilisés", ha="center", fontsize=9.5, color=TEAL,
            fontweight="bold")

    # Encoder box
    ax.add_patch(FancyBboxPatch((8.5, 0.8), 2.2, 4.4, boxstyle="round,pad=0.2",
                                facecolor="#3a1e5f", edgecolor=PINK, linewidth=2))
    ax.text(9.6, 5.35, "Encodeur", ha="center", fontsize=12, color=PINK, fontweight="bold")
    ax.text(9.6, 4.9, "ResNet34", ha="center", fontsize=10, color=GRAY)
    layers = ["Conv features\n(bords, textures)", "Formes\nlocales", "Structures\nplus grandes", "Représentation\nsémantique"]
    colors_l = [PINK + "66", PINK + "88", PINK + "aa", PINK + "cc"]
    for i, (lbl, col) in enumerate(zip(layers, colors_l)):
        y = 3.8 - i * 0.85
        ax.add_patch(FancyBboxPatch((8.65, y - 0.3), 1.9, 0.65,
                                   boxstyle="round,pad=0.05",
                                   facecolor=col, edgecolor=PINK, linewidth=1))
        ax.text(9.6, y + 0.02, lbl, ha="center", va="center", fontsize=7, color=WHITE)

    # Arrow to decoder
    ax.annotate("", xy=(12.5, 3.0), xytext=(11.2, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3, mutation_scale=20))
    ax.text(11.85, 3.45, "Fine-\ntuning", ha="center", fontsize=9.5, color=ORANGE,
            fontweight="bold")

    # Decoder box
    ax.add_patch(FancyBboxPatch((12.5, 0.8), 1.2, 4.4, boxstyle="round,pad=0.2",
                                facecolor="#1e4a1e", edgecolor=GREEN, linewidth=2))
    ax.text(13.1, 5.35, "Décodeur", ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(13.1, 3.0, "Skip\nconnections\n+ Upsample\n→ Masque\nsémantique", ha="center",
            va="center", fontsize=7.5, color=WHITE, linespacing=1.6)

    plt.tight_layout()
    plt.savefig(OUT / "02_transfer_learning.png", dpi=150, bbox_inches="tight",
                facecolor=DARK)
    plt.close()
    print("02_transfer_learning.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tiling — extraction des tuiles avec overlap
# ─────────────────────────────────────────────────────────────────────────────
def fig_tiling():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sliding Window — Découpe en tuiles avec chevauchement", fontsize=14,
                 fontweight="bold", color=WHITE)

    # Left: show tiles on a fake image
    ax = axes[0]
    ax.set_title("Extraction des tuiles (entraînement)", color=TEAL, fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Fake microscopy image background
    rng = np.random.default_rng(42)
    bg = rng.uniform(0.2, 0.35, (80, 100))
    ax.imshow(bg, extent=[0, 10, 0, 8], cmap="pink", alpha=0.5, aspect="auto")

    # Draw tiles (stride=1.28, tile=2.56 in figure units, i.e. 512/400 * scale)
    tile_w, tile_h = 2.5, 2.0
    stride_w, stride_h = 1.28, 1.02
    colors_t = [BLUE, ORANGE, TEAL, GREEN, PINK]
    ci = 0
    for row in range(3):
        for col in range(4):
            x = col * stride_w
            y = row * stride_h + 1.5
            if x + tile_w > 10 or y + tile_h > 8:
                continue
            c = colors_t[ci % len(colors_t)]
            ci += 1
            rect = plt.Rectangle((x, y), tile_w, tile_h,
                                  edgecolor=c, facecolor=c + "18", linewidth=1.5)
            ax.add_patch(rect)
    ax.text(5, 0.7, "stride = 128 px  (75% overlap)\ntuile = 512×512 px",
            ha="center", fontsize=9, color=GRAY, style="italic")
    ax.text(5, 0.1, "MIN_FG_FRAC = 5%  →  tuiles quasi-fond ignorées",
            ha="center", fontsize=8.5, color=ORANGE, style="italic")

    # Right: bar chart — tiles per image
    ax2 = axes[1]
    ax2.set_title("Impact du chevauchement sur le volume de données", color=TEAL, fontsize=11)
    n_images = np.arange(1, 11)
    tiles_no_overlap = n_images * 4
    tiles_with_overlap = n_images * 35  # approximate after filtering
    ax2.bar(n_images - 0.2, tiles_no_overlap, 0.35, color=ORANGE + "aa",
            edgecolor=ORANGE, label="Sans overlap (4 tuiles/image)")
    ax2.bar(n_images + 0.2, tiles_with_overlap, 0.35, color=TEAL + "aa",
            edgecolor=TEAL, label="Avec 75% overlap (~35 tuiles/image)")
    ax2.set_xlabel("Nombre d'images annotées", color=WHITE)
    ax2.set_ylabel("Nombre de tuiles", color=WHITE)
    ax2.legend(facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE, fontsize=8.5)
    ax2.tick_params(colors=WHITE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2, color=GRAY)
    ax2.set_facecolor(DARK)

    plt.tight_layout()
    plt.savefig(OUT / "03_tiling.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("03_tiling.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def _make_fake_cell(ax, x, y, r_out, r_in, color_out, color_in, alpha=0.9):
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.fill(x + r_out * np.cos(theta), y + r_out * np.sin(theta),
            color=color_out, alpha=alpha)
    ax.fill(x + r_in * np.cos(theta), y + r_in * np.sin(theta),
            color=color_in, alpha=alpha)


def _draw_scene(ax, transform_label, color, flip_h=False, flip_v=False,
                rotate=0, brightness=1.0, blur=False, distort=False, noise=False):
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#0d1117")
    ax.set_title(transform_label, color=color, fontsize=8.5, fontweight="bold", pad=3)

    rng = np.random.default_rng(7)
    cells = [(0.0, 0.0, 0.75, 0.38), (-0.9, 0.7, 0.55, 0.28),
             (0.9, -0.7, 0.6, 0.3), (-0.6, -0.9, 0.5, 0.25)]

    for cx, cy, ro, ri in cells:
        if flip_h:
            cx = -cx
        if flip_v:
            cy = -cy
        if rotate != 0:
            th = np.radians(rotate)
            cx, cy = cx * np.cos(th) - cy * np.sin(th), cx * np.sin(th) + cy * np.cos(th)
        if distort:
            cx += rng.uniform(-0.15, 0.15)
            cy += rng.uniform(-0.15, 0.15)
            ro *= rng.uniform(0.85, 1.15)
            ri *= rng.uniform(0.85, 1.15)

        alpha = min(0.9 * brightness, 1.0)
        c_out = "#6a3fa0" if not noise else "#8a5fc0"
        c_in = "#e8c4f0" if not noise else "#f0d8f8"
        if blur:
            for dr in np.linspace(0, 0.08, 4):
                _make_fake_cell(ax, cx, cy, ro + dr, ri + dr * 0.5, c_out, c_in,
                                alpha=alpha * (1 - dr * 5))
        else:
            _make_fake_cell(ax, cx, cy, ro, ri, c_out, c_in, alpha=alpha)

    if noise:
        npts = 300
        nx = rng.uniform(-2, 2, npts)
        ny = rng.uniform(-2, 2, npts)
        ax.scatter(nx, ny, s=0.5, color=WHITE, alpha=0.25)


def fig_augmentation():
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Data Augmentation — Multiplier artificiellement le dataset", fontsize=15,
                 fontweight="bold", color=WHITE, y=0.99)

    configs = [
        ("Original", TEAL, dict()),
        ("Flip\nhorizontal", BLUE, dict(flip_h=True)),
        ("Flip\nvertical", BLUE, dict(flip_v=True)),
        ("Rotation\n90°", ORANGE, dict(rotate=90)),
        ("Rotation\n20° + shift", ORANGE, dict(rotate=20)),
        ("Brightness\n+25%", YELLOW, dict(brightness=1.4)),
        ("Blur\nGaussien", PINK, dict(blur=True)),
        ("Elastic\ndistort", GREEN, dict(distort=True)),
        ("Bruit\nGaussien", "#ff9800", dict(noise=True)),
    ]
    # Also show HueSaturation as color shift on a separate subplot
    nrows, ncols = 2, 5
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx >= len(configs):
                ax = fig.add_subplot(nrows, ncols, idx + 1)
                ax.axis("off")
            else:
                ax = fig.add_subplot(nrows, ncols, idx + 1)
                axes.append((ax, configs[idx]))

    for ax, (label, color, kwargs) in axes:
        _draw_scene(ax, label, color, **kwargs)

    # Last cell: summary table of all transforms
    ax_last = fig.add_subplot(nrows, ncols, nrows * ncols)
    ax_last.axis("off")
    ax_last.set_facecolor(DARK)
    lines = [
        ("HFlip", "p=0.5", BLUE),
        ("VFlip", "p=0.5", BLUE),
        ("Rotate90", "p=0.75", ORANGE),
        ("ShiftScaleRot", "p=0.5", ORANGE),
        ("Brightness+Contrast", "p=0.7", YELLOW),
        ("HueSaturation", "p=0.5", YELLOW),
        ("CLAHE", "p=0.3", GREEN),
        ("GaussBlur", "p=0.2", PINK),
        ("GaussNoise", "p=0.3", "#ff9800"),
        ("ElasticTransform", "p=0.3", TEAL),
    ]
    ax_last.set_title("Toutes les transforms", color=WHITE, fontsize=8.5, fontweight="bold", pad=3)
    for k, (name, prob, col) in enumerate(lines):
        y_pos = 0.93 - k * 0.093
        ax_last.text(0.02, y_pos, f"• {name}", transform=ax_last.transAxes,
                     fontsize=7.5, color=col, va="top")
        ax_last.text(0.75, y_pos, prob, transform=ax_last.transAxes,
                     fontsize=7.5, color=GRAY, va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT / "04_augmentation.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("04_augmentation.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loss Function — Dice + CE avec poids de classe
# ─────────────────────────────────────────────────────────────────────────────
def fig_loss():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Fonction de perte — Dice + Cross-Entropy pondérée", fontsize=14,
                 fontweight="bold", color=WHITE)

    # Left: class imbalance visualization
    ax = axes[0]
    ax.set_title("Déséquilibre de classes dans les images", color=ORANGE, fontsize=11)
    classes = ["Fond\n(bg)", "Myéline", "Axone"]
    fracs = [72, 18, 10]
    colors_pie = [GRAY, BLUE, TEAL]
    wedges, texts, autotexts = ax.pie(
        fracs, labels=classes, colors=colors_pie, autopct="%1.0f%%",
        startangle=140, wedgeprops=dict(edgecolor=DARK, linewidth=2)
    )
    for t in texts:
        t.set_color(WHITE)
        t.set_fontsize(9)
    for at in autotexts:
        at.set_color(DARK)
        at.set_fontweight("bold")

    # Middle: class weights
    ax2 = axes[1]
    ax2.set_title("Poids de classe (CLASS_WEIGHTS)", color=YELLOW, fontsize=11)
    weights = [0.5, 1.0, 2.5]
    bars = ax2.bar(classes, weights, color=[GRAY + "bb", BLUE + "bb", TEAL + "bb"],
                   edgecolor=[GRAY, BLUE, TEAL], linewidth=2)
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"×{w}", ha="center", fontsize=12, color=YELLOW, fontweight="bold")
    ax2.set_ylabel("Poids relatif", color=WHITE)
    ax2.set_ylim(0, 3.2)
    ax2.set_facecolor(DARK)
    ax2.tick_params(colors=WHITE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)
    ax2.text(1, 2.8, "L'axone est rare\n→ poids ×5 vs fond", ha="center",
             fontsize=8.5, color=YELLOW, style="italic")

    # Right: loss formula visualization
    ax3 = axes[2]
    ax3.axis("off")
    ax3.set_title("Formule de la loss combinée", color=PINK, fontsize=11)

    ax3.text(0.5, 0.88, "L  =  0.5 × L_Dice  +  0.5 × L_CE", transform=ax3.transAxes,
             ha="center", fontsize=13, color=WHITE, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor=PINK + "22", edgecolor=PINK, linewidth=1.5))

    explanations = [
        (ORANGE, "L_Dice", "Overlap entre prédiction et GT\n"
          "→ Robuste au déséquilibre de classes\n"
          "→ Pénalise les faux négatifs"),
        (BLUE, "L_CE", "Cross-Entropy pondérée\n"
          "→ Probabilités calibrées\n"
          "→ Gradient stable en début"),
    ]
    for i, (col, name, desc) in enumerate(explanations):
        y0 = 0.62 - i * 0.38
        ax3.add_patch(FancyBboxPatch((0.05, y0 - 0.04), 0.9, 0.28,
                                    transform=ax3.transAxes,
                                    boxstyle="round,pad=0.02",
                                    facecolor=col + "18", edgecolor=col, linewidth=1.2))
        ax3.text(0.5, y0 + 0.2, name, transform=ax3.transAxes,
                 ha="center", fontsize=11, color=col, fontweight="bold")
        ax3.text(0.5, y0 + 0.06, desc, transform=ax3.transAxes,
                 ha="center", fontsize=8, color=WHITE, linespacing=1.5)

    plt.tight_layout()
    plt.savefig(OUT / "05_loss.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("05_loss.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Optimiseur + Cosine LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────
def fig_scheduler():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Optimisation — AdamW + Cosine Annealing LR", fontsize=14,
                 fontweight="bold", color=WHITE)

    # Left: LR curve
    ax = axes[0]
    ax.set_title("Learning rate schedule", color=TEAL, fontsize=11)
    epochs = np.arange(0, 121)
    lr0 = 3e-4
    lr = lr0 * 0.5 * (1 + np.cos(np.pi * epochs / 120))
    ax.plot(epochs, lr * 1e4, color=TEAL, linewidth=2.5)
    ax.fill_between(epochs, 0, lr * 1e4, color=TEAL, alpha=0.15)
    ax.set_xlabel("Epoch", color=WHITE)
    ax.set_ylabel("LR (×10⁻⁴)", color=WHITE)
    ax.set_xlim(0, 120)
    ax.set_facecolor(DARK)
    ax.tick_params(colors=WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, color=GRAY)
    ax.axhline(lr0 * 1e4, color=GRAY, linestyle="--", linewidth=1, label=f"LR₀ = {lr0:.0e}")
    ax.legend(facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE, fontsize=9)
    ax.annotate("Descend\ndoucement\nvers 0", xy=(110, lr[110] * 1e4 + 0.02),
                xytext=(80, 1.8), color=YELLOW, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=YELLOW))

    # Right: AdamW schema
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("AdamW — pourquoi ?", color=ORANGE, fontsize=11)
    props = [
        ("Adam", "Moments adaptatifs\n→ Convergence rapide\nquels que soient les gradients", BLUE),
        ("Weight Decay\n1×10⁻⁴", "Régularisation L2\n→ Évite l'overfitting\nsur petit dataset", ORANGE),
        ("Grad Clip\n= 1.0", "Norme max du gradient\n→ Stabilise l'entraînement\nen début", GREEN),
    ]
    for i, (name, desc, col) in enumerate(props):
        y0 = 0.82 - i * 0.33
        ax2.add_patch(FancyBboxPatch((0.05, y0 - 0.1), 0.9, 0.28,
                                    transform=ax2.transAxes,
                                    boxstyle="round,pad=0.02",
                                    facecolor=col + "22", edgecolor=col, linewidth=1.5))
        ax2.text(0.5, y0 + 0.13, name, transform=ax2.transAxes,
                 ha="center", fontsize=11, color=col, fontweight="bold")
        ax2.text(0.5, y0 + 0.00, desc, transform=ax2.transAxes,
                 ha="center", fontsize=8.5, color=WHITE, linespacing=1.5)

    plt.tight_layout()
    plt.savefig(OUT / "06_scheduler.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("06_scheduler.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Inférence sliding window + averaging des prédictions
# ─────────────────────────────────────────────────────────────────────────────
def fig_inference():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Inférence — Sliding Window avec moyennage des prédictions", fontsize=14,
                 fontweight="bold", color=WHITE)

    # Left: tile coverage heatmap
    ax = axes[0]
    ax.set_title("Couverture par pixel (coverage map)", color=TEAL, fontsize=11)
    H, W = 8, 10
    TILE, STRIDE = 2.5, 1.25  # in figure units
    coverage = np.zeros((H, W))
    xs = np.arange(0, W - TILE + 0.1, STRIDE)
    if xs[-1] + TILE < W:
        xs = np.append(xs, W - TILE)
    ys = np.arange(0, H - TILE + 0.1, STRIDE)
    if ys[-1] + TILE < H:
        ys = np.append(ys, H - TILE)
    scale_x = 10 / W
    scale_y = 8 / H
    for y in ys:
        for x in xs:
            y1, x1 = int(y), int(x)
            y2, x2 = min(int(y + TILE), H), min(int(x + TILE), W)
            coverage[y1:y2, x1:x2] += 1
    im = ax.imshow(coverage, cmap="YlGnBu", vmin=0)
    plt.colorbar(im, ax=ax, label="Nb. de tuiles couvrant ce pixel")
    ax.set_xlabel("x (pixels →)", color=WHITE)
    ax.set_ylabel("y (pixels ↓)", color=WHITE)
    ax.tick_params(colors=WHITE)
    ax.text(5, 9.2, "Pixels du centre: couverts par plusieurs tuiles\n"
            "→ La moyenne réduit le bruit de prédiction",
            ha="center", fontsize=8, color=GRAY, transform=ax.transData,
            clip_on=False)

    # Middle: averaging diagram
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Averaging des probabilités", color=ORANGE, fontsize=11)
    y_base = 0.75
    for i, (label, col) in enumerate([("Tile A\np(axon)=0.82", TEAL),
                                       ("Tile B\np(axon)=0.76", BLUE),
                                       ("Tile C\np(axon)=0.79", GREEN)]):
        y = y_base - i * 0.18
        ax2.add_patch(FancyBboxPatch((0.05, y - 0.07), 0.55, 0.13,
                                    transform=ax2.transAxes,
                                    boxstyle="round,pad=0.02",
                                    facecolor=col + "33", edgecolor=col, linewidth=1.2))
        ax2.text(0.32, y, label, transform=ax2.transAxes,
                 ha="center", va="center", fontsize=9, color=WHITE)

    ax2.annotate("", xy=(0.72, 0.57), xytext=(0.62, 0.57),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2, mutation_scale=15))

    ax2.add_patch(FancyBboxPatch((0.73, 0.47), 0.22, 0.18,
                                transform=ax2.transAxes,
                                boxstyle="round,pad=0.02",
                                facecolor=YELLOW + "33", edgecolor=YELLOW, linewidth=2))
    ax2.text(0.84, 0.57, "Moyenne\n0.79", transform=ax2.transAxes,
             ha="center", va="center", fontsize=10, color=YELLOW, fontweight="bold")

    ax2.text(0.5, 0.3, "→ argmax → classe: axone (2)", transform=ax2.transAxes,
             ha="center", fontsize=9.5, color=TEAL, fontweight="bold")
    ax2.text(0.5, 0.18, "prob_acc / weight_acc", transform=ax2.transAxes,
             ha="center", fontsize=8, color=GRAY, style="italic",
             bbox=dict(boxstyle="round", facecolor=DARK, edgecolor=GRAY))

    # Right: edge tile handling
    ax3 = axes[2]
    ax3.axis("off")
    ax3.set_title("Gestion des bords (bord atteint)", color=GREEN, fontsize=11)
    W_ex, H_ex = 10.0, 6.0
    TILE_ex, STRIDE_ex = 2.5, 1.5

    def tile_origins(size, tile, stride):
        origs = list(np.arange(0, max(size - tile, 0) + 0.01, stride))
        if not origs or origs[-1] + tile < size:
            origs.append(max(size - tile, 0))
        return origs

    xs_ex = tile_origins(W_ex, TILE_ex, STRIDE_ex)
    ys_ex = tile_origins(H_ex, TILE_ex, STRIDE_ex)

    rect_bg = plt.Rectangle((0.05, 0.1), 0.9, 0.85, transform=ax3.transAxes,
                             facecolor="#0d1117", edgecolor=GRAY, linewidth=1.5)
    ax3.add_patch(rect_bg)

    for j, x in enumerate(xs_ex):
        for i, y in enumerate(ys_ex):
            rx = 0.05 + x / W_ex * 0.9
            ry = 0.1 + y / H_ex * 0.85
            rw = min(TILE_ex, W_ex - x) / W_ex * 0.9
            rh = min(TILE_ex, H_ex - y) / H_ex * 0.85
            is_edge = (x + TILE_ex >= W_ex) or (y + TILE_ex >= H_ex)
            col = ORANGE if is_edge else BLUE
            r = plt.Rectangle((rx, ry), rw, rh, transform=ax3.transAxes,
                               edgecolor=col, facecolor=col + "20", linewidth=1.5)
            ax3.add_patch(r)

    ax3.add_patch(plt.Rectangle((0.95, 0.1), 0.0, 0.85, transform=ax3.transAxes,
                                edgecolor=YELLOW, facecolor="none", linewidth=3))
    ax3.text(0.5, 0.02, "Bord atteint garanti:\norigins.append(max(size - tile, 0))",
             transform=ax3.transAxes, ha="center", fontsize=8, color=YELLOW, style="italic")

    patch_blue = mpatches.Patch(color=BLUE, label="Tuiles normales")
    patch_orange = mpatches.Patch(color=ORANGE, label="Tuiles de bord")
    ax3.legend(handles=[patch_blue, patch_orange], loc="upper right",
               facecolor=DARK, edgecolor=GRAY, labelcolor=WHITE, fontsize=8,
               bbox_to_anchor=(1.0, 0.99), bbox_transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(OUT / "07_inference.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("07_inference.png ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Résumé — Pourquoi ça marche avec peu de données
# ─────────────────────────────────────────────────────────────────────────────
def fig_summary():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.suptitle("Résumé — Pourquoi ça marche avec si peu de données ?", fontsize=16,
                 fontweight="bold", color=WHITE, y=0.99)

    strategies = [
        # (x_center, y_center, width, height, title, bullets, color)
        (3.5, 7.2, 6.5, 1.6, "Transfer Learning (ImageNet → toluidine blue)", [
            "ResNet34 pré-entraîné reconnaît déjà: bords, textures, formes",
            "Seul le décodeur U-Net s'adapte au domaine",
            "→ Nécessite BEAUCOUP moins d'exemples",
        ], PINK),
        (10.5, 7.2, 6.5, 1.6, "Sliding Window (tile = 512×512, stride = 128)", [
            "1 image → ~35+ tuiles avec 75% d'overlap",
            "10 images annotées → ~350+ tuiles d'entraînement",
            "→ Multiplie le dataset ×35 sans nouvelles annotations",
        ], BLUE),
        (3.5, 4.8, 6.5, 1.8, "Data Augmentation (10 transforms)", [
            "Flips H+V, Rotate90, ShiftScaleRotate",
            "Brightness/Contrast, Hue/Saturation, CLAHE",
            "GaussBlur, GaussNoise, ElasticTransform",
            "→ Chaque tuile vue sous des dizaines de variantes",
        ], ORANGE),
        (10.5, 4.8, 6.5, 1.8, "Loss Dice + CE pondérée", [
            "Axone: poids ×2.5 vs fond (classe rare)",
            "Dice: robuste au déséquilibre de classes",
            "CE: gradient stable pour convergence",
            "→ Le réseau ne 'néglige' pas les axones",
        ], YELLOW),
        (3.5, 2.3, 6.5, 1.8, "Architecture U-Net", [
            "Skip connections → préserve les détails fins",
            "Encoder: features profondes (myéline/axone)",
            "Decoder: reconstruction à pleine résolution",
            "→ Adaptée aux structures annulaires microscopiques",
        ], TEAL),
        (10.5, 2.3, 6.5, 1.8, "Entraînement sans split val (VAL_SPLIT=0)", [
            "Toutes les annotations utilisées pour l'entraînement",
            "Validation = inspection visuelle par la clinicienne",
            "AdamW + Cosine LR: 120 epochs, gradient clip=1",
            "→ Maximise l'info extraite des rares annotations",
        ], GREEN),
    ]

    for x, y, w, h, title, bullets, color in strategies:
        bx = x - w / 2
        by = y - h / 2
        ax.add_patch(FancyBboxPatch((bx, by), w, h,
                                   boxstyle="round,pad=0.15",
                                   facecolor=color + "18", edgecolor=color,
                                   linewidth=2))
        ax.text(x, y + h / 2 - 0.25, title, ha="center", va="top",
                fontsize=9, color=color, fontweight="bold")
        for k, bullet in enumerate(bullets):
            ax.text(bx + 0.2, y + h / 2 - 0.55 - k * 0.33, f"• {bullet}",
                    ha="left", va="top", fontsize=7.8, color=WHITE)

    # Bottom label
    ax.text(7, 0.45, "Résultat : un modèle fonctionnel entraîné sur ~10 images annotées à la main",
            ha="center", fontsize=11, color=WHITE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK,
                      edgecolor=TEAL, linewidth=2))

    plt.tight_layout()
    plt.savefig(OUT / "00_summary.png", dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("00_summary.png ✓")


if __name__ == "__main__":
    print("Generating figures...")
    fig_summary()
    fig_pipeline()
    fig_transfer_learning()
    fig_tiling()
    fig_augmentation()
    fig_loss()
    fig_scheduler()
    fig_inference()
    print("\nAll done → training_summary/")
