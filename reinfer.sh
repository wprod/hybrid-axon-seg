#!/usr/bin/env bash
# reinfer.sh — Relance l'inférence U-Net sur les images avec l'artifact de tile boundary.
#
# Usage:
#   ./reinfer.sh            # détecte + reinfère toutes les images affectées
#   ./reinfer.sh --all      # reinfère TOUTES les images avec un cache existant
#   ./reinfer.sh --dry-run  # affiche la liste sans rien faire
#   ./reinfer.sh stem1 stem2 ...  # force la liste donnée

set -euo pipefail
PYTHON="${PYTHON:-python}"
DRY=0
ALL=0
FORCED_STEMS=()

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then DRY=1
  elif [[ "$arg" == "--all" ]]; then ALL=1
  else FORCED_STEMS+=("$arg")
  fi
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  reinfer.sh — U-Net tile-boundary fix"
echo "════════════════════════════════════════════════════════"
echo ""

# ── 1. Détecte les stems affectés ─────────────────────────────────────────────
if [[ ${#FORCED_STEMS[@]} -gt 0 ]]; then
  STEMS=("${FORCED_STEMS[@]}")
elif [[ $ALL -eq 1 ]]; then
  IFS=$'\n' read -r -d '' -a STEMS < <($PYTHON - <<'PYEOF'
import config
for d in sorted(config.OUTPUT_DIR.iterdir()):
    stem = d.name
    if (d / f"{stem}_cellpose_outer.npy").exists():
        print(stem)
PYEOF
printf '\0') || true
else
  IFS=$'\n' read -r -d '' -a STEMS < <($PYTHON - <<'PYEOF'
import numpy as np
from pathlib import Path
import config

TILE, STRIDE = 512, 256

def old_coverage(size):
    origins = list(range(0, max(size - TILE + 1, 1), STRIDE))
    return origins[-1] + TILE

for d in sorted(config.OUTPUT_DIR.iterdir()):
    stem = d.name
    cache = d / f"{stem}_cellpose_outer.npy"
    if not cache.exists():
        continue
    outer = np.load(str(cache), mmap_mode="r")
    H, W = outer.shape
    cx, cy = old_coverage(W), old_coverage(H)
    if (cx < W and outer[:, cx-1:cx+1].any()) or \
       (cy < H and outer[cy-1:cy+1, :].any()):
        print(stem)
PYEOF
printf '\0') || true
fi

N=${#STEMS[@]}
if [[ $N -eq 0 ]]; then
  echo "  Aucune image affectée trouvée."
  exit 0
fi

echo "  ${N} image(s) à retraiter"
[[ $DRY -eq 1 ]] && echo "  (dry-run — rien ne sera modifié)" && echo ""

for stem in "${STEMS[@]}"; do
  echo "  $stem"
done
echo ""

[[ $DRY -eq 1 ]] && exit 0

# ── 2. Relance l'inférence + reconstitue les caches ───────────────────────────
OK=0; FAIL=0
for stem in "${STEMS[@]}"; do
  echo "▶  $stem"
  $PYTHON - "$stem" <<'PYEOF'
import sys, shutil
import numpy as np
from pathlib import Path
from skimage import io
import config
from utils import clean_stem

stem = sys.argv[1]
d = config.OUTPUT_DIR / stem
exts = {".tif", ".tiff", ".png"}

# Find raw image
raw_path = None
for p in config.INPUT_DIR.rglob("*"):
    if p.suffix.lower() in exts and clean_stem(p) == stem:
        raw_path = p
        break
if raw_path is None:
    print(f"  ✗ image source introuvable pour {stem}", flush=True)
    sys.exit(1)

# Load image
img = io.imread(str(raw_path))
if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, :3]
if img.ndim == 2:
    img = np.stack([img] * 3, axis=-1)

print(f"  image {img.shape[1]}×{img.shape[0]}", flush=True)

# Run inference
from train.predict import predict_masks, semantic_to_instance_labels
axon_mask, fiber_mask = predict_masks(img)
outer_labels, inner_labels = semantic_to_instance_labels(axon_mask, fiber_mask)

print(f"  → {outer_labels.max()} fibres détectées", flush=True)

# Save instance label caches
d.mkdir(parents=True, exist_ok=True)
np.save(str(d / f"{stem}_cellpose_outer.npy"), outer_labels)
np.save(str(d / f"{stem}_axon_inner.npy"),     inner_labels)

# Reset multicore (always empty for U-Net)
np.save(str(d / f"{stem}_multicore_labels.npy"), np.array([], dtype=np.int32))

# Remove derived caches so they get regenerated on next Recompute
for suffix in ("overlay.png", "numbered.png", "dashboard.png",
               "gratio_map.png", "morphometrics.csv", "morphometrics.xlsx",
               "aggregate.csv", "fascicle_mask.npy",
               "outer_edited.npy", "outer_gt.npy",
               "axon_inner_original.npy", "edits.json"):
    f = d / f"{stem}_{suffix}"
    if f.exists():
        f.unlink()

print(f"  ✓ caches mis à jour", flush=True)
PYEOF
  # shellcheck disable=SC2181
  if [[ $? -eq 0 ]]; then
    OK=$((OK + 1))
  else
    FAIL=$((FAIL + 1))
    echo "  ✗ échec — on continue"
  fi
  echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  Terminé — ${OK} OK / ${FAIL} échec(s)"
echo ""
echo "  Lance ↻ Recompute (ou ↻ All) dans l'app pour"
echo "  régénérer les morphométries et les overlays."
echo "════════════════════════════════════════════════════════"
echo ""
