# CLAUDE.md — hybrid-axon-seg

## What this project is

Medical image analysis pipeline for quantifying nerve morphometry from toluidine-blue stained
semi-thin cross-sections. Combines Cellpose (deep learning fiber detection) with classical image
processing (Otsu axon detection) and a FastAPI web UI for clinician review/correction.

## Language & conventions

- Python 3.11+, double quotes, 100-char line length
- Formatter/linter: **ruff** (pre-commit hooks via `.pre-commit-config.yaml`)
- Constants in `config.py` (single source of truth for all tunable parameters)
- Modules follow pipeline order: detection → preprocessing → morphometrics → qc → visualization
- `stem` = image filename without extension (used everywhere as unique key)
- Label arrays: `outer_labels` (fibers), `axon_labels` (axons inside fibers)
- DataFrames: `df_pass` / `df_rej` for QC-passed / rejected fibers
- `.npy` files are cached intermediate results (Cellpose output, axon masks, fascicle masks)
- Dual label maps: `*_outer_edited.npy` (clinician corrections) vs `*_outer_gt.npy` (ground truth with additions, for future fine-tuning)

## How to run

```bash
# Install
pip install -e .

# Process single image
python segment.py edited/GROUP/image.tif

# Batch process all images
./run_all.sh

# Start web validation UI (prints random password to terminal)
python app.py
# Or with fixed password:
APP_PASSWORD=mypassword python app.py
```

## Project structure

```
app.py              # FastAPI web validation UI (main entry for clinician review)
segment.py          # CLI entry — orchestrates full pipeline on one image
config.py           # All tunable parameters
detection.py        # Cellpose pass 1 (outer fibers) + Otsu axon detection
preprocessing.py    # Per-fiber normalized-inversion → axon_input image
morphometrics.py    # Per-fiber geometry (g-ratio, diameter, area fractions)
qc.py               # QC filtering with rejection reason codes
visualization.py    # Overlay, numbered, g-ratio map, dashboard
compare.py          # Cross-sample morphometry comparison dashboard
utils.py            # Shared helpers (image I/O, satellite detection)
static/             # Web UI (index.html, app.js, style.css)
edited/             # INPUT: original microscopy images (gitignored)
output/             # OUTPUT: all results — overlays, CSVs, .npy caches (gitignored)
```

## Key architecture details

- **Thread safety**: per-stem `threading.Lock` in `app.py` serializes concurrent edits
- **Cache invalidation**: Cellpose cache invalidated if fascicle mask is newer; axon cache
  invalidated when `_AXON_CACHE_VERSION` changes
- **Frontend**: vanilla JS canvas app with pan/zoom, 9 drawing modes, pointer events
  (mouse + Apple Pencil), undo/redo, real-time multi-user presence
- **QC rejection codes**: G (g-ratio), lgG (large+low g-ratio), shp (shape discordance),
  sol (solidity), off (centroid offset), Ø (diameter), brd (border)
- **Morphometrics**: area-weighted g-ratio = `sqrt(sum_axon_area / sum_fiber_area)`;
  N-ratio counts ALL fibers; AVF/MVF use QC-passed only

## CRITICAL — Ground truth data

**NEVER modify, overwrite, or delete any file under `ground_truth/`.**
This directory contains manually annotated reference data by the clinician (Marie).
It is the gold standard used for pipeline calibration and future model training.
Read-only in all circumstances — no exceptions.

## Training (U-Net)

- Machine: Mac Pro M4, 64 Go RAM unifiée, device=MPS
- MPS: CPU et GPU partagent la RAM mais sont des unités de calcul séparées — le GPU sature avant la RAM
- **Batch size**: 4–8 optimal sur MPS. Au-delà (16, 32) la RAM n'est pas le problème mais le GPU ne va pas plus vite → epochs plus lentes pour rien
- **num_workers**: 0 par défaut dans le config (compatibilité), passer `--workers 2` en CLI sur macOS
- ~354s/epoch avec batch 4, workers 0 sur M4 — 120 epochs ≈ 12h
- `compare_gt.py` : script de comparaison pipeline vs GT (pixel IoU + instance matching)
- **alloB12w10L** : annotation GT partiellement circulaire (85% des labels copiés du pipeline via restore) — garder en tête lors de l'évaluation

## Experimental design

Étude de régénération nerveuse périphérique chez le rat. Nommage des fichiers :
`{allo|auto}{A|B|O|X}{12w|16w}{N}{L|R}.tif`

**Type de greffe :**
- `allo` = allogreffe (nerf d'un donneur)
- `auto` = autogreffe (nerf du même animal — gold standard)

**Traitement cellulaire :**
- `X` = greffe seule, sans cellules
- `A` = + cellules adipeuses
- `B` = + cellules de la moelle osseuse
- `O` = + cellules de la muqueuse olfactive

**Timepoint :** `12w` = 12 semaines post-chirurgie, `16w` = 16 semaines

**Côté :** `L` = patte opérée (greffée), `R` = patte saine controlatérale (contrôle)

**Groupes :** 9 au total — Healthy (R) + 4 allo (X/A/B/O) + 4 auto (X/A/B/O), ~20 images/groupe

**Résultats clés :**
- G-ratio : entièrement récupéré dans tous les groupes (96–102% du sain, ns)
- N-ratio : partiellement récupéré — auto 63–68%, allo 54–58% (p<0.001 vs sain)
- La qualité de myélinisation est restaurée ; c'est le nombre d'axones qui repoussent qui limite la récupération
- Aucun effet cellulaire (A/B/O vs X) détecté sur les métriques morphométriques à 12–16w

**Scripts d'analyse :**
- `compare_all.py` : analyse complète rigoureuse — 4 figures + CSV stats (Mann-Whitney + Bonferroni)
- `compare_gt.py` : validation pipeline vs annotations GT (IoU pixel + instance matching)

## Things to watch out for

- `numpy<2` constraint is intentional (Cellpose compatibility)
- Images are large microscopy TIFFs — all processing is done in-memory with numpy arrays
- `edited/` and `output/` are gitignored — never commit image data or results
- The web UI uses HTTP Basic Auth — password is random per session unless `APP_PASSWORD` is set
- No automated test suite — testing is manual via `test_one.py` and the web UI
