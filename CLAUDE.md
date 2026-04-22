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

## Things to watch out for

- `numpy<2` constraint is intentional (Cellpose compatibility)
- Images are large microscopy TIFFs — all processing is done in-memory with numpy arrays
- `edited/` and `output/` are gitignored — never commit image data or results
- The web UI uses HTTP Basic Auth — password is random per session unless `APP_PASSWORD` is set
- No automated test suite — testing is manual via `test_one.py` and the web UI
