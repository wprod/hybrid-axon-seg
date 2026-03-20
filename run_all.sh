#!/usr/bin/env bash
# run_all.sh — Run the full pipeline on every image in edited/, then compare.
#
# Usage:
#   ./run_all.sh           # process all images + run comparison
#   ./run_all.sh --skip    # skip images that already have a Cellpose cache

set -e
PYTHON=axons/bin/python
EDITED=edited
SKIP=0

for arg in "$@"; do [[ "$arg" == "--skip" ]] && SKIP=1; done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  hybrid-axon-seg — batch run"
echo "════════════════════════════════════════════════════════"

# ── 1. Segment all images ──────────────────────────────────────────────────────
found=0
for f in "$EDITED"/*.tif; do
  [[ -f "$f" ]] || continue
  found=$((found + 1))
done

if [[ $found -eq 0 ]]; then
  echo "No .tif files found in $EDITED/"
  exit 1
fi

echo ""
echo "Found $found image(s) in $EDITED/"
echo ""

for f in "$EDITED"/*.tif; do
  [[ -f "$f" ]] || continue
  # Strip " (clean).tif" suffix to get the stem (same logic as clean_stem in utils.py)
  stem=$(basename "$f" | sed -E 's/ \(clean\)\.tiff?$//')

  if [[ $SKIP -eq 1 ]]; then
    cache="output/$stem/${stem}_cellpose_outer.npy"
    if [[ -f "$cache" ]]; then
      echo "⏭  Skipping: $stem  (cache exists)"
      continue
    fi
  fi

  echo "▶  Processing: $stem"
  $PYTHON segment.py "$f"
  echo ""
done

echo "════════════════════════════════════════════════════════"
echo "  All images processed."
echo "════════════════════════════════════════════════════════"
echo ""

# ── 2. Compare all morphometrics CSVs ─────────────────────────────────────────
csvs=()
labels=()
while IFS= read -r csv; do
  csvs+=("$csv")
  # Label = parent directory name (= sample stem)
  labels+=("$(basename "$(dirname "$csv")")")
done < <(find output -name "*_morphometrics.csv" | sort)

if [[ ${#csvs[@]} -lt 2 ]]; then
  echo "Need at least 2 processed samples to compare — skipping."
  exit 0
fi

echo "▶  Comparing ${#csvs[@]} samples..."
$PYTHON compare.py "${csvs[@]}" --labels "${labels[@]}" --out output/comparison/

echo ""
echo "✓  Done. Results in output/comparison/"
