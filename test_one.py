#!/usr/bin/env python3
"""Test run on first image only."""
from pathlib import Path
from segment import process_image

if __name__ == "__main__":
    img = sorted(
        p for p in Path("edited").iterdir()
        if p.suffix.lower() in (".tif", ".tiff")
    )[0]

    print(f"Testing with: {img.name}")
    stem, n, agg = process_image(img)
    print(f"\nResult: {n} axons passed QC")
    print(f"Aggregate: {agg}")
