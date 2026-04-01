#!/usr/bin/env python3
"""
clean_output.py — Remove all pipeline outputs, keeping only manual fascicle masks.

Usage:
    python clean_output.py          # dry-run (shows what would be deleted)
    python clean_output.py --go     # actually delete
"""

import sys
from pathlib import Path

import config

KEEP_SUFFIXES = ("_fascicle_mask_edited.npy", "_raw.png")


def _keep(f: Path) -> bool:
    return any(f.name.endswith(s) for s in KEEP_SUFFIXES)


def main():
    dry_run = "--go" not in sys.argv
    if dry_run:
        print("DRY RUN — pass --go to actually delete\n")

    output = config.OUTPUT_DIR
    if not output.exists():
        print(f"Output dir '{output}' not found.")
        return

    to_delete = [f for f in output.rglob("*") if f.is_file() and not _keep(f)]

    if not to_delete:
        print("Nothing to delete.")
        return

    kept = [f for f in output.rglob("*") if f.is_file() and _keep(f)]
    print(f"Keeping  {len(kept):3d} file(s) (fascicle masks + raw cache)")
    print(f"Deleting {len(to_delete):3d} file(s)\n")

    for f in sorted(to_delete):
        print(f"  {'[DRY]' if dry_run else '[DEL]'} {f}")
        if not dry_run:
            f.unlink()

    if not dry_run:
        # Remove empty subdirectories
        for d in sorted(output.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        print(f"\nDone — {len(to_delete)} file(s) deleted.")
    else:
        print(f"\nRun with --go to delete these {len(to_delete)} file(s).")


if __name__ == "__main__":
    main()
