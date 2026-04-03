#!/usr/bin/env python3
"""app.py -- Validation & correction UI for nerve segmentations.

Launch:
    pip install fastapi uvicorn
    python app.py          # opens http://127.0.0.1:8000
"""

import contextlib
import json
import os
import secrets
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from skimage import io, measure

import config
from morphometrics import compute_aggregate, process_fibers
from qc import apply_qc
from segment import _remove_labels
from utils import build_fascicle_mask, clean_stem, find_low_qc_cluster_labels, find_satellite_labels
from visualization import (
    make_comparison,
    make_dashboard,
    make_gratio_map,
    make_numbered,
    make_overlay,
)

_AUTH_USER = "axon"
_AUTH_PASS = os.environ.get("APP_PASSWORD") or secrets.token_urlsafe(10)

_security = HTTPBasic()


def _check_auth(creds: HTTPBasicCredentials = Depends(_security)):  # noqa: B008
    ok_user = secrets.compare_digest(creds.username.encode(), _AUTH_USER.encode())
    ok_pass = secrets.compare_digest(creds.password.encode(), _AUTH_PASS.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


app = FastAPI(title="Nerve Segmentation Validator", dependencies=[Depends(_check_auth)])
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _stems() -> list[str]:
    """Processed stems only (have Cellpose cache)."""
    if not config.OUTPUT_DIR.exists():
        return []
    return sorted(
        d.name
        for d in config.OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_cellpose_outer.npy").exists()
    )


def _raw_image_paths() -> dict[str, Path]:
    """Map stem → raw image path for every image in INPUT_DIR."""
    result: dict[str, Path] = {}
    _EXTS = {".tif", ".tiff", ".png"}
    if not config.INPUT_DIR.exists():
        return result
    for p in sorted(config.INPUT_DIR.rglob("*")):
        if p.suffix.lower() in _EXTS and p.is_file():
            result[clean_stem(p)] = p
    return result


def _out(stem: str) -> Path:
    d = config.OUTPUT_DIR / stem
    if not d.exists():
        raise HTTPException(404, f"Image '{stem}' not found")
    return d


def _find_raw(stem: str) -> Path | None:
    for ext in (".tif", ".tiff", ".png"):
        for p in config.INPUT_DIR.rglob(f"*{ext}"):
            if clean_stem(p) == stem:
                return p
    return None


def _load_outer_base(stem: str) -> np.ndarray:
    """Load outer labels from Cellpose cache + erosion + satellite removal."""
    d = _out(stem)
    cache = d / f"{stem}_cellpose_outer.npy"
    old = d / f"{stem}_cellpose.npy"
    if not cache.exists() and old.exists():
        cache = old
    outer = np.load(str(cache))

    if config.OUTER_ERODE_PX > 0:
        from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

        nz_max = maximum_filter(outer, size=3)
        nz_min = minimum_filter(outer, size=3)
        border = (outer != 0) & (nz_max != nz_min)
        dist = distance_transform_edt(~border)
        outer = (outer * (dist > config.OUTER_ERODE_PX)).astype(outer.dtype)

    fascicle_edited = d / f"{stem}_fascicle_mask_edited.npy"
    if not fascicle_edited.exists():
        satellites = find_satellite_labels(outer, config.PIXEL_SIZE, config.CP_DIAM_UM)
        outer = _remove_labels(outer, satellites)

    return outer


def _load_outer(stem: str) -> np.ndarray:
    """Load outer labels — gt > edited > cellpose base.

    Returns the most complete label map available:
    - ``_outer_gt.npy``     ground truth (clinician-validated, includes additions)
    - ``_outer_edited.npy`` corrected prediction (deletions/modifications only)
    - Cellpose cache        raw model output + erosion + satellite removal
    """
    d = _out(stem)
    gt = d / f"{stem}_outer_gt.npy"
    if gt.exists():
        return np.load(str(gt))
    edited = d / f"{stem}_outer_edited.npy"
    if edited.exists():
        return np.load(str(edited))
    return _load_outer_base(stem)


def _load_inner(stem: str) -> np.ndarray:
    return np.load(str(_out(stem) / f"{stem}_axon_inner.npy"))


def _load_multicore(stem: str) -> set:
    p = _out(stem) / f"{stem}_multicore_labels.npy"
    return set(np.load(str(p)).tolist()) if p.exists() else set()


def _backup_inner(stem: str) -> None:
    d = _out(stem)
    bak = d / f"{stem}_axon_inner_original.npy"
    if not bak.exists():
        shutil.copy2(d / f"{stem}_axon_inner.npy", bak)


def _backup_outer(stem: str) -> None:
    """Ensure both _outer_edited.npy and _outer_gt.npy exist on first edit.

    - ``_outer_edited.npy`` — corrected prediction (no manual additions)
    - ``_outer_gt.npy``     — ground truth (includes manual additions)

    On first call, both are bootstrapped from the Cellpose base.  If only
    ``_outer_edited.npy`` exists (legacy data), ``_outer_gt.npy`` is copied
    from it so existing edits (including any legacy additions) are preserved.
    """
    d = _out(stem)
    edited = d / f"{stem}_outer_edited.npy"
    gt = d / f"{stem}_outer_gt.npy"
    if not edited.exists():
        base = _load_outer_base(stem)
        np.save(str(edited), base)
    if not gt.exists():
        shutil.copy2(str(edited), str(gt))


def _load_fascicle_mask(stem: str, outer: np.ndarray) -> np.ndarray:
    """Load fascicle mask — user-edited version if it exists, else auto-computed."""
    edited = _out(stem) / f"{stem}_fascicle_mask_edited.npy"
    if edited.exists():
        return np.load(str(edited))
    return build_fascicle_mask(outer, config.PIXEL_SIZE, config.CP_DIAM_UM)


def _load_exclusion_mask(stem: str, shape: tuple) -> np.ndarray | None:
    """Load exclusion mask if it exists and matches shape."""
    excl_path = config.OUTPUT_DIR / stem / f"{stem}_exclusion_mask.npy"
    if not excl_path.exists():
        return None
    mask = np.load(str(excl_path))
    return mask if mask.shape == shape else None


def _edits_path(stem: str) -> Path:
    return _out(stem) / f"{stem}_edits.json"


def _load_edits(stem: str) -> dict:
    p = _edits_path(stem)
    if p.exists():
        return json.loads(p.read_text())
    return {"deleted": [], "added": []}


def _save_edits(stem: str, edits: dict) -> None:
    _edits_path(stem).write_text(json.dumps(edits, indent=2))


def _fast_overlay(stem: str) -> bool:
    """Fast preview: colour masks without per-fiber QC (< 1 s)."""
    from skimage import morphology as morph
    from skimage.segmentation import find_boundaries

    from utils import to_rgb_uint8

    d = _out(stem)
    raw_path = _find_raw(stem)
    if raw_path is None:
        return False

    img = io.imread(str(raw_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    outer = _load_outer(stem)
    inner = _load_inner(stem)
    mc = _load_multicore(stem)

    rgb = to_rgb_uint8(img)
    ov = (rgb.astype(np.float32) * 0.4).astype(np.uint8)

    def blend(mask, color, alpha=0.6):
        c = np.array(color, dtype=np.float32)
        ov[mask] = np.clip(ov[mask].astype(np.float32) * (1 - alpha) + c * alpha, 0, 255).astype(
            np.uint8
        )

    mc_mask = np.isin(outer, list(mc)) if mc else np.zeros(outer.shape, bool)
    axon = (inner > 0) & ~mc_mask
    myelin = (outer > 0) & ~axon & ~mc_mask
    no_axon = (outer > 0) & (inner == 0) & ~mc_mask

    blend(no_axon, [220, 50, 50])
    blend(mc_mask & (outer > 0), [210, 50, 85])
    blend(myelin, [50, 50, 240])
    blend(axon, [0, 210, 60])

    # Contours
    outer_b = morph.binary_dilation(find_boundaries(outer, mode="thick"), morph.disk(1))
    ov[outer_b] = [70, 70, 220]
    inner_b = find_boundaries(inner, mode="thick") & (inner > 0)
    inner_b = morph.binary_dilation(inner_b, morph.disk(1))
    ov[inner_b] = [0, 240, 80]

    io.imsave(str(d / f"{stem}_overlay.png"), ov, check_contrast=False)
    return True


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/images")
def list_images():
    processed_set = set(_stems())
    raw_paths = _raw_image_paths()
    all_stems = sorted(processed_set | set(raw_paths.keys()))

    images = []
    for stem in all_stems:
        d = config.OUTPUT_DIR / stem
        processed = stem in processed_set
        modified = d.exists() and (
            (d / f"{stem}_axon_inner_original.npy").exists()
            or (d / f"{stem}_outer_edited.npy").exists()
            or (d / f"{stem}_outer_gt.npy").exists()
            or (d / f"{stem}_fascicle_mask_edited.npy").exists()
        )
        n_edits = 0
        n_axons = 0
        if processed and d.exists():
            edits = _load_edits(stem)
            n_edits = len(edits.get("deleted", [])) + len(edits.get("added", []))
            agg_path = d / f"{stem}_aggregate.csv"
            if agg_path.exists():
                agg = pd.read_csv(agg_path)
                if "n_axons" in agg.columns:
                    n_axons = int(agg["n_axons"].iloc[0])
                else:
                    morph = d / f"{stem}_morphometrics.csv"
                    if morph.exists():
                        n_axons = len(pd.read_csv(morph))
        # Needs recompute: edits pending since last morphometrics run
        edits_path = d / f"{stem}_edits.json"
        morph_path = d / f"{stem}_morphometrics.csv"
        needs_resegment = (
            processed
            and edits_path.exists()
            and n_edits > 0
            and (not morph_path.exists() or edits_path.stat().st_mtime > morph_path.stat().st_mtime)
        )
        images.append(
            {
                "stem": stem,
                "processed": processed,
                "modified": modified,
                "n_edits": n_edits,
                "n_axons": n_axons,
                "needs_resegment": needs_resegment,
            }
        )
    return images


# ── Serve images ─────────────────────────────────────────────────────────────


@app.get("/api/image/{stem}/overlay")
def get_overlay(stem: str):
    p = _out(stem) / f"{stem}_overlay.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.post("/api/image/{stem}/rebuild-overlay")
def rebuild_overlay(stem: str):
    """Regenerate overlay PNG after deferred (refresh=false) edits."""
    _fast_overlay(stem)
    return {"status": "ok"}


@app.get("/api/image/{stem}/numbered")
def get_numbered(stem: str):
    p = _out(stem) / f"{stem}_numbered.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/image/{stem}/dashboard")
def get_dashboard(stem: str):
    p = _out(stem) / f"{stem}_dashboard.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/image/{stem}/gratio_map")
def get_gratio_map(stem: str):
    p = _out(stem) / f"{stem}_gratio_map.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.post("/api/image/{stem}/clear-fascicle")
def clear_fascicle(stem: str):
    """Delete the edited fascicle mask."""
    mask_path = config.OUTPUT_DIR / stem / f"{stem}_fascicle_mask_edited.npy"
    if mask_path.exists():
        mask_path.unlink()
    return {"status": "ok"}


@app.get("/api/image/{stem}/fascicle")
def get_fascicle_contour(stem: str, t: int = 0):
    """Return the saved fascicle boundary as a downsampled polygon [[x,y],…]."""
    from skimage.measure import find_contours

    mask_path = config.OUTPUT_DIR / stem / f"{stem}_fascicle_mask_edited.npy"
    if not mask_path.exists():
        return {"points": None}

    mask = np.load(str(mask_path))
    contours = find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return {"contours": []}

    result = []
    for contour in sorted(contours, key=len, reverse=True):
        if len(contour) < 40:  # skip tiny edge artifacts
            continue
        step = max(1, len(contour) // 300)
        result.append([[int(c[1]), int(c[0])] for c in contour[::step]])
    return {"contours": result if result else []}


@app.get("/api/image/{stem}/raw")
def get_raw(stem: str):
    """Serve the raw input image as PNG (cached). Works for unprocessed images too."""
    from utils import to_rgb_uint8

    d = config.OUTPUT_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    png_cache = d / f"{stem}_raw.png"

    if not png_cache.exists():
        raw_path = _find_raw(stem)
        if raw_path is None:
            raise HTTPException(404, f"Raw image not found for '{stem}'")
        img = io.imread(str(raw_path))
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        io.imsave(str(png_cache), to_rgb_uint8(img), check_contrast=False)

    return FileResponse(str(png_cache), media_type="image/png")


# ── Image info ───────────────────────────────────────────────────────────────


@app.get("/api/image/{stem}/info")
def get_info(stem: str):
    d = _out(stem)
    inner = _load_inner(stem)
    edits = _load_edits(stem)

    axons = []
    for p in measure.regionprops(inner):
        if p.label == 0:
            continue
        cy, cx = int(p.centroid[0]), int(p.centroid[1])
        axons.append({"label": int(p.label), "x": cx, "y": cy})

    metrics = {}
    agg_path = d / f"{stem}_aggregate.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        metrics = {k: _safe(v) for k, v in agg.iloc[0].to_dict().items()}

    return {
        "stem": stem,
        "axons": axons,
        "edits": edits,
        "metrics": metrics,
        "has_backup": (d / f"{stem}_axon_inner_original.npy").exists(),
        "image_shape": list(inner.shape),
    }


def _safe(v):
    """Make numpy scalars JSON-serialisable."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ── Edit: identify / delete / add ───────────────────────────────────────────


class PointReq(BaseModel):
    x: int
    y: int


@app.post("/api/image/{stem}/identify")
def identify(stem: str, pt: PointReq):
    inner = _load_inner(stem)
    outer = _load_outer(stem)
    y, x = pt.y, pt.x

    if not (0 <= y < inner.shape[0] and 0 <= x < inner.shape[1]):
        return {"label": None}

    label = int(inner[y, x])
    if label > 0:
        ys, xs = np.where(inner == label)
        return {"label": label, "x": int(xs.mean()), "y": int(ys.mean())}

    fiber = int(outer[y, x])
    if fiber > 0 and (inner == fiber).any():
        ys, xs = np.where(inner == fiber)
        return {"label": fiber, "x": int(xs.mean()), "y": int(ys.mean())}

    return {"label": None}


@app.post("/api/image/{stem}/delete")
def delete_axon(stem: str, pt: PointReq, refresh: bool = True):
    _backup_inner(stem)
    _backup_outer(stem)
    d = _out(stem)

    inner = _load_inner(stem)
    gt = np.load(str(d / f"{stem}_outer_gt.npy"))
    y, x = pt.y, pt.x

    if not (0 <= y < inner.shape[0] and 0 <= x < inner.shape[1]):
        raise HTTPException(400, "Coordinates out of bounds")

    # Find label at click — try inner first, then gt (complete picture)
    label = int(inner[y, x])
    if label == 0:
        fiber = int(gt[y, x])
        if fiber > 0:
            label = fiber
    if label == 0:
        raise HTTPException(404, "No fiber found at this position")

    # Centroid before deletion
    ys, xs = np.where(gt == label)
    if len(ys) == 0:
        ys, xs = np.where(inner == label)
    cx, cy = int(xs.mean()), int(ys.mean())

    # Delete from inner + both outer files (edited & gt)
    inner[inner == label] = 0
    np.save(str(d / f"{stem}_axon_inner.npy"), inner)

    gt[gt == label] = 0
    np.save(str(d / f"{stem}_outer_gt.npy"), gt)

    edited = np.load(str(d / f"{stem}_outer_edited.npy"))
    edited[edited == label] = 0
    np.save(str(d / f"{stem}_outer_edited.npy"), edited)

    edits = _load_edits(stem)
    edits["deleted"].append({"label": int(label), "x": cx, "y": cy})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem) if refresh else False
    return {"deleted": label, "x": cx, "y": cy, "refreshed": refreshed}


class PolyReq(BaseModel):
    points: list[list[int]]


@app.post("/api/image/{stem}/add")
def add_axon(stem: str, poly: PolyReq, refresh: bool = True):
    from skimage.draw import polygon as draw_polygon

    inner = _load_inner(stem)
    outer = _load_outer(stem)

    if len(poly.points) < 3:
        raise HTTPException(400, "Need at least 3 points")

    xs = [p[0] for p in poly.points]
    ys = [p[1] for p in poly.points]
    rr, cc = draw_polygon(ys, xs, shape=inner.shape)
    if len(rr) == 0:
        raise HTTPException(400, "Polygon is empty or out of bounds")

    fiber_vals = outer[rr, cc]
    fiber_vals = fiber_vals[fiber_vals > 0]
    if len(fiber_vals) == 0:
        raise HTTPException(400, "Polygon is not inside any fiber")
    fiber_label = int(np.bincount(fiber_vals).argmax())

    # Only fill pixels belonging to that fiber
    mask = outer[rr, cc] == fiber_label
    rr, cc = rr[mask], cc[mask]

    _backup_inner(stem)
    d = _out(stem)
    inner[rr, cc] = fiber_label
    np.save(str(d / f"{stem}_axon_inner.npy"), inner)

    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    edits = _load_edits(stem)
    edits["added"].append({"label": int(fiber_label), "x": cx, "y": cy})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem) if refresh else False
    return {"added": fiber_label, "x": cx, "y": cy, "refreshed": refreshed}


class FiberReq(BaseModel):
    outer_points: list[list[int]]
    inner_points: list[list[int]]


@app.post("/api/image/{stem}/add-fiber")
def add_fiber(stem: str, req: FiberReq, refresh: bool = True):
    """Add a brand-new fiber: outer (myelin boundary) + inner (axon).

    The outer boundary is written to ``_outer_gt.npy`` only (ground truth)
    — NOT to ``_outer_edited.npy`` (corrected prediction).  This lets us
    distinguish model false-negatives from true-positives for fine-tuning.
    """
    from skimage.draw import polygon as draw_polygon

    if len(req.outer_points) < 3 or len(req.inner_points) < 3:
        raise HTTPException(400, "Both polygons need at least 3 points")

    _backup_inner(stem)
    _backup_outer(stem)
    d = _out(stem)

    gt = np.load(str(d / f"{stem}_outer_gt.npy"))
    edited = np.load(str(d / f"{stem}_outer_edited.npy"))
    inner = _load_inner(stem)

    # Rasterise outer polygon
    o_xs = [p[0] for p in req.outer_points]
    o_ys = [p[1] for p in req.outer_points]
    o_rr, o_cc = draw_polygon(o_ys, o_xs, shape=gt.shape)
    if len(o_rr) == 0:
        raise HTTPException(400, "Outer polygon is empty or out of bounds")

    # Rasterise inner polygon
    i_xs = [p[0] for p in req.inner_points]
    i_ys = [p[1] for p in req.inner_points]
    i_rr, i_cc = draw_polygon(i_ys, i_xs, shape=inner.shape)
    if len(i_rr) == 0:
        raise HTTPException(400, "Inner polygon is empty or out of bounds")

    # Label unique across both files
    new_label = max(int(edited.max()), int(gt.max()), int(inner.max())) + 1

    # Outer goes to GT only — this is a manual addition (model false-negative)
    gt[o_rr, o_cc] = new_label
    np.save(str(d / f"{stem}_outer_gt.npy"), gt)

    inner[i_rr, i_cc] = new_label
    np.save(str(d / f"{stem}_axon_inner.npy"), inner)

    cx, cy = int(np.mean(i_xs)), int(np.mean(i_ys))
    edits = _load_edits(stem)
    edits["added"].append({"label": int(new_label), "x": cx, "y": cy, "type": "fiber"})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem) if refresh else False
    return {"added": new_label, "x": cx, "y": cy, "refreshed": refreshed}


# ── Fascicle delineation ─────────────────────────────────────────────────────


class FascicleReq(BaseModel):
    points: list[list[int]]


@app.post("/api/image/{stem}/set-fascicle")
def set_fascicle(stem: str, req: FascicleReq):
    """Rasterise a user-drawn polygon → save as edited fascicle mask.
    Works for both processed and unprocessed (raw) images.
    """
    from skimage.draw import polygon as draw_polygon

    if len(req.points) < 3:
        raise HTTPException(400, "Need at least 3 points")

    # Create output dir for unprocessed images
    d = config.OUTPUT_DIR / stem
    d.mkdir(parents=True, exist_ok=True)

    # Resolve image shape: Cellpose cache if available, else raw image
    outer_cache = d / f"{stem}_cellpose_outer.npy"
    old_cache = d / f"{stem}_cellpose.npy"
    if outer_cache.exists():
        shape = np.load(str(outer_cache), mmap_mode="r").shape
    elif old_cache.exists():
        shape = np.load(str(old_cache), mmap_mode="r").shape
    else:
        raw_path = _find_raw(stem)
        if raw_path is None:
            raise HTTPException(404, f"Raw image not found for '{stem}'")
        img = io.imread(str(raw_path))
        shape = img.shape[:2]

    xs = [p[0] for p in req.points]
    ys = [p[1] for p in req.points]
    rr, cc = draw_polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    if len(rr) > 0:
        mask[rr, cc] = True

    # Union with existing mask (supports multi-fascicle nerves)
    existing = d / f"{stem}_fascicle_mask_edited.npy"
    if existing.exists():
        prev = np.load(str(existing))
        if prev.shape == mask.shape:
            mask = mask | prev

    np.save(str(d / f"{stem}_fascicle_mask_edited.npy"), mask)
    return {"status": "ok", "area_px": int(mask.sum())}


@app.post("/api/image/{stem}/paint-outer")
def paint_outer(stem: str, poly: PolyReq, refresh: bool = True):
    """Paint a new outer-fiber region from a freehand lasso polygon.

    Written to ``_outer_gt.npy`` only (manual addition = model false-negative).
    """
    from skimage.draw import polygon as draw_polygon

    if len(poly.points) < 3:
        raise HTTPException(400, "Need at least 3 points")

    _backup_outer(stem)
    d = _out(stem)

    gt = np.load(str(d / f"{stem}_outer_gt.npy"))
    edited = np.load(str(d / f"{stem}_outer_edited.npy"))
    inner = _load_inner(stem)

    xs = [p[0] for p in poly.points]
    ys = [p[1] for p in poly.points]
    rr, cc = draw_polygon(ys, xs, shape=gt.shape)
    if len(rr) == 0:
        raise HTTPException(400, "Polygon is empty or out of bounds")

    new_label = max(int(edited.max()), int(gt.max()), int(inner.max())) + 1

    gt[rr, cc] = new_label
    np.save(str(d / f"{stem}_outer_gt.npy"), gt)

    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    edits = _load_edits(stem)
    edits["added"].append({"label": int(new_label), "x": cx, "y": cy, "type": "outer"})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem) if refresh else False
    return {"added": new_label, "x": cx, "y": cy, "refreshed": refreshed}


@app.post("/api/image/{stem}/erase-outer")
def erase_outer(stem: str, poly: PolyReq, refresh: bool = True):
    """Zero out outer (and inner) mask within a freehand lasso polygon."""
    from skimage.draw import polygon as draw_polygon

    if len(poly.points) < 3:
        raise HTTPException(400, "Need at least 3 points")

    _backup_inner(stem)
    _backup_outer(stem)
    d = _out(stem)

    gt = np.load(str(d / f"{stem}_outer_gt.npy"))
    edited = np.load(str(d / f"{stem}_outer_edited.npy"))
    inner = _load_inner(stem)

    xs = [p[0] for p in poly.points]
    ys = [p[1] for p in poly.points]
    rr, cc = draw_polygon(ys, xs, shape=gt.shape)
    if len(rr) == 0:
        raise HTTPException(400, "Polygon is empty or out of bounds")

    gt[rr, cc] = 0
    edited[rr, cc] = 0
    inner[rr, cc] = 0
    np.save(str(d / f"{stem}_outer_gt.npy"), gt)
    np.save(str(d / f"{stem}_outer_edited.npy"), edited)
    np.save(str(d / f"{stem}_axon_inner.npy"), inner)

    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    edits = _load_edits(stem)
    edits["deleted"].append({"label": -1, "x": cx, "y": cy, "type": "erase"})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem) if refresh else False
    return {"erased": True, "x": cx, "y": cy, "refreshed": refreshed}


@app.post("/api/image/{stem}/set-exclusion")
def set_exclusion(stem: str, req: FascicleReq):
    """Rasterise polygon → save as exclusion zone mask (unioned with existing)."""
    from skimage.draw import polygon as draw_polygon

    if len(req.points) < 3:
        raise HTTPException(400, "Need at least 3 points")

    d = config.OUTPUT_DIR / stem
    d.mkdir(parents=True, exist_ok=True)

    outer_cache = d / f"{stem}_cellpose_outer.npy"
    old_cache = d / f"{stem}_cellpose.npy"
    if outer_cache.exists():
        shape = np.load(str(outer_cache), mmap_mode="r").shape
    elif old_cache.exists():
        shape = np.load(str(old_cache), mmap_mode="r").shape
    else:
        raw_path = _find_raw(stem)
        if raw_path is None:
            raise HTTPException(404, f"Raw image not found for '{stem}'")
        img = io.imread(str(raw_path))
        shape = img.shape[:2]

    xs = [p[0] for p in req.points]
    ys = [p[1] for p in req.points]
    rr, cc = draw_polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    if len(rr) > 0:
        mask[rr, cc] = True

    excl_path = d / f"{stem}_exclusion_mask.npy"
    if excl_path.exists():
        prev = np.load(str(excl_path))
        if prev.shape == mask.shape:
            mask = mask | prev

    np.save(str(excl_path), mask)
    return {"status": "ok", "area_px": int(mask.sum())}


@app.get("/api/image/{stem}/exclusion")
def get_exclusion(stem: str, t: int = 0):
    """Return saved exclusion zone as polygon contours [[x,y],…]."""
    from skimage.measure import find_contours

    excl_path = config.OUTPUT_DIR / stem / f"{stem}_exclusion_mask.npy"
    if not excl_path.exists():
        return {"contours": []}

    mask = np.load(str(excl_path))
    contours = find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return {"contours": []}

    result = []
    for contour in sorted(contours, key=len, reverse=True):
        if len(contour) < 10:
            continue
        step = max(1, len(contour) // 300)
        result.append([[int(c[1]), int(c[0])] for c in contour[::step]])
    return {"contours": result}


@app.post("/api/image/{stem}/clear-exclusion")
def clear_exclusion(stem: str):
    """Delete the exclusion mask."""
    excl_path = config.OUTPUT_DIR / stem / f"{stem}_exclusion_mask.npy"
    if excl_path.exists():
        excl_path.unlink()
    return {"status": "ok"}


# ── Recompute ────────────────────────────────────────────────────────────────


def _read_group_timepoint(stem: str) -> tuple[str, str]:
    """Recover group/timepoint from existing aggregate CSV, or derive from INPUT_DIR folder."""
    agg_path = config.OUTPUT_DIR / stem / f"{stem}_aggregate.csv"
    if agg_path.exists():
        row = pd.read_csv(agg_path).iloc[0]
        g = str(row.get("group", "")) if "group" in row else ""
        t = str(row.get("timepoint", "")) if "timepoint" in row else ""
        if g or t:
            return g, t
    # Derive from INPUT_DIR folder structure (same logic as segment.py _parse_folder)
    import re

    for p in config.INPUT_DIR.rglob("*"):
        if p.suffix.lower() in {".tif", ".tiff", ".png"} and clean_stem(p) == stem:
            parent = p.parent
            if parent != config.INPUT_DIR:
                name = parent.name
                m = re.search(r"(\d+w)\s*$", name.strip())
                if m:
                    return name.strip()[: m.start()].strip(), m.group(1)
            break
    return "", ""


@app.post("/api/image/{stem}/recompute")
def recompute(stem: str):

    d = _out(stem)
    outer = _load_outer(stem)
    inner = _load_inner(stem)
    multicore = _load_multicore(stem)

    # Remove fibers outside manual fascicle boundary (if drawn by user)
    fascicle_edited = d / f"{stem}_fascicle_mask_edited.npy"
    if fascicle_edited.exists():
        fm = np.load(str(fascicle_edited))
        outside = {
            p.label
            for p in measure.regionprops(outer)
            if not fm[
                min(int(p.centroid[0]), fm.shape[0] - 1),
                min(int(p.centroid[1]), fm.shape[1] - 1),
            ]
        }
        if outside:
            outer = _remove_labels(outer, outside)
            inner = _remove_labels(inner, outside)

    # Reconstruct axon_assignments from inner_labels
    fiber_bboxes = {p.label: p.bbox for p in measure.regionprops(outer)}
    axon_assignments = {}
    for lbl in np.unique(inner):
        if lbl == 0:
            continue
        bbox = fiber_bboxes.get(int(lbl))
        if bbox is None:
            continue
        minr, minc, maxr, maxc = bbox
        axon_assignments[int(lbl)] = (minr, minc, inner[minr:maxr, minc:maxc] == lbl)

    # Morphometrics + QC
    inner_new, pairs, df_all, index_image = process_fibers(
        outer, axon_assignments, config.PIXEL_SIZE
    )
    n_matched = len(pairs)
    df_pass, df_rej = apply_qc(df_all)

    # Remove low-QC clusters only when no manual fascicle mask (mirrors segment.py)
    if not fascicle_edited.exists():
        bad_cl = find_low_qc_cluster_labels(
            outer, df_pass, df_rej, config.PIXEL_SIZE, config.CP_DIAM_UM
        )
        if bad_cl:
            outer = _remove_labels(outer, bad_cl)
            inner = _remove_labels(inner, bad_cl)
            inner_new = _remove_labels(inner_new, bad_cl)
            df_pass = df_pass[~df_pass["_fiber_label"].isin(bad_cl)]
            df_rej = df_rej[~df_rej["_fiber_label"].isin(bad_cl)]

    fascicle_mask = _load_fascicle_mask(stem, outer)
    excl_mask = _load_exclusion_mask(stem, fascicle_mask.shape)
    group, timepoint = _read_group_timepoint(stem)

    agg = compute_aggregate(
        outer,
        df_pass,
        fascicle_mask,
        config.PIXEL_SIZE,
        group=group,
        timepoint=timepoint,
        excl_mask=excl_mask,
    )

    # Save CSVs
    pub_cols = [c for c in df_pass.columns if not c.startswith("_")]
    df_pass[pub_cols].to_csv(d / f"{stem}_morphometrics.csv", index=False)
    with contextlib.suppress(ImportError):
        df_pass[pub_cols].to_excel(d / f"{stem}_morphometrics.xlsx", index=False)
    pd.DataFrame([agg]).to_csv(d / f"{stem}_aggregate.csv", index=False)

    # Visualizations
    raw_path = _find_raw(stem)
    if raw_path is None:
        raise HTTPException(500, f"Raw image not found for '{stem}' in {config.INPUT_DIR}")
    img = io.imread(str(raw_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    n_outer = int(outer.max())
    overlay = make_overlay(
        img, outer, inner_new, df_pass, df_rej, multicore, fascicle_mask=fascicle_mask
    )
    io.imsave(str(d / f"{stem}_overlay.png"), overlay, check_contrast=False)

    numbered = make_numbered(
        overlay, df_pass, n_outer, stem, nerve_area_mm2=agg.get("nerve_area_mm2", 0.0)
    )
    io.imsave(str(d / f"{stem}_numbered.png"), numbered, check_contrast=False)

    make_gratio_map(img, df_pass, index_image, d / f"{stem}_gratio_map.png")
    make_dashboard(
        df_pass,
        df_rej,
        agg,
        n_outer,
        n_matched,
        stem,
        d / f"{stem}_dashboard.png",
        n_multicore=len(multicore),
    )

    # Clear pending edits
    _save_edits(stem, {"deleted": [], "added": []})

    return {"status": "ok", "n_axons": len(df_pass), "agg": {k: _safe(v) for k, v in agg.items()}}


@app.post("/api/image/{stem}/reset")
def reset_image(stem: str):
    d = _out(stem)
    bak_inner = d / f"{stem}_axon_inner_original.npy"
    edited_outer = d / f"{stem}_outer_edited.npy"
    gt_outer = d / f"{stem}_outer_gt.npy"
    fascicle_edited = d / f"{stem}_fascicle_mask_edited.npy"

    excl_path = d / f"{stem}_exclusion_mask.npy"
    if (
        not bak_inner.exists()
        and not edited_outer.exists()
        and not gt_outer.exists()
        and not fascicle_edited.exists()
        and not excl_path.exists()
    ):
        return {"status": "no_backup"}

    if bak_inner.exists():
        shutil.copy2(bak_inner, d / f"{stem}_axon_inner.npy")
        bak_inner.unlink()
    if edited_outer.exists():
        edited_outer.unlink()
    if gt_outer.exists():
        gt_outer.unlink()
    if fascicle_edited.exists():
        fascicle_edited.unlink()
    if excl_path.exists():
        excl_path.unlink()

    _save_edits(stem, {"deleted": [], "added": []})
    return {"status": "ok"}


# ── Background recompute-all ──────────────────────────────────────────────────

_batch_lock = threading.Lock()
_batch_status: dict = {"running": False, "done": 0, "total": 0, "current": "", "error": None}


def _batch_recompute_worker(stems: list[str]):
    """Run recompute for every image, then rebuild summary CSV."""

    global _batch_status
    _batch_status["done"] = 0
    _batch_status["total"] = len(stems)
    _batch_status["error"] = None

    results = []
    for i, stem in enumerate(stems):
        _batch_status["current"] = stem
        _batch_status["done"] = i
        try:
            d = _out(stem)
            outer = _load_outer(stem)
            inner = _load_inner(stem)
            multicore = _load_multicore(stem)

            # Remove fibers outside manual fascicle boundary
            fascicle_edited = d / f"{stem}_fascicle_mask_edited.npy"
            if fascicle_edited.exists():
                fm = np.load(str(fascicle_edited))
                outside = {
                    p.label
                    for p in measure.regionprops(outer)
                    if not fm[
                        min(int(p.centroid[0]), fm.shape[0] - 1),
                        min(int(p.centroid[1]), fm.shape[1] - 1),
                    ]
                }
                if outside:
                    outer = _remove_labels(outer, outside)
                    inner = _remove_labels(inner, outside)

            fiber_bboxes = {p.label: p.bbox for p in measure.regionprops(outer)}
            axon_assignments = {}
            for lbl in np.unique(inner):
                if lbl == 0:
                    continue
                bbox = fiber_bboxes.get(int(lbl))
                if bbox is None:
                    continue
                minr, minc, maxr, maxc = bbox
                axon_assignments[int(lbl)] = (minr, minc, inner[minr:maxr, minc:maxc] == lbl)

            inner_new, pairs, df_all, index_image = process_fibers(
                outer, axon_assignments, config.PIXEL_SIZE
            )
            n_matched = len(pairs)
            df_pass, df_rej = apply_qc(df_all)

            # Remove low-QC clusters only when no manual fascicle mask (mirrors segment.py)
            if not fascicle_edited.exists():
                bad_cl = find_low_qc_cluster_labels(
                    outer, df_pass, df_rej, config.PIXEL_SIZE, config.CP_DIAM_UM
                )
                if bad_cl:
                    outer = _remove_labels(outer, bad_cl)
                    inner_new = _remove_labels(inner_new, bad_cl)
                    df_pass = df_pass[~df_pass["_fiber_label"].isin(bad_cl)]
                    df_rej = df_rej[~df_rej["_fiber_label"].isin(bad_cl)]

            fascicle_mask = _load_fascicle_mask(stem, outer)
            excl_mask = _load_exclusion_mask(stem, fascicle_mask.shape)
            group, timepoint = _read_group_timepoint(stem)

            agg = compute_aggregate(
                outer,
                df_pass,
                fascicle_mask,
                config.PIXEL_SIZE,
                group=group,
                timepoint=timepoint,
                excl_mask=excl_mask,
            )

            pub_cols = [c for c in df_pass.columns if not c.startswith("_")]
            df_pass[pub_cols].to_csv(d / f"{stem}_morphometrics.csv", index=False)
            with contextlib.suppress(ImportError):
                df_pass[pub_cols].to_excel(d / f"{stem}_morphometrics.xlsx", index=False)
            pd.DataFrame([agg]).to_csv(d / f"{stem}_aggregate.csv", index=False)

            raw_path = _find_raw(stem)
            if raw_path is not None:
                img = io.imread(str(raw_path))
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                n_outer = int(outer.max())
                overlay = make_overlay(
                    img, outer, inner_new, df_pass, df_rej, multicore, fascicle_mask=fascicle_mask
                )
                io.imsave(str(d / f"{stem}_overlay.png"), overlay, check_contrast=False)
                numbered = make_numbered(
                    overlay,
                    df_pass,
                    n_outer,
                    stem,
                    nerve_area_mm2=agg.get("nerve_area_mm2", 0.0),
                )
                io.imsave(str(d / f"{stem}_numbered.png"), numbered, check_contrast=False)
                make_gratio_map(img, df_pass, index_image, d / f"{stem}_gratio_map.png")
                make_dashboard(
                    df_pass,
                    df_rej,
                    agg,
                    n_outer,
                    n_matched,
                    stem,
                    d / f"{stem}_dashboard.png",
                    n_multicore=len(multicore),
                )

            _save_edits(stem, {"deleted": [], "added": []})
            results.append({"image": stem, "n_axons": int(agg["n_axons"]), **agg})
        except Exception as exc:
            _batch_status["error"] = f"{stem}: {exc}"
            results.append({"image": stem, "error": str(exc)})

    # Rebuild global summary + comparison figure
    ok = [r for r in results if "error" not in r]
    if ok:
        pd.DataFrame(ok).to_csv(config.OUTPUT_DIR / "summary.csv", index=False)
        make_comparison(ok, config.OUTPUT_DIR / "comparison.png")

    _batch_status["done"] = len(stems)
    _batch_status["current"] = ""
    _batch_status["running"] = False


@app.post("/api/recompute-all")
def recompute_all():
    global _batch_status
    with _batch_lock:
        if _batch_status["running"]:
            return {"status": "already_running", **_batch_status}
        stems = _stems()
        if not stems:
            return {"status": "no_images"}
        _batch_status = {
            "running": True,
            "done": 0,
            "total": len(stems),
            "current": "",
            "error": None,
        }
        t = threading.Thread(target=_batch_recompute_worker, args=(stems,), daemon=True)
        t.start()
    return {"status": "started", "total": len(stems)}


@app.get("/api/recompute-all/status")
def recompute_all_status():
    return _batch_status


@app.get("/api/comparison")
def get_comparison():
    p = config.OUTPUT_DIR / "comparison.png"
    if not p.exists():
        raise HTTPException(404, "No comparison figure yet — run Recompute All first")
    return FileResponse(p, media_type="image/png")


@app.post("/api/recompute-summary")
def recompute_summary():
    """Quick summary rebuild from existing aggregate CSVs (no recompute)."""
    results = []
    for stem in _stems():
        d = config.OUTPUT_DIR / stem
        agg_path = d / f"{stem}_aggregate.csv"
        morph_path = d / f"{stem}_morphometrics.csv"
        if not agg_path.exists():
            continue
        agg = pd.read_csv(agg_path).iloc[0].to_dict()
        n = len(pd.read_csv(morph_path)) if morph_path.exists() else 0
        results.append({"image": stem, "n_axons": n, **agg})

    if results:
        pd.DataFrame(results).to_csv(config.OUTPUT_DIR / "summary.csv", index=False)
        make_comparison(results, config.OUTPUT_DIR / "comparison.png")
        return {"status": "ok", "n_images": len(results)}
    return {"status": "no_data"}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Nerve Segmentation Validator")
    print("  http://127.0.0.1:8000")
    print(f"\n  Login  →  user: {_AUTH_USER}  |  password: {_AUTH_PASS}\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
