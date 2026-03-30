#!/usr/bin/env python3
"""app.py -- Validation & correction UI for nerve segmentations.

Launch:
    pip install fastapi uvicorn
    python app.py          # opens http://127.0.0.1:8000
"""

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from skimage import io, measure

import config
from morphometrics import process_fibers
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

app = FastAPI(title="Nerve Segmentation Validator")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _stems() -> list[str]:
    if not config.OUTPUT_DIR.exists():
        return []
    return sorted(
        d.name
        for d in config.OUTPUT_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_cellpose_outer.npy").exists()
    )


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


def _load_outer(stem: str) -> np.ndarray:
    """Load outer labels — user-edited version if it exists, else Cellpose + erosion."""
    d = _out(stem)

    # User-edited post-erosion labels take precedence
    edited = d / f"{stem}_outer_edited.npy"
    if edited.exists():
        return np.load(str(edited))

    # Original: Cellpose cache + erosion + cluster (mirrors segment.py)
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

    # Remove satellite fibers (low local neighbour density)
    satellites = find_satellite_labels(outer, config.PIXEL_SIZE, config.CP_DIAM_UM)
    outer = _remove_labels(outer, satellites)

    return outer


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
    """Save current outer labels as _outer_edited.npy if not already edited."""
    d = _out(stem)
    edited = d / f"{stem}_outer_edited.npy"
    if not edited.exists():
        outer = _load_outer(stem)  # applies erosion from cache
        np.save(str(edited), outer)


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
    images = []
    for stem in _stems():
        d = config.OUTPUT_DIR / stem
        modified = (d / f"{stem}_axon_inner_original.npy").exists() or (
            d / f"{stem}_outer_edited.npy"
        ).exists()
        edits = _load_edits(stem)
        n_edits = len(edits.get("deleted", [])) + len(edits.get("added", []))
        n_axons = 0
        agg_path = d / f"{stem}_aggregate.csv"
        if agg_path.exists():
            agg = pd.read_csv(agg_path)
            if "n_axons" in agg.columns:
                n_axons = int(agg["n_axons"].iloc[0])
            else:
                morph = d / f"{stem}_morphometrics.csv"
                if morph.exists():
                    n_axons = len(pd.read_csv(morph))
        images.append({"stem": stem, "modified": modified, "n_edits": n_edits, "n_axons": n_axons})
    return images


# ── Serve images ─────────────────────────────────────────────────────────────


@app.get("/api/image/{stem}/overlay")
def get_overlay(stem: str):
    p = _out(stem) / f"{stem}_overlay.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


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
def delete_axon(stem: str, pt: PointReq):
    inner = _load_inner(stem)
    outer = _load_outer(stem)
    y, x = pt.y, pt.x

    if not (0 <= y < inner.shape[0] and 0 <= x < inner.shape[1]):
        raise HTTPException(400, "Coordinates out of bounds")

    # Find label at click — try inner first, then outer
    label = int(inner[y, x])
    if label == 0:
        fiber = int(outer[y, x])
        if fiber > 0:
            label = fiber
    if label == 0:
        raise HTTPException(404, "No fiber found at this position")

    # Centroid before deletion (from outer since it's the larger region)
    ys, xs = np.where(outer == label)
    if len(ys) == 0:
        ys, xs = np.where(inner == label)
    cx, cy = int(xs.mean()), int(ys.mean())

    # Delete fiber from BOTH inner (axon) and outer (myelin)
    _backup_inner(stem)
    _backup_outer(stem)
    d = _out(stem)

    inner[inner == label] = 0
    np.save(str(d / f"{stem}_axon_inner.npy"), inner)

    outer[outer == label] = 0
    np.save(str(d / f"{stem}_outer_edited.npy"), outer)

    edits = _load_edits(stem)
    edits["deleted"].append({"label": int(label), "x": cx, "y": cy})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem)
    return {"deleted": label, "x": cx, "y": cy, "refreshed": refreshed}


class PolyReq(BaseModel):
    points: list[list[int]]


@app.post("/api/image/{stem}/add")
def add_axon(stem: str, poly: PolyReq):
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

    refreshed = _fast_overlay(stem)
    return {"added": fiber_label, "x": cx, "y": cy, "refreshed": refreshed}


class FiberReq(BaseModel):
    outer_points: list[list[int]]
    inner_points: list[list[int]]


@app.post("/api/image/{stem}/add-fiber")
def add_fiber(stem: str, req: FiberReq):
    """Add a brand-new fiber: outer (myelin boundary) + inner (axon)."""
    from skimage.draw import polygon as draw_polygon

    outer = _load_outer(stem)
    inner = _load_inner(stem)

    if len(req.outer_points) < 3 or len(req.inner_points) < 3:
        raise HTTPException(400, "Both polygons need at least 3 points")

    # Rasterise outer polygon
    o_xs = [p[0] for p in req.outer_points]
    o_ys = [p[1] for p in req.outer_points]
    o_rr, o_cc = draw_polygon(o_ys, o_xs, shape=outer.shape)
    if len(o_rr) == 0:
        raise HTTPException(400, "Outer polygon is empty or out of bounds")

    # Rasterise inner polygon
    i_xs = [p[0] for p in req.inner_points]
    i_ys = [p[1] for p in req.inner_points]
    i_rr, i_cc = draw_polygon(i_ys, i_xs, shape=inner.shape)
    if len(i_rr) == 0:
        raise HTTPException(400, "Inner polygon is empty or out of bounds")

    new_label = max(int(outer.max()), int(inner.max())) + 1

    # Backup inner (original .npy kept as safety net)
    _backup_inner(stem)

    d = _out(stem)
    outer[o_rr, o_cc] = new_label
    inner[i_rr, i_cc] = new_label

    # Save outer as post-erosion edited version (never touches cellpose cache)
    np.save(str(d / f"{stem}_outer_edited.npy"), outer)
    np.save(str(d / f"{stem}_axon_inner.npy"), inner)

    cx, cy = int(np.mean(i_xs)), int(np.mean(i_ys))
    edits = _load_edits(stem)
    edits["added"].append({"label": int(new_label), "x": cx, "y": cy, "type": "fiber"})
    _save_edits(stem, edits)

    refreshed = _fast_overlay(stem)
    return {"added": new_label, "x": cx, "y": cy, "refreshed": refreshed}


# ── Recompute ────────────────────────────────────────────────────────────────


@app.post("/api/image/{stem}/recompute")
def recompute(stem: str):

    d = _out(stem)
    outer = _load_outer(stem)
    inner = _load_inner(stem)
    multicore = _load_multicore(stem)

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
    inner_new, pairs, df_all, index_image, _ = process_fibers(
        outer, axon_assignments, config.PIXEL_SIZE
    )
    n_matched = len(pairs)
    df_pass, df_rej = apply_qc(df_all)

    # Remove low-QC clusters
    bad_cl = find_low_qc_cluster_labels(
        outer, df_pass, df_rej, config.PIXEL_SIZE, config.CP_DIAM_UM
    )
    if bad_cl:
        outer = _remove_labels(outer, bad_cl)
        inner = _remove_labels(inner, bad_cl)
        df_pass = df_pass[~df_pass["_fiber_label"].isin(bad_cl)]
        df_rej = df_rej[~df_rej["_fiber_label"].isin(bad_cl)]

    # Aggregate stats — build fascicle mask from cleaned labels
    fascicle_mask = build_fascicle_mask(outer, config.PIXEL_SIZE, config.CP_DIAM_UM)
    nerve_um2 = int(fascicle_mask.sum()) * config.PIXEL_SIZE**2

    if len(df_pass) and nerve_um2:
        total_axon_um2 = df_pass["axon_area"].sum()
        total_myelin_um2 = df_pass["fiber_area"].sum() - total_axon_um2
        avf = total_axon_um2 / nerve_um2
        mvf = total_myelin_um2 / nerve_um2
        agg = {
            "n_axons": len(df_pass),
            "nerve_area_mm2": nerve_um2 * 1e-6,
            "total_axon_area_mm2": total_axon_um2 * 1e-6,
            "total_myelin_area_mm2": total_myelin_um2 * 1e-6,
            "nratio": avf + mvf,
            "gratio_aggr": df_pass["gratio"].mean(),
            "avf": avf,
            "mvf": mvf,
            "axon_density_mm2": len(df_pass) / (nerve_um2 * 1e-6),
        }
    else:
        agg = {
            "n_axons": 0,
            "nerve_area_mm2": 0.0,
            "total_axon_area_mm2": 0.0,
            "total_myelin_area_mm2": 0.0,
            "nratio": 0.0,
            "gratio_aggr": 0.0,
            "avf": 0.0,
            "mvf": 0.0,
            "axon_density_mm2": 0.0,
        }

    # Save CSVs
    pub_cols = [c for c in df_pass.columns if not c.startswith("_")]
    df_pass[pub_cols].to_csv(d / f"{stem}_morphometrics.csv", index=False)
    try:
        df_pass[pub_cols].to_excel(d / f"{stem}_morphometrics.xlsx", index=False)
    except ImportError:
        pass
    pd.DataFrame([agg]).to_csv(d / f"{stem}_aggregate.csv", index=False)

    # Visualizations
    raw_path = _find_raw(stem)
    if raw_path is None:
        raise HTTPException(500, f"Raw image not found for '{stem}' in {config.INPUT_DIR}")
    img = io.imread(str(raw_path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    n_outer = int(outer.max())
    overlay = make_overlay(img, outer, inner_new, df_pass, df_rej, multicore)
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

    if not bak_inner.exists() and not edited_outer.exists():
        return {"status": "no_backup"}

    if bak_inner.exists():
        shutil.copy2(bak_inner, d / f"{stem}_axon_inner.npy")
        bak_inner.unlink()
    if edited_outer.exists():
        edited_outer.unlink()

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

            inner_new, pairs, df_all, index_image, _ = process_fibers(
                outer, axon_assignments, config.PIXEL_SIZE
            )
            n_matched = len(pairs)
            df_pass, df_rej = apply_qc(df_all)

            bad_cl = find_low_qc_cluster_labels(
                outer, df_pass, df_rej, config.PIXEL_SIZE, config.CP_DIAM_UM
            )
            if bad_cl:
                outer = _remove_labels(outer, bad_cl)
                df_pass = df_pass[~df_pass["_fiber_label"].isin(bad_cl)]
                df_rej = df_rej[~df_rej["_fiber_label"].isin(bad_cl)]

            fascicle_mask = build_fascicle_mask(outer, config.PIXEL_SIZE, config.CP_DIAM_UM)
            nerve_um2 = int(fascicle_mask.sum()) * config.PIXEL_SIZE**2

            if len(df_pass) and nerve_um2:
                total_axon_um2 = df_pass["axon_area"].sum()
                total_myelin_um2 = df_pass["fiber_area"].sum() - total_axon_um2
                avf = total_axon_um2 / nerve_um2
                mvf = total_myelin_um2 / nerve_um2
                agg = {
                    "n_axons": len(df_pass),
                    "nerve_area_mm2": nerve_um2 * 1e-6,
                    "total_axon_area_mm2": total_axon_um2 * 1e-6,
                    "total_myelin_area_mm2": total_myelin_um2 * 1e-6,
                    "nratio": avf + mvf,
                    "gratio_aggr": df_pass["gratio"].mean(),
                    "avf": avf,
                    "mvf": mvf,
                    "axon_density_mm2": len(df_pass) / (nerve_um2 * 1e-6),
                }
            else:
                agg = dict.fromkeys(
                    [
                        "n_axons",
                        "nerve_area_mm2",
                        "total_axon_area_mm2",
                        "total_myelin_area_mm2",
                        "nratio",
                        "gratio_aggr",
                        "avf",
                        "mvf",
                        "axon_density_mm2",
                    ],
                    0.0,
                )

            pub_cols = [c for c in df_pass.columns if not c.startswith("_")]
            df_pass[pub_cols].to_csv(d / f"{stem}_morphometrics.csv", index=False)
            try:
                df_pass[pub_cols].to_excel(d / f"{stem}_morphometrics.xlsx", index=False)
            except ImportError:
                pass
            pd.DataFrame([agg]).to_csv(d / f"{stem}_aggregate.csv", index=False)

            raw_path = _find_raw(stem)
            if raw_path is not None:
                img = io.imread(str(raw_path))
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                n_outer = int(outer.max())
                overlay = make_overlay(img, outer, inner_new, df_pass, df_rej, multicore)
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
    print("  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
