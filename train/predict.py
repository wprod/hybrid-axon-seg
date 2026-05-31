"""train/predict.py — Sliding-window inference on a full microscopy image.

Usage:
    # From a TIF path — outputs a semantic mask npy alongside the image:
    python -m train.predict --image ground_truth/images/sample.tif

    # From an existing raw.png (e.g. from the output/ directory):
    python -m train.predict --image output/GROUP/sample_raw.png --out /tmp/pred.npy

    # Programmatic use from the pipeline (returns numpy arrays):
    from train.predict import predict_masks
    axon_mask, fiber_mask = predict_masks(img_rgb_uint8, checkpoint="train/checkpoints/best.pt")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train import config as C
from train.model import build_model

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalise(img: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3) → float32 (H,W,3) normalised with ImageNet stats."""
    return (img.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD


def _load_checkpoint(checkpoint: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt.get("config", {})
    model = build_model(
        encoder=cfg.get("encoder", C.ENCODER),
        encoder_weights=None,  # inference — don't re-download ImageNet weights
        num_classes=cfg.get("num_classes", C.NUM_CLASSES),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


def predict_masks(
    img: np.ndarray,
    checkpoint: str = "train/checkpoints/best.pt",
    device: str | None = None,
    tile_size: int = C.TILE_SIZE,
    stride: int = C.TILE_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sliding-window inference and return binary masks.

    Args:
        img:        uint8 RGB image (H, W, 3).
        checkpoint: path to .pt checkpoint.
        device:     torch device string (default: auto-detect).
        tile_size:  must match the size used during training.
        stride:     sliding window stride (smaller = smoother, slower).

    Returns:
        axon_mask:  bool (H, W) — True where axon predicted.
        fiber_mask: bool (H, W) — True where fiber (myelin or axon) predicted.
    """
    dev = device or C.DEVICE
    model = _load_checkpoint(checkpoint, dev)

    H, W = img.shape[:2]
    num_classes = C.NUM_CLASSES

    prob_acc = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W), dtype=np.float32)

    norm_img = _normalise(img)

    # Build tile origins — ensure the last tile always reaches the image edge
    def _tile_origins(size: int, tile: int, step: int) -> list[int]:
        origins = list(range(0, max(size - tile + 1, 1), step))
        if not origins or origins[-1] + tile < size:
            origins.append(max(size - tile, 0))
        return origins

    with torch.no_grad():
        for y in _tile_origins(H, tile_size, stride):
            for x in _tile_origins(W, tile_size, stride):
                y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
                patch = norm_img[y:y2, x:x2]

                # Pad to tile_size if needed
                ph, pw = tile_size - patch.shape[0], tile_size - patch.shape[1]
                if ph > 0 or pw > 0:
                    patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)), mode="reflect")

                # (H,W,3) → (1,3,H,W)
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(dev)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # (C, tile_size, tile_size)

                # Unpad
                probs = probs[:, :y2 - y, :x2 - x]
                prob_acc[:, y:y2, x:x2] += probs
                weight_acc[y:y2, x:x2] += 1.0

    # Average overlapping predictions
    weight_acc = np.maximum(weight_acc, 1e-8)
    prob_acc /= weight_acc[np.newaxis]

    semantic = prob_acc.argmax(axis=0).astype(np.uint8)  # (H, W) values 0/1/2
    axon_mask = semantic == 2
    fiber_mask = semantic >= 1

    # Fill holes globally — handles simple hollow rings.
    from scipy.ndimage import binary_fill_holes
    axon_mask = binary_fill_holes(axon_mask)

    return axon_mask, fiber_mask


def semantic_to_instance_labels(
    axon_mask: np.ndarray,
    fiber_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert binary semantic masks to instance label arrays compatible with the pipeline.

    Uses watershed seeded by individual axon components to separate adjacent fibers
    whose myelin sheaths touch.  Each axon seed grows outward through the myelin,
    producing one fiber instance per axon.

    Args:
        axon_mask:  bool (H, W) — True where axon predicted.
        fiber_mask: bool (H, W) — True where fiber (myelin or axon) predicted.

    Returns:
        outer_labels: int32 (H, W) — each connected fiber region has a unique ID.
        inner_labels: int32 (H, W) — each connected axon region has its fiber's ID.
    """
    from skimage.measure import label as skimage_label
    from skimage.segmentation import watershed
    from scipy.ndimage import binary_fill_holes, distance_transform_edt

    # Label each connected axon component — these become the watershed seeds.
    axon_labels = skimage_label(axon_mask, connectivity=2)

    # Build the fiber region (myelin + axon, holes filled) where seeds will grow.
    fiber_filled = binary_fill_holes(fiber_mask | axon_mask)

    # Distance from background — used as the "elevation" map so the watershed
    # preferentially splits along thin myelin bridges between fibres.
    dist = distance_transform_edt(fiber_filled)

    # Watershed: each axon seed expands through the myelin, bounded by fiber_filled.
    outer_labels = watershed(-dist, markers=axon_labels, mask=fiber_filled).astype(np.int32)

    # Axon pixels keep the label of the fiber they belong to.
    inner_labels = np.zeros_like(outer_labels)
    inner_labels[axon_mask] = outer_labels[axon_mask]

    return outer_labels, inner_labels


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run axon/myelin prediction on an image")
    parser.add_argument("--image", required=True, help="Input image path (.tif or .png)")
    parser.add_argument("--checkpoint", default="train/checkpoints/best.pt")
    parser.add_argument("--out", default=None,
                        help="Output .npy path (default: <image_stem>_pred_semantic.npy)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    parser.add_argument("--stride", type=int, default=C.TILE_STRIDE)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    print(f"Loading image: {img_path}")
    raw = np.array(Image.open(img_path).convert("RGB"))

    print(f"Running inference (tile={args.tile_size}, stride={args.stride})…")
    axon_mask, fiber_mask = predict_masks(
        raw,
        checkpoint=args.checkpoint,
        device=args.device,
        tile_size=args.tile_size,
        stride=args.stride,
    )

    out_path = Path(args.out) if args.out else img_path.parent / f"{img_path.stem}_pred_semantic.npy"
    semantic = np.zeros(raw.shape[:2], dtype=np.uint8)
    semantic[fiber_mask] = 1
    semantic[axon_mask] = 2
    np.save(str(out_path), semantic)

    total_px = raw.shape[0] * raw.shape[1]
    print(f"Axon:   {axon_mask.sum() / total_px * 100:.1f}% of image")
    print(f"Myelin: {(fiber_mask & ~axon_mask).sum() / total_px * 100:.1f}% of image")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
