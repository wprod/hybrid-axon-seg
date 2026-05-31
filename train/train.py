"""train/train.py — Training loop for the axon/myelin U-Net.

Usage:
    python -m train.train
    python -m train.train --epochs 80 --batch 2
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import config as C
from train.dataset import GTDataset, annotated_stems
from train.losses import DiceCELoss
from train.model import build_model


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> list[float]:
    """Per-class IoU (ignores classes with no ground-truth pixels)."""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        true_c = targets == c
        inter = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


# ── Training loop ────────────────────────────────────────────────────────────

def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: str,
    use_amp: bool,
    num_classes: int,
) -> tuple[float, list[float]]:
    """Run one train or val epoch. Returns (mean_loss, per_class_iou)."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    all_iou: list[list[float]] = []

    with torch.set_grad_enabled(train_mode):
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            if train_mode and optimizer is not None:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                    optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_iou.append(compute_iou(preds.cpu(), masks.cpu(), num_classes))

    mean_loss = total_loss / len(loader)
    # Average IoU per class, ignoring NaN (class absent in batch)
    mean_iou = [
        float(np.nanmean([b[c] for b in all_iou])) for c in range(num_classes)
    ]
    return mean_loss, mean_iou


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = args.device or C.DEVICE
    use_amp = C.USE_AMP and device == "cuda"

    print(f"\n{'='*60}")
    print(f"  U-Net axon/myelin training")
    print(f"  device={device}  amp={use_amp}  epochs={args.epochs}  batch={args.batch}")
    print(f"{'='*60}")

    # ── Stems ─────────────────────────────────────────────────────────────
    stems = annotated_stems()
    if not stems:
        print("\nNo annotated stems found in ground_truth/masks/.")
        print("Clinician needs to annotate images in GT mode first.")
        return

    print(f"\nAnnotated stems ({len(stems)}): {stems}")

    random.shuffle(stems)
    n_val = int(len(stems) * C.VAL_SPLIT)
    val_stems = stems[:n_val]
    train_stems = stems[n_val:]
    print(f"Train: {train_stems}   Val: {val_stems if val_stems else '(none — all stems used for training)'}")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\nBuilding train dataset...")
    train_ds = GTDataset(train_stems, augment=True)
    print(f"  → {len(train_ds)} tiles")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == "cuda"))

    val_loader = None
    if val_stems:
        print("Building val dataset...")
        val_ds = GTDataset(val_stems, augment=False)
        print(f"  → {len(val_ds)} tiles")
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.workers, pin_memory=(device == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────
    ckpt: dict | None = None
    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.exists():
        print(f"\nResuming from {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        model = build_model(
            cfg.get("encoder", C.ENCODER),
            encoder_weights=None,  # don't re-download ImageNet weights
            num_classes=cfg.get("num_classes", C.NUM_CLASSES),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou = ckpt.get("val_miou", -1.0)
        print(f"  → epoch {start_epoch - 1}  val_mIoU={best_miou:.3f}")
    else:
        if args.resume:
            print(f"\nWarning: checkpoint '{args.resume}' not found — starting from scratch")
        model = build_model(C.ENCODER, C.ENCODER_WEIGHTS, C.NUM_CLASSES).to(device)
        start_epoch = 1
        best_miou = -1.0

    # ── Loss ──────────────────────────────────────────────────────────────
    w = torch.tensor(C.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = DiceCELoss(C.NUM_CLASSES, class_weights=w,
                           dice_weight=C.DICE_WEIGHT, ce_weight=C.CE_WEIGHT)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Load optimizer state BEFORE creating the scheduler so that initial_lr
    # is already set in param_groups when the scheduler initialises.
    if ckpt is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 2 if start_epoch > 1 else -1
    )

    if ckpt is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    # ── Checkpoint directory ──────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    class_names = ["bg", "myelin", "axon"]

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, C.NUM_CLASSES
        )
        scheduler.step()
        elapsed = time.time() - t0

        if val_loader is not None:
            val_loss, val_iou = run_epoch(
                model, val_loader, criterion, None, None, device, use_amp, C.NUM_CLASSES
            )
            miou = float(np.nanmean(val_iou[1:]))
            iou_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(class_names, val_iou))
            print(
                f"[{epoch:3d}/{args.epochs}]  "
                f"loss {train_loss:.4f} → {val_loss:.4f}  |  "
                f"{iou_str}  |  mIoU(fg)={miou:.3f}  |  {elapsed:.1f}s"
            )
        else:
            miou = float(np.nanmean(train_iou[1:]))  # use train mIoU as proxy
            iou_str = "  ".join(f"{n}={v:.3f}" for n, v in zip(class_names, train_iou))
            print(
                f"[{epoch:3d}/{args.epochs}]  "
                f"loss {train_loss:.4f}  |  "
                f"{iou_str}  |  train mIoU(fg)={miou:.3f}  |  {elapsed:.1f}s"
            )

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_miou": miou,
            "config": {
                "encoder": C.ENCODER,
                "num_classes": C.NUM_CLASSES,
                "tile_size": C.TILE_SIZE,
            },
        }, ckpt_dir / "last.pt")

        # Save best
        if miou > best_miou:
            best_miou = miou
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_miou": miou,
                "config": {
                    "encoder": C.ENCODER,
                    "num_classes": C.NUM_CLASSES,
                    "tile_size": C.TILE_SIZE,
                },
            }, ckpt_dir / "best.pt")
            print(f"  ★ new best  mIoU(fg)={miou:.3f}  saved → {ckpt_dir}/best.pt")

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_miou": miou,
                "config": {
                    "encoder": C.ENCODER,
                    "num_classes": C.NUM_CLASSES,
                    "tile_size": C.TILE_SIZE,
                },
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")
            print(f"  💾 checkpoint → {ckpt_dir}/epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Best val mIoU(fg): {best_miou:.3f}")
    print(f"Checkpoints in: {ckpt_dir.resolve()}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train axon/myelin U-Net")
    parser.add_argument("--epochs", type=int, default=C.NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LR)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/mps/cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(C.CHECKPOINT_DIR))
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint .pt to resume from (e.g. train/checkpoints/best.pt)")
    args = parser.parse_args()
    main(args)
