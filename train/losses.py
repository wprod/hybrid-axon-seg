"""train/losses.py — Dice + CrossEntropy combined loss for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # sum over batch + spatial
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceCELoss(nn.Module):
    """Weighted combination of soft Dice loss and cross-entropy.

    Args:
        num_classes:   number of output classes
        class_weights: per-class weight tensor for CE (upweights rare classes)
        dice_weight:   contribution of Dice loss  (default 0.5)
        ce_weight:     contribution of CE loss     (default 0.5)
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + self.cw * self.ce(logits, targets)
