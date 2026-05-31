"""train/model.py — U-Net with ResNet34 encoder via segmentation-models-pytorch."""

from __future__ import annotations
import sys
import types
import torch.nn as nn

# Workaround: torchvision (a timm/smp dependency) imports `lzma` at module load
# time for its dataset utilities — this fails on pyenv Pythons built without xz.
# We only need torchvision for model weights, not for datasets, so a stub is safe.
# Fix: brew install xz && pyenv install 3.12.13 removes the need for this.
def _install_lzma_stub() -> None:
    stub = types.ModuleType("lzma")
    stub.open = None  # type: ignore[attr-defined]
    stub.LZMAFile = None  # type: ignore[attr-defined]
    stub.LZMAError = Exception  # type: ignore[attr-defined]
    stub.CHECK_NONE = 0  # type: ignore[attr-defined]
    sys.modules.setdefault("_lzma", stub)
    sys.modules.setdefault("lzma", stub)

try:
    import _lzma  # noqa: F401 — available → nothing to do
except ModuleNotFoundError:
    _install_lzma_stub()


def build_model(
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 3,
) -> nn.Module:
    """Return a U-Net with the given encoder.

    Requires: pip install segmentation-models-pytorch
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            "segmentation-models-pytorch is required for training. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from e

    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,  # raw logits — loss handles softmax internally
    )
