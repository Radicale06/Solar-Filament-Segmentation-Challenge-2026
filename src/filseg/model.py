"""U-Net++ with a timm encoder and four 1-channel heads sharing one decoder.

Heads (fixed order = output channels):
    mask      consensus filament probability
    boundary  3 px boundary band probability
    spine     spine/centerline heatmap
    fraction  predicted annotator fraction (existence-confidence signal)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .constants import HEADS


class FilSegModel(nn.Module):
    """NOTE: the encoder must expose a full 5-level pyramid (strides 2/4/8/16/32).
    ConvNeXt-style encoders (no stride-2 stage) make smp inject a zero-channel
    dummy skip at stride 2, silently removing all fine detail from the decoder —
    fatal here given the measured ±1 px boundary budget. timm-efficientnet-* and
    resnet-family encoders are known-good."""

    def __init__(self, encoder: str = "timm-efficientnet-b4", weights: str | None = "imagenet",
                 in_channels: int = 3):
        super().__init__()
        self.net = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=len(HEADS),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(x)
        return {h: out[:, i:i + 1] for i, h in enumerate(HEADS)}


def build_model(cfg: dict) -> FilSegModel:
    m = cfg.get("model", {})
    return FilSegModel(
        encoder=m.get("encoder", "timm-efficientnet-b4"),
        weights=m.get("weights", "imagenet"),
        in_channels=m.get("in_channels", 3),
    )


def encoder_decoder_param_groups(model: FilSegModel, lr: float, encoder_mult: float):
    """Two param groups: pretrained encoder at lr*mult, everything else at lr."""
    enc_params, rest = [], []
    enc_prefix = "net.encoder."
    for name, p in model.named_parameters():
        (enc_params if name.startswith(enc_prefix) else rest).append(p)
    return [
        {"params": enc_params, "lr": lr * encoder_mult},
        {"params": rest, "lr": lr},
    ]
