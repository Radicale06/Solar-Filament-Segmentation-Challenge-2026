"""Losses.

Mask head: soft-BCE against the annotator-fraction map (encodes annotator
disagreement), soft Dice + clDice against the hard consensus mask. clDice
(Shit et al., CVPR 2021) is the centerline-Dice for tubular structures and
penalizes fragmentation, which the instance metric punishes far harder than
over-merging.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(2, 3)) + eps
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()


# ----------------------------------------------------------------- soft skeleton

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, 3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, iters: int = 8) -> torch.Tensor:
    skel = F.relu(img - _soft_open(img))
    for _ in range(iters):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(logits: torch.Tensor, target: torch.Tensor, iters: int = 8,
                 eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    skel_p = soft_skeleton(p, iters)
    skel_t = soft_skeleton(target, iters)
    tprec = ((skel_p * target).sum(dim=(2, 3)) + eps) / (skel_p.sum(dim=(2, 3)) + eps)
    tsens = ((skel_t * p).sum(dim=(2, 3)) + eps) / (skel_t.sum(dim=(2, 3)) + eps)
    return 1.0 - (2.0 * tprec * tsens / (tprec + tsens)).mean()


# ----------------------------------------------------------------- combined

class CombinedLoss(nn.Module):
    """Weighted multi-head loss; weights come from cfg['loss']."""

    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg.get("loss", {})
        self.w_bce = lc.get("mask_bce", 0.4)
        self.w_dice = lc.get("mask_dice", 0.4)
        self.w_cldice = lc.get("mask_cldice", 0.2)
        self.cl_iters = lc.get("cldice_iters", 8)
        self.w_boundary = lc.get("boundary", 0.3)
        self.w_spine = lc.get("spine", 0.5)
        self.w_fraction = lc.get("fraction", 0.3)

    def forward(self, out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        soft = batch["soft"]        # annotator fraction in [0,1]
        hard = batch["hard"]        # binary consensus mask

        l_bce = F.binary_cross_entropy_with_logits(out["mask"], soft)
        l_dice = soft_dice_loss(out["mask"], hard)
        l_cl = cl_dice_loss(out["mask"], hard, iters=self.cl_iters)
        l_bnd = soft_dice_loss(out["boundary"], batch["boundary"])
        l_spine = F.mse_loss(torch.sigmoid(out["spine"]), batch["spine"])

        # fraction head is supervised only on multi-annotator files (frac_w flag)
        fw = batch["frac_w"].view(-1)                                    # (B,)
        per_sample = F.mse_loss(torch.sigmoid(out["fraction"]), soft,
                                reduction="none").mean(dim=(1, 2, 3))    # (B,)
        l_frac = (per_sample * fw).sum() / fw.sum().clamp(min=1.0)

        total = (self.w_bce * l_bce + self.w_dice * l_dice + self.w_cldice * l_cl
                 + self.w_boundary * l_bnd + self.w_spine * l_spine
                 + self.w_fraction * l_frac)
        parts = dict(bce=l_bce.item(), dice=l_dice.item(), cldice=l_cl.item(),
                     boundary=l_bnd.item(), spine=l_spine.item(),
                     fraction=l_frac.item(), total=total.item())
        return total, parts
