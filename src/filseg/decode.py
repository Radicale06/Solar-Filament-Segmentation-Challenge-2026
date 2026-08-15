"""Probability maps -> filament instances.

Chain: hysteresis threshold -> optional gentle closing -> connected components
-> per-instance hole fill (the annotation spec forbids holes) -> min-area
filter -> mean-probability confidence pruning.

Thresholds stay generous and nothing ever erodes: under-segmenting a thin
filament below IoU 0.5 costs both an FP and an FN, and splitting one filament
is far more expensive than merging two.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .constants import HEADS

_MASK = HEADS.index("mask")
_SPINE = HEADS.index("spine")
_FRAC = HEADS.index("fraction")


def hysteresis(prob: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Keep lo-threshold components that contain at least one hi pixel."""
    weak = (prob >= lo).astype(np.uint8)
    if not weak.any():
        return weak.astype(bool)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = np.unique(lab[prob >= hi])
    strong_labels = strong_labels[strong_labels > 0]
    return np.isin(lab, strong_labels)


def decode_probs(probs: np.ndarray, cfg: dict) -> tuple[list[np.ndarray], list[dict]]:
    """Returns (instance boolean masks, per-instance feature dicts)."""
    dc = cfg.get("decode", {})
    hi = dc.get("hi", 0.55)
    lo = dc.get("lo", 0.40)
    close = dc.get("close", 0)
    min_area = dc.get("min_area", 200)
    fill = dc.get("fill_holes", True)
    keep_mean_prob = dc.get("keep_mean_prob", 0.0)

    pm = probs[_MASK]
    binary = hysteresis(pm, hi, lo)

    if close and close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

    n, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    masks: list[np.ndarray] = []
    feats: list[dict] = []
    for i in range(1, n):
        core = lab == i
        m = ndimage.binary_fill_holes(core) if fill else core
        area = int(m.sum())
        if area < min_area:
            continue
        # confidence stats use pre-fill pixels: filled holes have ~zero
        # probability and would dilute the pruning signal
        mean_p = float(pm[core].mean())
        if mean_p < keep_mean_prob:
            continue
        feats.append(dict(
            area=area,
            mean_prob=mean_p,
            p90_prob=float(np.percentile(pm[core], 90)),
            mean_fraction=float(probs[_FRAC][core].mean()),
            mean_spine=float(probs[_SPINE][core].mean()),
        ))
        masks.append(m)
    return masks, feats
