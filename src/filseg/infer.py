"""Gaussian-blended sliding-window inference at native resolution."""
from __future__ import annotations

import numpy as np
import torch

from .constants import HEADS, FULL


def _gaussian_window(size: int, sigma_frac: float = 0.5) -> np.ndarray:
    ax = np.linspace(-1, 1, size)
    g = np.exp(-(ax ** 2) / (2 * sigma_frac ** 2))
    w = np.outer(g, g).astype(np.float32)
    return np.maximum(w, 1e-3)


def _positions(full: int, window: int, stride: int) -> list[int]:
    pos = list(range(0, full - window + 1, stride))
    if pos[-1] != full - window:
        pos.append(full - window)
    return pos


@torch.no_grad()
def predict_full(model: torch.nn.Module, prep_u8: np.ndarray, device: torch.device,
                 window: int = 1024, stride: int = 768, tta: bool = False,
                 amp: bool = True, batch_tiles: int = 4) -> np.ndarray:
    """Predict all heads for one full 2048² image.

    Args:
        prep_u8: HxWx3 uint8 preprocessed stack.
    Returns:
        (len(HEADS), H, W) float32 probability maps in head order.
    """
    model.eval()
    x_full = torch.from_numpy(prep_u8.astype(np.float32) / 255.0).permute(2, 0, 1)

    wmap = torch.from_numpy(_gaussian_window(window))
    acc = torch.zeros(len(HEADS), FULL, FULL)
    wacc = torch.zeros(1, FULL, FULL)

    tiles, coords = [], []
    for y0 in _positions(FULL, window, stride):
        for x0 in _positions(FULL, window, stride):
            tiles.append(x_full[:, y0:y0 + window, x0:x0 + window])
            coords.append((y0, x0))

    use_amp = amp and device.type == "cuda"
    for i in range(0, len(tiles), batch_tiles):
        batch = torch.stack(tiles[i:i + batch_tiles]).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch)
            probs = torch.sigmoid(torch.cat([out[h] for h in HEADS], dim=1))
            if tta:
                out_f = model(torch.flip(batch, dims=[3]))
                probs_f = torch.sigmoid(torch.cat([out_f[h] for h in HEADS], dim=1))
                probs = 0.5 * (probs + torch.flip(probs_f, dims=[3]))
        probs = probs.float().cpu()
        for j, (y0, x0) in enumerate(coords[i:i + batch_tiles]):
            acc[:, y0:y0 + window, x0:x0 + window] += probs[j] * wmap
            wacc[:, y0:y0 + window, x0:x0 + window] += wmap

    return (acc / wacc).numpy()


@torch.no_grad()
def predict_full_multi(models, prep_u8: np.ndarray, device: torch.device,
                       **kwargs) -> np.ndarray:
    """Ensemble: average of predict_full over one or more models."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    acc = None
    for m in models:
        p = predict_full(m, prep_u8, device, **kwargs)
        acc = p if acc is None else acc + p
    return acc / len(models)
