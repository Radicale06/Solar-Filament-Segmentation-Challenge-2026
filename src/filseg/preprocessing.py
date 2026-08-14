"""Image preprocessing: disk geometry, radial flattening, CLAHE, model channels.

Channel stack fed to the model (all uint8 in the cache, scaled to [0,1] at load):
    0: CLAHE-enhanced, radially flattened intensity (0 off-disk)
    1: Sobel gradient magnitude of channel 0 (robustly normalized, 0 off-disk)
    2: normalized radial distance from disk center (r / disk_radius, clipped to 1.2)

GONG JPEGs are already exposure- and limb-darkening-normalized by NSO; the radial
flattening here removes the mild residual radial profile so intensities are
comparable from disk center to limb.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from . import paths

H = W = 2048


def read_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def solar_disk(img: np.ndarray, thresh: int = 10) -> tuple[float, float, float, np.ndarray]:
    """Locate the solar disk. Returns (cx, cy, radius, disk_mask_bool)."""
    m = (img > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return W / 2, H / 2, W / 2 * 0.83, m.astype(bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = lab == big
    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(np.sqrt(mask.sum() / np.pi))
    return cx, cy, r, mask


def radial_grid(cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(r, 1.0)


def radial_flatten(img: np.ndarray, rr: np.ndarray, disk: np.ndarray, nbins: int = 64) -> np.ndarray:
    """Divide out the median radial intensity profile inside the disk."""
    vals = img.astype(np.float32)
    prof_r = np.linspace(0, 1.0, nbins + 1)
    centers = 0.5 * (prof_r[:-1] + prof_r[1:])
    med = np.empty(nbins, dtype=np.float32)
    idx = np.clip((rr[disk] * nbins).astype(np.int32), 0, nbins - 1)
    dvals = vals[disk]
    for b in range(nbins):
        sel = dvals[idx == b]
        med[b] = np.median(sel) if sel.size else np.nan
    # fill empty bins from neighbors
    ok = ~np.isnan(med)
    if not ok.any():
        return np.where(disk, vals, 0.0)
    med = np.interp(centers, centers[ok], med[ok]).astype(np.float32)
    target = float(np.median(dvals)) or 1.0
    prof_img = np.interp(np.clip(rr, 0, 1.0), centers, med).astype(np.float32)
    out = vals / np.maximum(prof_img, 1.0) * target
    return np.where(disk, out, 0.0)


def preprocess(img: np.ndarray) -> np.ndarray:
    """uint8 HxW -> uint8 HxWx3 model-channel stack."""
    cx, cy, r, disk = solar_disk(img)
    rr = radial_grid(cx, cy, r)
    flat = radial_flatten(img, rr, disk)
    flat_u8 = np.clip(flat, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    clahe = np.where(disk, clahe, 0).astype(np.uint8)

    g = clahe.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    inside = mag[disk]
    p99 = float(np.percentile(inside, 99)) if inside.size else 1.0
    mag = np.clip(mag / max(p99, 1e-6), 0, 1.0)
    mag = np.where(disk, mag, 0.0)

    rad = np.clip(rr / 1.2, 0, 1.0)

    return np.stack([
        clahe,
        np.round(mag * 255).astype(np.uint8),
        np.round(rad * 255).astype(np.uint8),
    ], axis=-1)


# Bump this whenever preprocess() or its parameters change: the cache is keyed by
# it, so stale features from an older recipe can never be silently served.
PREP_VERSION = 1


def prep_path(stem: str) -> Path:
    d = paths.prep_dir() / f"v{PREP_VERSION}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.png"


def load_prep(stem: str, image_dir=None) -> np.ndarray:
    """Load the cached preprocessed stack for a stem, computing it on miss."""
    p = prep_path(stem)
    if p.exists():
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.shape == (H, W, 3):
            return arr
    if image_dir is None:
        cand = paths.train_images_dir() / f"{stem}.jpeg"
        image_dir = paths.train_images_dir() if cand.exists() else paths.test_images_dir()
    arr = preprocess(read_gray(Path(image_dir) / f"{stem}.jpeg"))
    tmp = p.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), arr)
    os.replace(tmp, p)
    return arr


def build_prep_cache(stems=None, log_every: int = 100) -> None:
    """Precompute the preprocessed stack for all (or given) images."""
    if stems is None:
        stems = sorted(p.stem for p in paths.train_images_dir().glob("*.jpeg"))
        stems += sorted(p.stem for p in paths.test_images_dir().glob("*.jpeg"))
    for i, stem in enumerate(stems):
        load_prep(stem)
        if log_every and (i + 1) % log_every == 0:
            print(f"  prep {i + 1}/{len(stems)}", flush=True)
