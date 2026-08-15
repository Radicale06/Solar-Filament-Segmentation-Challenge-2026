"""COCO compressed-RLE helpers and submission I/O.

Submission format (per competition Submission File page):
one row per predicted filament instance, columns `filament_id` = "<imageStem>_<n>"
and `segmentation_rle` = compressed RLE counts string for a 2048x2048 mask,
unquoted. The compressed-RLE alphabet (ASCII 48..111) never contains commas or
quotes, so plain csv writing never triggers quoting.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

HEIGHT = WIDTH = 2048
SIZE = [HEIGHT, WIDTH]


def mask_to_rle_obj(mask: np.ndarray) -> dict:
    """Binary HxW mask -> pycocotools compressed RLE dict."""
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def rle_obj_to_str(rle: dict) -> str:
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def str_to_rle_obj(s: str, size=None) -> dict:
    return {"size": list(size or SIZE), "counts": s.encode("utf-8")}


def mask_to_rle_str(mask: np.ndarray) -> str:
    return rle_obj_to_str(mask_to_rle_obj(mask))


def rle_str_to_mask(s: str, size=None) -> np.ndarray:
    return mask_util.decode(str_to_rle_obj(s, size))


def write_submission(preds: dict[str, list], out_path) -> int:
    """Write a submission CSV. Images with zero predicted instances get no rows.

    Args:
        preds: {image_stem: [rle, ...]} where each rle is a compressed RLE dict
            or an already-encoded counts string.
    Returns: number of instance rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        for stem in sorted(preds):
            for i, r in enumerate(preds[stem], start=1):
                s = r if isinstance(r, str) else rle_obj_to_str(r)
                w.writerow([f"{stem}_{i}", s])
                n += 1
    return n


def read_submission(path) -> dict[str, list[dict]]:
    """Read a submission CSV back into {image_stem: [rle_obj, ...]}."""
    preds: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            stem = row["filament_id"].rsplit("_", 1)[0]
            preds.setdefault(stem, []).append(str_to_rle_obj(row["segmentation_rle"]))
    return preds
