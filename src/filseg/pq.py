"""Faithful replica of the organizer's Panoptic Quality scorer (self-evaluation notebook).

Semantics (verified against selfeval/self-evaluation-notebook.ipynb):
- Evaluation units = GT "annotator-image" entries. Each annotator's set for a file is
  a SEPARATE unit scored against the SAME predictions for that file.
- IoU on flattened bool masks; TP iff IoU > 0.5 STRICTLY.
- Every IoU-matrix cell > 0.5 contributes its IoU to the TP sum (one-to-one in practice).
- FP = predictions with no cell > 0.5 in their column; FN = GT with none in their row.
- Global pooling: PQ = sum(TP IoUs) / (TP + 0.5*FP + 0.5*FN), single global ratio.
- Predictions for images absent from GT are ignored (as in the notebook loop over GT units).

Masks are handled as pycocotools compressed RLE dicts ({'size':[H,W],'counts':bytes}).
mask_util.iou(gt, pred, iscrowd=0) is mathematically identical to the notebook's
flattened matmul IoU and orders of magnitude faster / lighter.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

H = W = 2048


# ---------------------------------------------------------------- mask helpers

def poly_to_rle(seg_polys, h: int = H, w: int = W):
    """COCO polygon list -> single compressed RLE (merged)."""
    rles = mask_util.frPyObjects(seg_polys, h, w)
    return mask_util.merge(rles)


def mask_to_rle(mask: np.ndarray):
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


def rle_to_mask(rle) -> np.ndarray:
    return mask_util.decode(rle)


def rle_area(rle) -> float:
    return float(mask_util.area(rle))


def iou_matrix(gt_rles, pred_rles) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix; empty-safe."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    m = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))  # (n_pred, n_gt)
    return np.asarray(m, dtype=np.float64).T


# ---------------------------------------------------------------- PQ scoring

def pq_score(gt_units, preds, iou_thresh: float = 0.5, return_detail: bool = False):
    """Global PQ.

    Args:
        gt_units: dict {unit_id: (image_key, [rle, ...])} — one entry per
            annotator-image unit; image_key links the unit to predictions.
        preds: dict {image_key: [rle, ...]} — one prediction set per image.
        iou_thresh: strict lower bound for a TP match (default 0.5).
        return_detail: also return dict with counts and matched IoUs.

    Returns:
        pq (float), and detail dict if requested:
        {tp, fp, fn, sum_iou, matched_ious (list), per_unit (list of dicts)}
    """
    tp_ious = []
    fp = 0
    fn = 0
    per_unit = []
    for unit_id, (image_key, gt_rles) in gt_units.items():
        pred_rles = preds.get(image_key, [])
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            per_unit.append(dict(unit=unit_id, tp=0, fp=0, fn=n_gt))
            continue
        m = iou_matrix(gt_rles, pred_rles)
        hit = m > iou_thresh
        unit_tp_ious = m[hit].tolist()
        tp_ious.extend(unit_tp_ious)
        u_fp = int((hit.sum(axis=0) == 0).sum())
        u_fn = int((hit.sum(axis=1) == 0).sum())
        fp += u_fp
        fn += u_fn
        per_unit.append(dict(unit=unit_id, tp=len(unit_tp_ious), fp=u_fp, fn=u_fn,
                             ious=unit_tp_ious, iou_mat=m))
    tp = len(tp_ious)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = (sum(tp_ious) / denom) if denom > 0 else 0.0
    if return_detail:
        return pq, dict(tp=tp, fp=fp, fn=fn, sum_iou=sum(tp_ious),
                        matched_ious=tp_ious, per_unit=per_unit,
                        mean_matched_iou=(float(np.mean(tp_ious)) if tp_ious else 0.0))
    return pq


# ---------------------------------------------------------------- data loading

def load_gt(json_path, h: int = H, w: int = W):
    """Load the competition train JSON.

    Returns:
        files: dict {file_stem: [ (entry_id, [rle,...], [ann dict,...]) sorted by entry_id ]}
               one tuple per annotator set for that physical file.
    """
    import json as _json
    from collections import defaultdict

    with open(json_path, encoding="utf-8") as f:
        d = _json.load(f)
    anns_by_entry = defaultdict(list)
    for a in d["annotations"]:
        anns_by_entry[a["image_id"]].append(a)
    files = defaultdict(list)
    for im in d["images"]:
        stem = im["file_name"].rsplit(".", 1)[0]
        entry_id = im["id"]
        anns = anns_by_entry.get(entry_id, [])
        rles = [poly_to_rle(a["segmentation"], h, w) for a in anns]
        files[stem].append((entry_id, rles, anns))
    for stem in files:
        files[stem].sort(key=lambda t: t[0])
    return dict(files)


def units_from_files(files, file_subset=None, set_indices=None):
    """Build gt_units for pq_score from load_gt output.

    set_indices: which annotator-set positions (after sorting) to use as GT
        units; None = all.
    """
    units = {}
    for stem, sets in files.items():
        if file_subset is not None and stem not in file_subset:
            continue
        for i, (entry_id, rles, _) in enumerate(sets):
            if set_indices is not None and i not in set_indices:
                continue
            units[entry_id] = (stem, rles)
    return units
