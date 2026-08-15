"""OOF validation with the exact scorer replica.

Scores model predictions against ALL annotator sets of the validation files —
identical unit structure to the Kaggle evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

from . import decode, infer, paths, pq, preprocessing


def predict_stems(model, stems, cfg, device, tta: bool | None = None,
                  log_every: int = 20, pruner=None) -> dict[str, list]:
    """{stem: [pred rle, ...]} for a list of image stems.

    `model` may be a single model or a list (probability-averaged ensemble).
    `pruner` is an optional calibrate.Pruner applied to decoded instances.
    """
    ic = cfg.get("infer", {})
    tta = ic.get("tta", False) if tta is None else tta
    preds: dict[str, list] = {}
    for i, stem in enumerate(stems):
        prep = preprocessing.load_prep(stem)
        probs = infer.predict_full_multi(model, prep, device,
                                         window=ic.get("window", 1024),
                                         stride=ic.get("stride", 768), tta=tta)
        masks, feats = decode.decode_probs(probs, cfg)
        if pruner is not None and masks:
            keep = pruner.keep(feats)
            masks = [m for m, k in zip(masks, keep) if k]
        preds[stem] = [pq.mask_to_rle(m) for m in masks]
        if log_every and (i + 1) % log_every == 0:
            print(f"  predicted {i + 1}/{len(stems)}", flush=True)
    return preds


def score_preds(preds: dict[str, list], gt_files=None) -> dict:
    """Exact-PQ score of predictions against all annotator units of those files."""
    if gt_files is None:
        gt_files = pq.load_gt(paths.train_json_path())
    subset = {s: gt_files[s] for s in preds if s in gt_files}
    units = pq.units_from_files(subset)
    score, detail = pq.pq_score(units, preds, return_detail=True)

    # per-GT-area-quartile match survival
    areas, matched = [], []
    for unit_id, (stem, gt_rles) in units.items():
        pu = next((u for u in detail["per_unit"] if u["unit"] == unit_id), None)
        if pu is None or "iou_mat" not in pu:
            hit_rows = np.zeros(len(gt_rles), dtype=bool)
        else:
            hit_rows = (pu["iou_mat"] > 0.5).any(axis=1)
        for r, h in zip(gt_rles, hit_rows):
            areas.append(pq.rle_area(r))
            matched.append(bool(h))
    areas = np.asarray(areas)
    matched = np.asarray(matched)
    quart = {}
    if len(areas):
        qs = np.quantile(areas, [0.25, 0.5, 0.75])
        bins = np.digitize(areas, qs)
        for b in range(4):
            sel = bins == b
            quart[f"Q{b + 1}"] = dict(n=int(sel.sum()),
                                      match_rate=float(matched[sel].mean()) if sel.any() else 0.0)

    n_pred = sum(len(v) for v in preds.values())
    return dict(
        pq=score, tp=detail["tp"], fp=detail["fp"], fn=detail["fn"],
        mean_matched_iou=detail["mean_matched_iou"],
        n_pred_instances=n_pred, n_units=len(units),
        match_rate_by_area_quartile=quart,
    )


def evaluate_model(model, stems, cfg, device, tta=None, max_images=None) -> dict:
    stems = list(stems)
    if max_images and max_images < len(stems):
        # stems are time-sorted; stride evenly so every year/site is represented
        idx = np.linspace(0, len(stems) - 1, max_images).round().astype(int)
        stems = [stems[i] for i in sorted(set(idx.tolist()))]
    preds = predict_stems(model, stems, cfg, device, tta=tta)
    return score_preds(preds)
