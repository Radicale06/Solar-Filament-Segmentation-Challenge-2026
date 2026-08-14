"""Scorer-semantics parity tests.

These encode the organizer's self-evaluation-notebook behavior exactly
(research/report2_notebook.md). If any of these fail after a refactor, the
local validator no longer matches the leaderboard.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import pq  # noqa: E402


def _rect(x0, y0, w, h, H=64, W=64):
    m = np.zeros((H, W), dtype=np.uint8)
    m[y0:y0 + h, x0:x0 + w] = 1
    return pq.mask_to_rle(m)


def test_perfect_match():
    gt = {"a-img1": ("img1", [_rect(0, 0, 10, 10)])}
    preds = {"img1": [_rect(0, 0, 10, 10)]}
    score, d = pq.pq_score(gt, preds, return_detail=True)
    assert score == 1.0
    assert (d["tp"], d["fp"], d["fn"]) == (1, 0, 0)


def test_strictly_greater_than_half():
    # IoU exactly 0.5 must NOT match (organizer code uses strict >)
    gt = {"a-img1": ("img1", [_rect(0, 0, 10, 10)])}          # area 100
    preds = {"img1": [_rect(0, 0, 10, 5)]}                    # inter 50, union 100 -> IoU 0.5
    score, d = pq.pq_score(gt, preds, return_detail=True)
    assert d["tp"] == 0 and d["fp"] == 1 and d["fn"] == 1
    assert score == 0.0


def test_low_overlap_is_double_penalty():
    # a graze (IoU <= 0.5) costs both an FP and an FN
    gt = {"a-img1": ("img1", [_rect(0, 0, 10, 10)])}
    preds = {"img1": [_rect(8, 8, 10, 10)]}                   # tiny overlap
    _score, d = pq.pq_score(gt, preds, return_detail=True)
    assert (d["tp"], d["fp"], d["fn"]) == (0, 1, 1)


def test_duplicate_predictions_each_count_as_tp():
    # verified duplication property: every IoU>0.5 cell contributes
    gt = {"a-img1": ("img1", [_rect(0, 0, 10, 10)])}
    preds = {"img1": [_rect(0, 0, 10, 10), _rect(0, 0, 10, 10)]}
    score, d = pq.pq_score(gt, preds, return_detail=True)
    assert d["tp"] == 2 and d["fp"] == 0 and d["fn"] == 0
    assert score == 1.0  # sum_iou 2 / (2 + 0 + 0)


def test_multi_annotator_units_score_same_preds_independently():
    r = _rect(0, 0, 10, 10)
    other = _rect(30, 30, 8, 8)
    gt = {
        "a1-img1": ("img1", [r]),          # annotator 1 drew it
        "a2-img1": ("img1", [other]),      # annotator 2 drew something else
    }
    preds = {"img1": [r]}
    score, d = pq.pq_score(gt, preds, return_detail=True)
    # TP vs a1; same pred is FP vs a2, whose GT is FN
    assert (d["tp"], d["fp"], d["fn"]) == (1, 1, 1)
    assert abs(score - 1.0 / (1 + 0.5 + 0.5)) < 1e-9


def test_empty_predictions_all_fn():
    gt = {"a-img1": ("img1", [_rect(0, 0, 10, 10), _rect(20, 20, 5, 5)])}
    score, d = pq.pq_score(gt, {"img1": []}, return_detail=True)
    assert (d["tp"], d["fp"], d["fn"]) == (0, 0, 2)
    assert score == 0.0


def test_predictions_for_unknown_images_ignored():
    gt = {"a-img1": ("img1", [_rect(0, 0, 10, 10)])}
    preds = {"img1": [_rect(0, 0, 10, 10)], "img_unknown": [_rect(0, 0, 5, 5)]}
    score, d = pq.pq_score(gt, preds, return_detail=True)
    assert (d["tp"], d["fp"], d["fn"]) == (1, 0, 0)
    assert score == 1.0


def test_global_pooling_not_per_image_mean():
    # one perfect image + one all-FN image: global ratio, not mean of (1.0, 0.0)
    gt = {
        "a-img1": ("img1", [_rect(0, 0, 10, 10)]),
        "a-img2": ("img2", [_rect(0, 0, 10, 10), _rect(20, 20, 10, 10), _rect(40, 40, 10, 10)]),
    }
    preds = {"img1": [_rect(0, 0, 10, 10)], "img2": []}
    score = pq.pq_score(gt, preds)
    assert abs(score - 1.0 / (1 + 0.5 * 3)) < 1e-9  # = 0.4, not 0.5


def test_iou_matrix_matches_manual():
    a = _rect(0, 0, 10, 10)
    b = _rect(5, 0, 10, 10)   # inter 50, union 150
    m = pq.iou_matrix([a], [b])
    assert abs(m[0, 0] - 50.0 / 150.0) < 1e-9
