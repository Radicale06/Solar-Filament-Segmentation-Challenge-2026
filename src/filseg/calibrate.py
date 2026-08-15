"""Calibrated instance pruning.

A candidate instance is kept iff p_match * expected_iou > tau, where p_match is
a logistic model over per-instance features and expected_iou is the mean
matched IoU of similar-sized training candidates. tau is not derived
analytically — it is swept on the exact-PQ replica and the best value stored.

The fitted pruner is persisted as plain JSON so inference needs no sklearn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FEATURES = ("mean_prob", "p90_prob", "mean_fraction", "mean_spine", "log_area")


def featurize(cands: list[dict]) -> np.ndarray:
    rows = [[c["mean_prob"], c["p90_prob"], c["mean_fraction"], c["mean_spine"],
             math.log(max(c["area"], 1))] for c in cands]
    return np.asarray(rows, dtype=np.float64)


class Pruner:
    def __init__(self, mean, std, coef, intercept, area_edges, iou_by_bin, tau):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.area_edges = list(area_edges)
        self.iou_by_bin = list(iou_by_bin)
        self.tau = float(tau)

    # ------------------------------------------------------------- scoring

    def p_match(self, cands: list[dict]) -> np.ndarray:
        X = (featurize(cands) - self.mean) / self.std
        z = X @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def expected_iou(self, cands: list[dict]) -> np.ndarray:
        areas = np.asarray([c["area"] for c in cands], dtype=np.float64)
        bins = np.digitize(areas, self.area_edges)
        return np.asarray([self.iou_by_bin[b] for b in bins])

    def scores(self, cands: list[dict]) -> np.ndarray:
        if not cands:
            return np.zeros(0)
        return self.p_match(cands) * self.expected_iou(cands)

    def keep(self, cands: list[dict], tau: float | None = None) -> list[bool]:
        t = self.tau if tau is None else tau
        return (self.scores(cands) > t).tolist()

    # ------------------------------------------------------------- persistence

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(dict(
            mean=self.mean.tolist(), std=self.std.tolist(), coef=self.coef.tolist(),
            intercept=self.intercept, area_edges=self.area_edges,
            iou_by_bin=self.iou_by_bin, tau=self.tau, features=list(FEATURES),
        ), indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path) -> "Pruner":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["mean"], d["std"], d["coef"], d["intercept"],
                   d["area_edges"], d["iou_by_bin"], d["tau"])


def fit_pruner(cands: list[dict], tau: float = 0.22) -> Pruner:
    """Fit on dumped candidates (each with features + `matched` + `best_iou`)."""
    from sklearn.linear_model import LogisticRegression

    X = featurize(cands)
    y = np.asarray([bool(c["matched"]) for c in cands])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((X - mean) / std, y)

    # expected IoU by area quartile of the matched candidates
    areas = np.asarray([c["area"] for c in cands], dtype=np.float64)
    edges = np.quantile(areas, [0.25, 0.5, 0.75]).tolist()
    matched = [c for c in cands if c["matched"]]
    m_areas = np.asarray([c["area"] for c in matched], dtype=np.float64)
    m_ious = np.asarray([c["best_iou"] for c in matched], dtype=np.float64)
    overall = float(m_ious.mean()) if len(m_ious) else 0.6
    iou_by_bin = []
    bins = np.digitize(m_areas, edges) if len(matched) else np.zeros(0)
    for b in range(4):
        sel = bins == b
        iou_by_bin.append(float(m_ious[sel].mean()) if sel.any() else overall)

    return Pruner(mean, std, lr.coef_[0], lr.intercept_[0], edges, iou_by_bin, tau)
