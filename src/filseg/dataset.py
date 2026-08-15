"""Crop dataset: native-resolution crops with positive-biased sampling.

No downscaling anywhere: the metric's strict IoU>0.5 matching leaves a ~±1 px
boundary error budget on the thin filaments. No elastic deformation (it breaks
thin-structure topology); geometric augs are the d4 group only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import paths, preprocessing

FULL = 2048


def _load_bboxes(labels_dir: Path) -> dict[str, list[list[int]]]:
    import json

    by_stem: dict[str, list[list[int]]] = {}
    with open(labels_dir / "meta.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            by_stem.setdefault(m["stem"], []).append(m["bbox"])
    return by_stem


def _load_k(labels_dir: Path) -> dict[str, int]:
    with open(labels_dir / "files.csv", newline="", encoding="utf-8") as f:
        return {r["stem"]: int(r["K"]) for r in csv.DictReader(f)}


class CropDataset(Dataset):
    def __init__(self, stems: list[str], cfg: dict, train: bool = True):
        d = cfg.get("data", {})
        self.stems = list(stems)
        self.crop = d.get("crop", 512)
        self.crops_per_image = d.get("crops_per_image", 8) if train else 2
        self.pos_frac = d.get("pos_frac", 0.7)
        self.train = train
        self.labels_dir = Path(d.get("labels_dir") or paths.labels_dir())
        self.bboxes = _load_bboxes(self.labels_dir)
        self.k_of = _load_k(self.labels_dir)

    def __len__(self) -> int:
        return len(self.stems) * self.crops_per_image

    # ------------------------------------------------------------- helpers

    def _read_labels(self, stem: str):
        inst = cv2.imread(str(self.labels_dir / "inst" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        soft = cv2.imread(str(self.labels_dir / "soft" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        spine = cv2.imread(str(self.labels_dir / "spine" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        return inst, soft, spine

    def _crop_origin(self, stem: str, rng: np.random.Generator) -> tuple[int, int]:
        c = self.crop
        boxes = self.bboxes.get(stem, [])
        if self.train and boxes and rng.random() < self.pos_frac:
            x, y, w, h = boxes[rng.integers(len(boxes))]
            cx = x + w // 2 + int(rng.integers(-c // 3, c // 3 + 1))
            cy = y + h // 2 + int(rng.integers(-c // 3, c // 3 + 1))
            x0, y0 = cx - c // 2, cy - c // 2
        else:
            x0 = int(rng.integers(0, FULL - c + 1))
            y0 = int(rng.integers(0, FULL - c + 1))
        return int(np.clip(x0, 0, FULL - c)), int(np.clip(y0, 0, FULL - c))

    # ------------------------------------------------------------- main

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx // self.crops_per_image]
        rng = np.random.default_rng(None if self.train else idx)
        c = self.crop

        img = preprocessing.load_prep(stem)          # HxWx3 uint8
        inst, soft, spine = self._read_labels(stem)

        x0, y0 = self._crop_origin(stem, rng)
        sl = np.s_[y0:y0 + c, x0:x0 + c]
        img_c = img[sl].astype(np.float32) / 255.0
        inst_c = inst[sl]
        soft_c = soft[sl].astype(np.float32) / 255.0
        spine_c = spine[sl].astype(np.float32) / 255.0

        if self.train:
            # d4: rotations by 90° + horizontal flip
            k = int(rng.integers(4))
            if k:
                img_c = np.rot90(img_c, k, axes=(0, 1))
                inst_c = np.rot90(inst_c, k)
                soft_c = np.rot90(soft_c, k)
                spine_c = np.rot90(spine_c, k)
            if rng.random() < 0.5:
                img_c = img_c[:, ::-1]
                inst_c = inst_c[:, ::-1]
                soft_c = soft_c[:, ::-1]
                spine_c = spine_c[:, ::-1]
            # photometric jitter on the intensity channel only
            g = img_c[..., 0]
            gain = float(rng.uniform(0.9, 1.1))
            bias = float(rng.uniform(-0.04, 0.04))
            gamma = float(rng.uniform(0.8, 1.25))
            g = np.clip(np.clip(g * gain + bias, 0, 1) ** gamma, 0, 1)
            img_c = np.stack([g, img_c[..., 1], img_c[..., 2]], axis=-1)

        img_c = np.ascontiguousarray(img_c)
        inst_c = np.ascontiguousarray(inst_c)
        soft_c = np.ascontiguousarray(soft_c)
        spine_c = np.ascontiguousarray(spine_c)

        hard = (inst_c > 0).astype(np.float32)

        # 3 px boundary band from the instance map
        hard_u8 = (inst_c > 0).astype(np.uint8)
        grad = cv2.morphologyEx(hard_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        boundary = cv2.dilate(grad, np.ones((3, 3), np.uint8)).astype(np.float32)

        # spine heatmap: blur the 3 px polyline map, rescale toward peak 1
        spine_t = cv2.GaussianBlur(spine_c, (13, 13), 3.0)
        spine_t = np.clip(spine_t * 2.5, 0, 1.0)

        return dict(
            image=torch.from_numpy(img_c).permute(2, 0, 1),
            soft=torch.from_numpy(soft_c)[None],
            hard=torch.from_numpy(hard)[None],
            boundary=torch.from_numpy(boundary)[None],
            spine=torch.from_numpy(spine_t)[None],
            frac_w=torch.tensor([1.0 if self.k_of.get(stem, 1) >= 2 else 0.0]),
        )
