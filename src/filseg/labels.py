"""Multi-annotator consensus label builder.

Instances are clustered across a file's annotator sets by pairwise IoU > 0.3
(greedy, largest-first). Each cluster's consensus mask is the pixelwise mean of
its member masks thresholded at 0.5 (ties in: `coverage * 2 >= n_members`).
Singleton clusters are kept. The per-pixel annotator fraction (how many of the
K sets cover the pixel) is stored as the soft supervision map.

Outputs under paths.labels_dir():
    inst/<stem>.png    uint16 instance-id map (0 = background); larger instances are
                       painted first so small ones win overlapping pixels
    soft/<stem>.png    uint8 annotator-fraction map (255 = all K sets cover the pixel)
    spine/<stem>.png   uint8 binary map of all annotators' spine polylines (3 px wide)
    meta.jsonl         one line per consensus instance:
                       {stem, inst_id, rle, area, bbox, n_members, n_sets, K, votes_frac}
    files.csv          per file: stem, K, n_clusters, n_anns, datetime, site
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import paths, pq
from .rle import rle_obj_to_str

H = W = 2048


# ------------------------------------------------------------------ clustering

def cluster_instances(rles: list, iou_thr: float = 0.3) -> list[list[int]]:
    """Greedy cross-set clustering: instances are visited largest-first; each
    joins the first cluster containing a member with IoU > iou_thr, else starts
    a new one. Returns clusters as lists of indices into `rles`."""
    n = len(rles)
    if n == 0:
        return []
    m = pq.iou_matrix(rles, rles)
    order = np.argsort([-pq.rle_area(r) for r in rles])
    clusters: list[list[int]] = []
    for i in order:
        target = None
        for ci, c in enumerate(clusters):
            if any(m[i, j] > iou_thr for j in c):
                target = ci
                break
        if target is None:
            clusters.append([int(i)])
        else:
            clusters[target].append(int(i))
    return clusters


def consensus_mask(member_rles: list) -> np.ndarray | None:
    """Mean-soft combination: pixels covered by >= half the members (ties in)."""
    if len(member_rles) == 1:
        return pq.rle_to_mask(member_rles[0]).astype(bool)
    votes = np.zeros((H, W), dtype=np.int16)
    for r in member_rles:
        votes += pq.rle_to_mask(r)
    out = votes * 2 >= len(member_rles)
    return out if out.any() else None


# ------------------------------------------------------------------ per-file build

def build_file(stem: str, sets: list) -> tuple[dict, list[dict]]:
    """Build consensus targets for one physical file.

    Args:
        sets: list of (entry_id, [rle...], [ann dict...]) — one per annotator set
            (pq.load_gt output for this stem).
    Returns:
        (arrays, instance_meta): arrays = {"inst": uint16 HxW, "soft": uint8 HxW,
        "spine": uint8 HxW}; instance_meta = list of per-instance dicts.
    """
    K = len(sets)
    flat_rles, flat_set_idx = [], []
    for si, (_eid, rles, _anns) in enumerate(sets):
        for r in rles:
            flat_rles.append(r)
            flat_set_idx.append(si)

    clusters = cluster_instances(flat_rles)

    # consensus instances, largest painted first so small ones survive overlaps
    inst_map = np.zeros((H, W), dtype=np.uint16)
    metas: list[dict] = []
    built = []
    for c in clusters:
        members = [flat_rles[j] for j in c]
        mask = consensus_mask(members)
        if mask is None:
            continue
        built.append((mask, c))
    built.sort(key=lambda t: -int(t[0].sum()))
    for inst_id, (mask, c) in enumerate(built, start=1):
        inst_map[mask] = inst_id
        rle = pq.mask_to_rle(mask)
        ys, xs = np.nonzero(mask)
        n_sets = len({flat_set_idx[j] for j in c})
        metas.append(dict(
            stem=stem,
            inst_id=inst_id,
            rle=rle_obj_to_str(rle),
            area=int(mask.sum()),
            bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            n_members=len(c),
            n_sets=n_sets,
            K=K,
            votes_frac=round(n_sets / K, 4),
        ))

    # annotator-fraction map: per set, union of its instances; average over sets
    frac = np.zeros((H, W), dtype=np.float32)
    for _eid, rles, _anns in sets:
        if rles:
            frac += pq.rle_to_mask(pq.mask_util.merge(rles)).astype(np.float32)
    soft = np.round(frac / K * 255.0).astype(np.uint8)

    # spines: all annotators' polylines, 3 px wide
    spine = np.zeros((H, W), dtype=np.uint8)
    for _eid, _rles, anns in sets:
        for a in anns:
            sp = a.get("spine")
            if not sp or len(sp) < 4:
                continue
            pts = np.round(np.asarray(sp, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(spine, [pts], isClosed=False, color=255, thickness=3)

    return {"inst": inst_map, "soft": soft, "spine": spine}, metas


# ------------------------------------------------------------------ dataset build

def parse_stem(stem: str) -> dict:
    """`YYYYMMDDHHMMSS` + site letter (+'h') -> {datetime, site}."""
    ts, site = stem[:14], stem[14]
    return dict(
        datetime=f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]}:{ts[12:14]}",
        site=site,
    )


def build_all(json_path=None, out_dir=None, log_every: int = 100) -> Path:
    """Build labels for every train file. Returns the output directory."""
    json_path = json_path or paths.train_json_path()
    out_dir = Path(out_dir) if out_dir else paths.labels_dir()
    for sub in ("inst", "soft", "spine"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    files = pq.load_gt(json_path)
    meta_f = open(out_dir / "meta.jsonl", "w", encoding="utf-8")
    rows = []
    for i, (stem, sets) in enumerate(sorted(files.items())):
        arrays, metas = build_file(stem, sets)
        cv2.imwrite(str(out_dir / "inst" / f"{stem}.png"), arrays["inst"])
        cv2.imwrite(str(out_dir / "soft" / f"{stem}.png"), arrays["soft"])
        cv2.imwrite(str(out_dir / "spine" / f"{stem}.png"), arrays["spine"])
        for m in metas:
            meta_f.write(json.dumps(m) + "\n")
        info = parse_stem(stem)
        rows.append(dict(stem=stem, K=len(sets), n_clusters=len(metas),
                         n_anns=sum(len(s[1]) for s in sets), **info))
        if log_every and (i + 1) % log_every == 0:
            print(f"  labels {i + 1}/{len(files)}", flush=True)
    meta_f.close()

    with open(out_dir / "files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "K", "n_clusters", "n_anns", "datetime", "site"])
        w.writeheader()
        w.writerows(rows)
    return out_dir


# ------------------------------------------------------------------ loading helpers

def load_files_csv(out_dir=None) -> list[dict]:
    out_dir = Path(out_dir) if out_dir else paths.labels_dir()
    with open(out_dir / "files.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_meta(out_dir=None) -> dict[str, list[dict]]:
    """meta.jsonl -> {stem: [instance meta dict, ...]}."""
    out_dir = Path(out_dir) if out_dir else paths.labels_dir()
    by_stem: dict[str, list[dict]] = defaultdict(list)
    with open(out_dir / "meta.jsonl", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            by_stem[m["stem"]].append(m)
    return dict(by_stem)
