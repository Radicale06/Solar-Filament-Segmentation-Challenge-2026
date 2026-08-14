"""Day-blocked GroupKFold.

Files whose capture times are less than `block_hours` apart are chained into the
same block (union-find over the time-sorted sequence), so near-simultaneous
cross-site pairs and same-day observations can never straddle a train/val split.
Blocks are then dealt to folds greedily (largest first, to the currently
smallest fold) for balanced fold sizes. Deterministic.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from . import paths
from .labels import load_files_csv


def make_folds(n_folds: int = 5, block_hours: float = 48.0, out_csv=None, labels_dir=None) -> Path:
    rows = load_files_csv(labels_dir)
    for r in rows:
        r["_dt"] = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M:%S")
    rows.sort(key=lambda r: r["_dt"])

    # chain into blocks
    block_ids = []
    block = 0
    for i, r in enumerate(rows):
        if i > 0:
            gap_h = (r["_dt"] - rows[i - 1]["_dt"]).total_seconds() / 3600.0
            if gap_h >= block_hours:
                block += 1
        block_ids.append(block)

    # greedy deal: largest blocks first into the smallest fold
    from collections import Counter

    sizes = Counter(block_ids)
    fold_of_block: dict[int, int] = {}
    fold_sizes = [0] * n_folds
    for b, sz in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        f = min(range(n_folds), key=lambda k: (fold_sizes[k], k))
        fold_of_block[b] = f
        fold_sizes[f] += sz

    out_csv = Path(out_csv) if out_csv else paths.folds_csv()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "fold", "block", "datetime", "site", "K"])
        for r, b in zip(rows, block_ids):
            w.writerow([r["stem"], fold_of_block[b], b, r["datetime"], r["site"], r["K"]])
    print(f"folds: {n_folds}, blocks: {len(sizes)}, fold sizes: {fold_sizes}")
    return out_csv


def load_folds(csv_path=None) -> dict[str, int]:
    csv_path = Path(csv_path) if csv_path else paths.folds_csv()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["stem"]: int(row["fold"]) for row in csv.DictReader(f)}


def split_stems(fold: int, csv_path=None) -> tuple[list[str], list[str]]:
    """Returns (train_stems, val_stems) for a fold index."""
    fm = load_folds(csv_path)
    train = [s for s, f in fm.items() if f != fold]
    val = [s for s, f in fm.items() if f == fold]
    return train, val
