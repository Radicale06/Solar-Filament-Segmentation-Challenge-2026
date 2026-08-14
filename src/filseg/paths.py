"""Path resolution for local and Kaggle environments."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_KAGGLE_COMP = Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")


def data_root() -> Path:
    """Root of the competition data (contains train/ and test/)."""
    env = os.environ.get("FILSEG_DATA")
    if env:
        return Path(env)
    local = REPO_ROOT / "MAGFiLO_1.0_Kaggle_2026"
    if local.exists():
        return local
    if _KAGGLE_COMP.exists():
        return _KAGGLE_COMP
    raise FileNotFoundError(
        "Competition data not found. Set FILSEG_DATA or place MAGFiLO_1.0_Kaggle_2026/ in the repo root."
    )


def train_images_dir() -> Path:
    return data_root() / "train" / "train_images"


def test_images_dir() -> Path:
    return data_root() / "test" / "test_images"


def train_json_path() -> Path:
    return data_root() / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def work_root() -> Path:
    """Writable root for derived artifacts (labels, caches, folds, runs)."""
    env = os.environ.get("FILSEG_WORK")
    if env:
        p = Path(env)
    elif Path("/kaggle/working").exists():
        p = Path("/kaggle/working") / "filseg_work"
    else:
        p = REPO_ROOT / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sub(name: str) -> Path:
    p = work_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def labels_dir() -> Path:
    return _sub("labels")


def prep_dir() -> Path:
    return _sub("prep")


def folds_csv() -> Path:
    return work_root() / "folds.csv"


def checkpoints_dir() -> Path:
    return _sub("checkpoints")


def runs_dir() -> Path:
    return _sub("runs")
