"""Create the frozen day-blocked GroupKFold split."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import folds  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--block-hours", type=float, default=48.0)
    args = ap.parse_args()
    out = folds.make_folds(n_folds=args.n_folds, block_hours=args.block_hours)
    print(f"folds written to {out}")
