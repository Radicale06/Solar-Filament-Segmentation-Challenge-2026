"""Train one fold: python scripts/train_fold.py --config configs/baseline.yaml --fold 0"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg.train import train_fold  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    result = train_fold(cfg, args.fold, resume=not args.no_resume, device_str=args.device)
    print(result)
