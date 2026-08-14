"""Predict the 180 test images and write a submission CSV.

python scripts/predict_test.py --config configs/baseline.yaml \
    [--ckpt path] [--out submissions/submission.csv] [--tta]
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import paths, rle, validate  # noqa: E402
from filseg.model import build_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = args.ckpt or (paths.checkpoints_dir() / f"{cfg['name']}_f{args.fold}" / "best.pth")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # architecture from the checkpoint's stored cfg; CLI cfg drives decode/infer
    mcfg = {**state.get("cfg", cfg)}
    mcfg["model"] = {**mcfg.get("model", {}), "weights": None}
    model = build_model(mcfg).to(device)
    model.load_state_dict(state.get("ema", state["model"]))  # prefer EMA weights

    stems = sorted(p.stem for p in paths.test_images_dir().glob("*.jpeg"))
    preds = validate.predict_stems(model, stems, cfg, device, tta=args.tta or None)

    out = Path(args.out) if args.out else paths.REPO_ROOT / "submissions" / f"{cfg['name']}_f{args.fold}.csv"
    n = rle.write_submission(preds, out)
    n_empty = sum(1 for v in preds.values() if not v)
    print(f"{n} instance rows for {len(preds)} images ({n_empty} images with no predictions) -> {out}")
