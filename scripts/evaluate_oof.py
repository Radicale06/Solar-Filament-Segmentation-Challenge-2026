"""Score a trained checkpoint on its validation fold with the exact PQ replica.

python scripts/evaluate_oof.py --config configs/baseline.yaml --fold 0 \
    [--ckpt path] [--max-images N] [--tta]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import folds, paths, validate  # noqa: E402
from filseg.model import build_model  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = args.ckpt or (paths.checkpoints_dir() / f"{cfg['name']}_f{args.fold}" / "best.pth")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # architecture from the checkpoint's stored cfg (post-training yaml edits must
    # not diverge weights from model); CLI cfg still drives decode/infer params
    mcfg = {**state.get("cfg", cfg)}
    mcfg["model"] = {**mcfg.get("model", {}), "weights": None}
    model = build_model(mcfg).to(device)
    model.load_state_dict(state.get("ema", state["model"]))  # prefer EMA weights

    _train, val_stems = folds.split_stems(args.fold)
    rep = validate.evaluate_model(model, val_stems, cfg, device,
                                  tta=args.tta or None, max_images=args.max_images)
    print(json.dumps(rep, indent=2))

    out = paths.runs_dir() / f"oof_{cfg['name']}_f{args.fold}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"report -> {out}")
