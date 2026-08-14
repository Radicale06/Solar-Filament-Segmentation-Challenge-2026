"""Training loop: AMP, EMA, cosine LR, checkpoint-resume (Kaggle-session-safe)."""
from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import folds, paths, validate
from .dataset import CropDataset
from .losses import CombinedLoss
from .model import build_model, encoder_decoder_param_groups


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1.0 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def _ckpt_dir(cfg: dict, fold: int) -> Path:
    d = paths.checkpoints_dir() / f"{cfg.get('name', 'run')}_f{fold}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_save(payload: dict, path: Path) -> None:
    """Session kills mid-write must never corrupt the only resume checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_fold(cfg: dict, fold: int, resume: bool = True, device_str: str | None = None) -> dict:
    tc = cfg.get("train", {})
    dc = cfg.get("data", {})
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(tc.get("seed", 0))

    train_stems, val_stems = folds.split_stems(fold)
    ds = CropDataset(train_stems, cfg, train=True)
    dl = DataLoader(ds, batch_size=dc.get("batch_size", 12), shuffle=True,
                    num_workers=dc.get("num_workers", 4), pin_memory=device.type == "cuda",
                    drop_last=True, persistent_workers=dc.get("num_workers", 4) > 0)

    model = build_model(cfg).to(device)
    loss_fn = CombinedLoss(cfg)
    lr = tc.get("lr", 3e-4)
    opt = torch.optim.AdamW(
        encoder_decoder_param_groups(model, lr, tc.get("encoder_lr_mult", 0.35)),
        weight_decay=tc.get("weight_decay", 1e-4))
    epochs = tc.get("epochs", 60)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(len(dl), 1))
    use_amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, tc.get("ema_decay", 0.999))

    ckdir = _ckpt_dir(cfg, fold)
    start_epoch, best_pq = 0, -1.0
    last = ckdir / "last.pth"
    if resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.shadow.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_pq = state.get("best_pq", -1.0)
        print(f"resumed from epoch {start_epoch}, best PQ {best_pq:.4f}")

    eval_every = tc.get("eval_every", 5)
    n_val = tc.get("n_val_images", 40)
    max_hours = tc.get("max_hours")
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch["image"])
                loss, _parts = loss_fn(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            run_loss += loss.item()
            nb += 1
        print(f"epoch {epoch}: loss {run_loss / max(nb, 1):.4f} "
              f"({time.time() - t0:.0f}s, lr {sched.get_last_lr()[-1]:.2e})", flush=True)

        is_eval = ((epoch + 1) % eval_every == 0) or (epoch == epochs - 1)
        if is_eval:
            ema.shadow.to(device)
            rep = validate.evaluate_model(ema.shadow, val_stems, cfg, device, max_images=n_val)
            print(f"  val PQ {rep['pq']:.4f} (TP {rep['tp']} FP {rep['fp']} FN {rep['fn']}, "
                  f"mIoU {rep['mean_matched_iou']:.3f}, {rep['n_pred_instances']} preds)", flush=True)
            if rep["pq"] > best_pq:
                best_pq = rep["pq"]
                _atomic_save({"model": ema.shadow.state_dict(), "cfg": cfg, "fold": fold,
                              "epoch": epoch, "pq": best_pq}, ckdir / "best.pth")
                print(f"  saved best.pth (PQ {best_pq:.4f})", flush=True)

        _atomic_save({"model": model.state_dict(), "ema": ema.shadow.state_dict(),
                      "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": epoch,
                      "best_pq": best_pq, "cfg": cfg}, last)

        if max_hours and (time.time() - t_start) / 3600.0 > max_hours:
            print(f"time budget {max_hours}h reached at epoch {epoch}; stopping cleanly", flush=True)
            break

    return {"best_pq": best_pq, "ckpt_dir": str(ckdir)}
