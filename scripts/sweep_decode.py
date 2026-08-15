"""Decode sweep + calibrated-pruner fit on the exact PQ replica.

Stages:
  1. Predict the fold's validation images once (single model or checkpoint
     ensemble, optional flip-TTA) and cache probability maps.
  2. Grid-sweep decode parameters (hysteresis, closing, dilation, min-area,
     mean-prob cut); candidates are decoded once per morphology combo and
     filters are evaluated on cached IoU matrices.
  3. For the best config: dump per-candidate features + match outcomes,
     fit a calibrated pruner (p_match * expected_iou > tau) on half the stems,
     sweep tau on the other half, refit on all candidates with the chosen tau.
  4. With --predict-test: write a submission using the best config + pruner.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import calibrate, decode, folds, infer, paths, pq, preprocessing, rle, validate  # noqa: E402
from filseg.model import build_model  # noqa: E402

# close/dilate were swept once and always lost; grids trimmed to what matters
HI_GRID = [0.50, 0.55, 0.60]
LO_GRID = [0.35, 0.40]
CLOSE_GRID = [0]
MIN_AREA_GRID = [100, 150, 200, 300, 400, 500]
KEEP_GRID = [0.0, 0.45, 0.50, 0.55]
DILATE_GRID = [0]
TAU_GRID = [0.0, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46]
CAND_FLOOR = 50  # decode floor; real min-area filters are applied post-hoc

KERNEL3 = np.ones((3, 3), np.uint8)


def load_model(ckpt_path, device):
    state = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
    cfg = state.get("cfg", {})
    mcfg = {**cfg}
    mcfg["model"] = {**mcfg.get("model", {}), "weights": None}
    m = build_model(mcfg).to(device)
    m.load_state_dict(state.get("ema", state["model"]))
    m.eval()
    return m, cfg


def predict_probs(model_of_stem, stems, cfg, device, tta):
    """Out-of-fold prediction: each stem is predicted by the model that did NOT
    train on it. Cached channels: mask, spine, fraction (uint8)."""
    ic = cfg.get("infer", {})
    cache = {}
    for i, stem in enumerate(stems):
        prep = preprocessing.load_prep(stem)
        probs = infer.predict_full_multi(model_of_stem[stem], prep, device,
                                         window=ic.get("window", 1024),
                                         stride=ic.get("stride", 768), tta=tta)
        cache[stem] = np.round(probs[[0, 2, 3]] * 255).astype(np.uint8)
        if (i + 1) % 20 == 0:
            print(f"  inference {i + 1}/{len(stems)}", flush=True)
    return cache


def decode_candidates(pm, spine_map, frac_map, hi, lo, close, want_dilate=False):
    """Candidates as features + compact RLEs — full masks are never retained
    (a cached-mask variant of this sweep OOM-killed a 284-image run)."""
    binary = decode.hysteresis(pm, hi, lo)
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    n, lab = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    from scipy import ndimage
    out = []
    for i in range(1, n):
        core = lab == i
        if core.sum() < CAND_FLOOR:
            continue
        m = ndimage.binary_fill_holes(core)
        c = dict(
            rle=pq.mask_to_rle(m),
            area=int(m.sum()),
            mean_prob=float(pm[core].mean()),
            p90_prob=float(np.percentile(pm[core], 90)),
            mean_fraction=float(frac_map[core].mean()),
            mean_spine=float(spine_map[core].mean()),
        )
        if want_dilate:
            c["rle_dil"] = pq.mask_to_rle(cv2.dilate(m.astype(np.uint8), KERNEL3).astype(bool))
        out.append(c)
    return out


def unit_matrices(gt_files, stems, cand_rles):
    units = []
    for stem in stems:
        for _entry_id, gt_rles, _ in gt_files[stem]:
            units.append((stem, pq.iou_matrix(gt_rles, cand_rles[stem])))
    return units


def pq_from_matrices(units, keep_of_stem, stems=None):
    keep_stems = set(stems) if stems is not None else None
    tp_iou, tp, fp, fn = 0.0, 0, 0, 0
    for stem, m in units:
        if keep_stems is not None and stem not in keep_stems:
            continue
        keep = keep_of_stem[stem]
        sub = m[:, keep] if m.size else m[:, :0]
        hit = sub > 0.5
        tp_iou += float(sub[hit].sum())
        tp += int(hit.sum())
        fp += int((~hit.any(axis=0)).sum()) if sub.shape[1] else 0
        fn += int((~hit.any(axis=1)).sum()) if sub.shape[0] else 0
    denom = tp + 0.5 * fp + 0.5 * fn
    return (tp_iou / denom if denom else 0.0), tp, fp, fn


def candidate_match_stats(units, cands_by_stem):
    """Attach matched / best_iou / unit counts to every candidate."""
    mats_by_stem: dict[str, list[np.ndarray]] = {}
    for stem, m in units:
        mats_by_stem.setdefault(stem, []).append(m)
    dump = []
    for stem, cs in cands_by_stem.items():
        mats = mats_by_stem.get(stem, [])
        for j, c in enumerate(cs):
            col_ious = [m[:, j] for m in mats if m.shape[1]]
            best = max((float(ci.max()) for ci in col_ious if ci.size), default=0.0)
            matched_units = sum(1 for ci in col_ious if ci.size and (ci > 0.5).any())
            sum_iou = sum(float(ci[ci > 0.5].sum()) for ci in col_ious if ci.size)
            dump.append(dict(stem=stem, area=c["area"], mean_prob=c["mean_prob"],
                             p90_prob=c["p90_prob"], mean_fraction=c["mean_fraction"],
                             mean_spine=c["mean_spine"], n_units=len(mats),
                             n_matched_units=matched_units, sum_matched_iou=round(sum_iou, 4),
                             best_iou=round(best, 4), matched=matched_units > 0))
    return dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True,
                    help="CKPT_PATH:FOLD pairs; each model predicts only its own held-out fold "
                         "for tuning (no leakage); all models ensemble for the test prediction")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny grids for smoke tests")
    ap.add_argument("--max-images", type=int, default=None, help="per fold")
    ap.add_argument("--out", default=None)
    ap.add_argument("--predict-test", action="store_true")
    args = ap.parse_args()

    global HI_GRID, LO_GRID, CLOSE_GRID, MIN_AREA_GRID, KEEP_GRID, DILATE_GRID
    if args.quick:
        HI_GRID, LO_GRID, CLOSE_GRID = [0.55], [0.40], [0]
        MIN_AREA_GRID, KEEP_GRID, DILATE_GRID = [100, 400], [0.0], [0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {}
    model_of_stem: dict[str, object] = {}
    all_models = []
    val_stems: list[str] = []
    for spec in args.oof:
        ckpt_path, fold_s = spec.rsplit(":", 1)
        fold = int(fold_s)
        model, cfg = load_model(ckpt_path, device)
        all_models.append(model)
        _, vs = folds.split_stems(fold)
        if args.max_images:
            idx = np.linspace(0, len(vs) - 1, args.max_images).round().astype(int)
            vs = [vs[i] for i in sorted(set(idx.tolist()))]
        for s in vs:
            model_of_stem[s] = model
        val_stems.extend(vs)
        print(f"fold {fold}: {len(vs)} val stems from {Path(ckpt_path).parent.name}", flush=True)
    print(f"{len(all_models)} model(s), {len(val_stems)} pooled OOF stems, tta={args.tta}", flush=True)

    probs = predict_probs(model_of_stem, val_stems, cfg, device, args.tta)
    gt_files = pq.load_gt(paths.train_json_path())

    # ---------------- stage 2: grid sweep ----------------
    results = []
    cache = {}  # (hi, lo, close) -> candidates by stem (features + RLEs only)
    want_dil = any(d for d in DILATE_GRID)
    for hi, lo, close in itertools.product(HI_GRID, LO_GRID, CLOSE_GRID):
        if lo >= hi:
            continue
        t0 = time.time()
        cands = {s: decode_candidates(probs[s][0].astype(np.float32) / 255.0,
                                      probs[s][1].astype(np.float32) / 255.0,
                                      probs[s][2].astype(np.float32) / 255.0,
                                      hi, lo, close, want_dilate=want_dil)
                 for s in val_stems}
        cache[(hi, lo, close)] = cands
        for dil in DILATE_GRID:
            key = "rle_dil" if dil else "rle"
            cand_rles = {s: [c[key] for c in cs] for s, cs in cands.items()}
            units = unit_matrices(gt_files, val_stems, cand_rles)
            for min_area, keep_p in itertools.product(MIN_AREA_GRID, KEEP_GRID):
                keep_of_stem = {s: [j for j, c in enumerate(cands[s])
                                    if c["area"] >= min_area and c["mean_prob"] >= keep_p]
                                for s in val_stems}
                score, tp, fp, fn = pq_from_matrices(units, keep_of_stem)
                results.append(dict(hi=hi, lo=lo, close=close, dilate=dil,
                                    min_area=min_area, keep_mean_prob=keep_p,
                                    pq=round(score, 5), tp=tp, fp=fp, fn=fn))
        print(f"pass hi={hi} lo={lo} close={close}: {time.time()-t0:.0f}s", flush=True)

    results.sort(key=lambda r: -r["pq"])
    best = results[0]
    out_path = Path(args.out) if args.out else paths.runs_dir() / "sweep_oof.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=1)
    print("top 5 grid configs:")
    for r in results[:5]:
        print(" ", r)

    # ---------------- stage 3: pruner ----------------
    cands = cache[(best["hi"], best["lo"], best["close"])]
    key = "rle_dil" if best["dilate"] else "rle"
    cand_rles = {s: [c[key] for c in cs] for s, cs in cands.items()}
    units = unit_matrices(gt_files, val_stems, cand_rles)
    dump = candidate_match_stats(units, cands)
    dump_path = out_path.parent / "candidates_oof.jsonl"
    with open(dump_path, "w", encoding="utf-8") as f:
        for d in dump:
            f.write(json.dumps(d) + "\n")
    print(f"{len(dump)} candidates dumped -> {dump_path}")

    fit_stems = set(val_stems[0::2])
    eval_stems = [s for s in val_stems if s not in fit_stems]
    fit_cands = [d for d in dump if d["stem"] in fit_stems]
    pruner = calibrate.fit_pruner(fit_cands)

    # tau sweep on the held-out half; pruner replaces the mean-prob cut and
    # uses a low area floor so the learned score does the filtering
    floor = 100
    print(f"tau sweep (eval half, {len(eval_stems)} stems, area floor {floor}):")
    tau_rows = []
    for tau in TAU_GRID:
        keep_of_stem = {}
        for s in val_stems:
            cs = cands[s]
            k = pruner.keep(cs, tau=tau) if cs else []
            keep_of_stem[s] = [j for j, c in enumerate(cs)
                               if c["area"] >= floor and (tau == 0.0 or k[j])]
        score, tp, fp, fn = pq_from_matrices(units, keep_of_stem, stems=eval_stems)
        tau_rows.append(dict(tau=tau, pq=round(score, 5), tp=tp, fp=fp, fn=fn))
        print(f"  tau={tau:.2f}: PQ {score:.4f} (TP {tp} FP {fp} FN {fn})")
    # best-grid-config reference on the same eval half
    ref_keep = {s: [j for j, c in enumerate(cands[s])
                    if c["area"] >= best["min_area"] and c["mean_prob"] >= best["keep_mean_prob"]]
                for s in val_stems}
    ref_score, *_ = pq_from_matrices(units, ref_keep, stems=eval_stems)
    print(f"  reference (best grid config): PQ {ref_score:.4f}")

    prunable = [r for r in tau_rows if r["tau"] > 0]
    best_tau_row = max(prunable, key=lambda r: r["pq"])
    use_pruner = best_tau_row["pq"] >= ref_score + 0.002  # must beat the grid by a real margin
    print(f"pruner {'WINS' if use_pruner else 'loses'}: tau={best_tau_row['tau']} "
          f"PQ {best_tau_row['pq']:.4f} vs grid {ref_score:.4f}")

    pruner = calibrate.fit_pruner(dump)  # refit on all candidates
    pruner.tau = best_tau_row["tau"]
    pruner_path = out_path.parent / "pruner_oof.json"
    pruner.to_json(pruner_path)
    json.dump(dict(grid_best=best, tau_rows=tau_rows, ref_eval_pq=ref_score,
                   use_pruner=use_pruner), open(out_path.parent / "decision_oof.json", "w"), indent=1)
    print(f"pruner -> {pruner_path} (use_pruner={use_pruner})")

    # ---------------- stage 4: test prediction ----------------
    if args.predict_test:
        best_cfg = dict(cfg)
        best_cfg["decode"] = dict(hi=best["hi"], lo=best["lo"], close=best["close"],
                                  fill_holes=True,
                                  min_area=floor if use_pruner else best["min_area"],
                                  keep_mean_prob=0.0 if use_pruner else best["keep_mean_prob"])
        test_stems = sorted(p.stem for p in paths.test_images_dir().glob("*.jpeg"))
        preds = validate.predict_stems(all_models, test_stems, best_cfg, device,
                                       tta=args.tta, pruner=pruner if use_pruner else None)
        if best["dilate"]:
            preds = {s: [pq.mask_to_rle(cv2.dilate(pq.rle_to_mask(r), KERNEL3).astype(bool))
                         for r in v] for s, v in preds.items()}
        sub_path = out_path.parent / "submission_sweep.csv"
        n = rle.write_submission({s: [rle.rle_obj_to_str(r) for r in v] for s, v in preds.items()}, sub_path)
        print(f"{n} rows -> {sub_path}")


if __name__ == "__main__":
    main()
