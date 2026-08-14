# Solar Filament Segmentation Challenge 2026 — Consensus-Spine Pipeline

Pipeline for the Kaggle competition [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE Big Data Cup 2026 — pixel-precise instance segmentation of solar filaments in GONG H-alpha images,
scored with global Panoptic Quality at strict IoU>0.5 against multi-annotator ground truth).

Design rationale and competition intelligence: [`COMPETITION_BRIEF.md`](COMPETITION_BRIEF.md) and [`research/`](research/)
(especially `research/panel/panel_sim.md` — the measured facts every design choice traces back to — and
`research/panel/panel_verdict.md` — the architecture decision record).

## Setup

```bash
# Windows: use a venv at a SHORT path (see notes at the top of requirements.txt)
C:\Python314\python.exe -m venv C:\Users\oussa\venvs\filseg
C:\Users\oussa\venvs\filseg\Scripts\pip install -r requirements.txt
```

Competition data is expected at `MAGFiLO_1.0_Kaggle_2026/` in the repo root
(download with `python download_dataset.py`, requires `KAGGLE_API_TOKEN` in `.env`).
On Kaggle the paths resolve automatically to `/kaggle/input/filament-segmentation-2026/`.

## Pipeline

```bash
python scripts/build_labels.py    # consensus targets: instance/soft/spine maps + metadata  -> data/labels/
python scripts/make_folds.py      # 5-fold day-blocked GroupKFold                            -> data/folds.csv
python scripts/train_fold.py --config configs/baseline.yaml --fold 0
python scripts/predict_test.py --config configs/baseline.yaml --ckpt checkpoints/baseline_f0/best.pth
python scripts/evaluate_oof.py  --config configs/baseline.yaml --fold 0   # exact-PQ local validation
python scripts/make_submission.py --pred-dir runs/test_preds --out submissions/submission.csv
python -m pytest tests/           # scorer-parity + format unit tests
```

## Layout

```
src/filseg/         library (importable, modular — competition rules require reviewable code)
  pq.py             exact replica of the organizer's Panoptic Quality scorer (validated by tests)
  labels.py         multi-annotator consensus label builder (mean-soft policy, IoU>0.3 clustering)
  folds.py          day-blocked GroupKFold (48h union-find blocks; no temporal leakage)
  preprocessing.py  disk mask, radial limb-darkening flattening, CLAHE, Sobel/radius channels
  dataset.py        512² native-res crop sampling, d4 augs, soft targets
  model.py          U-Net++ (ConvNeXt-Tiny encoder), 4 heads: mask/boundary/spine/fraction
  losses.py         soft BCE + soft Dice + clDice + boundary Dice + spine/fraction MSE
  train.py          AMP training loop, EMA, cosine LR, checkpoint-resume (Kaggle-session-safe)
  infer.py          Gaussian-blended sliding-window inference, optional flip TTA
  decode.py         hysteresis -> components -> fragment bridging -> hole fill -> area/confidence pruning
  rle.py            COCO compressed-RLE submission I/O
  validate.py       OOF scoring harness (global PQ + per-area-quartile dashboard)
scripts/            thin CLI entry points
configs/            YAML experiment configs
tests/              unit tests, incl. scorer-semantics parity cases
notebooks/          Kaggle training/inference drivers
research/           competition intelligence, design-panel reports, metric simulations
```

## Rules compliance

- Trains ONLY on annotations provided through the competition; the public MAGFiLO release is never read.
- Inference uses only the provided test H-alpha images (no retrieval from the training set at test time).
- Predictions are one coherent instance per filament; submission via COCO compressed RLE (2048×2048).
