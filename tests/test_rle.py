"""Submission RLE round-trip tests."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import rle  # noqa: E402


def _blob(seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((rle.HEIGHT, rle.WIDTH), dtype=np.uint8)
    y, x = rng.integers(200, 1800, 2)
    m[y:y + 60, x:x + 140] = 1
    m[y + 20:y + 40, x + 140:x + 200] = 1  # L-shape
    return m


def test_round_trip():
    m = _blob()
    s = rle.mask_to_rle_str(m)
    back = rle.rle_str_to_mask(s)
    assert (back == m).all()


def test_rle_string_is_csv_safe():
    for seed in range(5):
        s = rle.mask_to_rle_str(_blob(seed))
        assert '"' not in s and "," not in s and "\n" not in s


def test_submission_write_read(tmp_path):
    preds = {"20110101000000Bh": [rle.mask_to_rle_obj(_blob(1)), rle.mask_to_rle_obj(_blob(2))],
             "20120202000000Ch": [rle.mask_to_rle_obj(_blob(3))]}
    out = tmp_path / "sub.csv"
    n = rle.write_submission(preds, out)
    assert n == 3

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["filament_id", "segmentation_rle"]
    assert rows[1][0] == "20110101000000Bh_1"
    # no quoting artifacts in the raw file
    raw = out.read_text(encoding="utf-8")
    assert '"' not in raw

    back = rle.read_submission(out)
    assert set(back) == set(preds)
    for stem in preds:
        for orig, rt in zip(preds[stem], back[stem]):
            assert (rle.mask_util.decode(orig) == rle.mask_util.decode(rt)).all()
