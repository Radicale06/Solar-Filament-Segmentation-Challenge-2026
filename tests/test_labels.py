"""Consensus label-builder tests on synthetic annotator sets."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import labels, pq  # noqa: E402


def _rect_rle(x0, y0, w, h):
    m = np.zeros((labels.H, labels.W), dtype=np.uint8)
    m[y0:y0 + h, x0:x0 + w] = 1
    return pq.mask_to_rle(m)


def test_clustering_groups_overlapping_instances():
    a = _rect_rle(100, 100, 60, 60)
    b = _rect_rle(110, 100, 60, 60)   # IoU ~ 50/70 > 0.3 with a
    c = _rect_rle(1000, 1000, 40, 40)
    clusters = labels.cluster_instances([a, b, c])
    sizes = sorted(len(cl) for cl in clusters)
    assert sizes == [1, 2]


def test_consensus_mask_two_members_is_union():
    # mean-soft with 2 members: coverage*2 >= 2 -> any covered pixel (union)
    a = _rect_rle(100, 100, 50, 50)
    b = _rect_rle(120, 100, 50, 50)
    m = labels.consensus_mask([a, b])
    union = pq.rle_to_mask(pq.mask_util.merge([a, b])).astype(bool)
    assert (m == union).all()


def test_consensus_mask_three_members_majority():
    a = _rect_rle(100, 100, 50, 50)
    b = _rect_rle(100, 100, 50, 50)
    c = _rect_rle(400, 400, 50, 50)   # disjoint outlier member
    m = labels.consensus_mask([a, b, c])
    # pixels covered by >= ceil(3/2)=2 members -> only the a/b rectangle
    assert m[100:150, 100:150].all()
    assert not m[400:450, 400:450].any()


def test_build_file_singleton_kept_and_fraction_map():
    r1 = _rect_rle(200, 200, 80, 40)
    r2 = _rect_rle(205, 200, 80, 40)   # same filament, annotator 2's version
    solo = _rect_rle(900, 900, 60, 30)  # only annotator 1 drew this
    sets = [
        ("010101-x", [r1, solo], [dict(spine=[210.0, 220.0, 270.0, 220.0]), dict(spine=[905.0, 915.0, 950.0, 915.0])]),
        ("010102-x", [r2], [dict(spine=[215.0, 220.0, 275.0, 220.0])]),
    ]
    arrays, metas = labels.build_file("20110101000000Bh", sets)

    assert len(metas) == 2                       # cluster + kept singleton
    by_sets = sorted(m["n_sets"] for m in metas)
    assert by_sets == [1, 2]
    for m in metas:
        assert m["K"] == 2
        assert m["votes_frac"] in (0.5, 1.0)

    # fraction map: overlap of r1&r2 -> 255, solo region -> ~127
    soft = arrays["soft"]
    assert soft[220, 250] == 255                 # covered by both sets
    assert soft[915, 920] in (127, 128)          # covered by one of two sets
    assert arrays["spine"].max() == 255          # spines drawn
    assert arrays["inst"].max() == len(metas)    # instance ids painted
