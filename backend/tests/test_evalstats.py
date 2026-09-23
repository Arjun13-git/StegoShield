import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from app.ml.evalstats import auc_rank, holm_adjust, paired_auc_bootstrap


def test_auc_rank_matches_sklearn_including_ties() -> None:
    rng = np.random.default_rng(0)
    neg = np.round(rng.normal(0.0, 1.0, 200), 1)  # rounding creates ties
    pos = np.round(rng.normal(0.3, 1.0, 180), 1)
    y = np.array([0] * len(neg) + [1] * len(pos))
    assert auc_rank(neg, pos) == pytest.approx(roc_auc_score(y, np.concatenate([neg, pos])), abs=1e-12)


def test_auc_rank_extremes() -> None:
    assert auc_rank(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    assert auc_rank(np.array([0.8, 0.9]), np.array([0.1, 0.2])) == 0.0
    assert auc_rank(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5


def test_auc_rank_rejects_empty_class() -> None:
    with pytest.raises(ValueError):
        auc_rank(np.array([]), np.array([1.0]))


def test_holm_known_example() -> None:
    # sorted p: 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, max(0.06, 1*0.04) = 0.03, 0.06, 0.06
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_is_never_smaller_than_raw() -> None:
    raw = [0.5, 0.9, 0.001, 0.2]
    adj = holm_adjust(raw)
    assert all(a <= 1.0 for a in adj)
    assert all(a >= r for a, r in zip(adj, raw))


def test_bootstrap_ci_brackets_point_estimate_and_delta_zero_for_identical_baseline() -> None:
    rng = np.random.default_rng(1)
    cover = rng.normal(0.0, 1.0, 120)
    stego = cover + rng.normal(0.4, 1.0, 120)
    res = paired_auc_bootstrap(cover, stego, baseline_cover=cover, baseline_stego=stego, n_resamples=300)
    assert res["auc_ci"][0] <= res["auc"] <= res["auc_ci"][1]
    assert res["delta_auc"] == 0.0
    assert res["delta_auc_ci"] == [0.0, 0.0]


def test_bootstrap_requires_paired_inputs() -> None:
    with pytest.raises(ValueError):
        paired_auc_bootstrap(np.zeros(3), np.zeros(4))
