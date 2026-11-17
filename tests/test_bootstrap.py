import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.summarize_holdout import rank_groups, weighted_auc


def test_weighted_auc_matches_explicit_bootstrap_including_ties():
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 30)
    p = rng.choice([.1, .3, .3, .7, .9], len(y))
    ranks = rank_groups(p)
    for _ in range(20):
        indices = rng.integers(0, len(y), len(y))
        weights = np.bincount(indices, minlength=len(y))
        assert weighted_auc(y, weights, ranks) == pytest.approx(roc_auc_score(y[indices], p[indices]), abs=1e-14)


def test_weighted_auc_rejects_single_class_resample():
    with pytest.raises(ValueError, match="Both classes"):
        weighted_auc(np.array([0, 1]), np.array([1, 0]), rank_groups([.1, .9]))
