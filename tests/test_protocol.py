import numpy as np
import pytest

from credit_scoring.protocol import (
    Config, choose_threshold, decision_metrics, inner_partitions, make_splits,
)


@pytest.mark.parametrize("seed", range(12))
def test_threshold_matches_exhaustive_decisions_including_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 50)
    p = rng.choice([0., .2, .5, .8, 1.], 50)
    threshold = choose_threshold(y, p, 10, 1)
    actual = decision_metrics(y, p, threshold, 10, 1)["cost"]
    candidates = np.r_[-np.inf, np.unique(p), np.inf]
    expected = min(decision_metrics(y, p, t, 10, 1)["cost"] for t in candidates)
    assert actual == expected


def test_threshold_extremes_and_equal_scores():
    assert choose_threshold([1, 1], [.1, .9]) == -np.inf
    assert choose_threshold([0, 0], [.1, .9]) == np.inf
    t = choose_threshold([0, 1], [.5, .5])
    assert t == -np.inf
    assert decision_metrics([0, 1], [.5, .5], t)["approval_rate"] == 0
    assert decision_metrics([1, 0], [.1, .9], .5)["cost"] == 5.5
    assert decision_metrics([0, 1], [.5, .6], .5)["fp"] == 1  # reject equality


@pytest.mark.parametrize("y,p,fn,fp", [([], [], 10, 1), ([2], [.3], 10, 1),
    ([0], [np.nan], 10, 1), ([0], [1.1], 10, 1), ([0], [.3], -1, 1),
    ([0], [.3], 0, 0), ([0], [.3], np.inf, 1)])
def test_invalid_decision_inputs(y, p, fn, fp):
    with pytest.raises(ValueError):
        choose_threshold(y, p, fn, fp)


def test_group_isolation_at_both_levels():
    groups = np.repeat(np.arange(120), 4)
    y = np.tile([0, 1, 0, 1], 120)
    cfg = Config(outer_folds=3, inner_folds=3, split_mode="group")
    for train, val in make_splits(y, 3, 42, "group", groups):
        assert not set(groups[train]) & set(groups[val])
        partitions = inner_partitions(y[train], cfg, groups[train])
        for i, left in enumerate(partitions):
            for right in partitions[i + 1:]:
                assert not set(groups[train][left]) & set(groups[train][right])


def test_time_splits_do_not_split_equal_dates_and_purge_shared_borrowers():
    times = np.repeat(np.arange(120), 4)
    y = np.tile([0, 1, 0, 1], 120)
    groups = np.arange(len(y))
    groups[-4:] = groups[:4]  # crossing train/future boundary must be purged
    cfg = Config(outer_folds=3, inner_folds=3, split_mode="time")
    for tr, va in make_splits(y, 3, 42, "time", groups, times):
        assert times[tr].max() < times[va].min()
        assert not set(groups[tr]) & set(groups[va])
        fit, tune, cal = inner_partitions(y[tr], cfg, groups[tr], times[tr])
        assert times[tr][fit].max() < times[tr][tune].min()
        assert times[tr][tune].max() < times[tr][cal].min()


def test_random_split_rejects_known_repeated_people_or_timestamps():
    y = np.tile([0, 1], 30)
    with pytest.raises(ValueError, match="Repeated borrowers"):
        make_splits(y, 3, 42, groups=np.repeat(np.arange(30), 2))
    with pytest.raises(ValueError, match="Timestamps supplied"):
        make_splits(y, 3, 42, times=np.arange(60))
    with pytest.raises(ValueError, match="requires verified"):
        make_splits(y, 3, 42, "group")
