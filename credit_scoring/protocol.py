"""Split and decision rules shared by tuning, evaluation, and tests."""
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class Config:
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 4
    split_mode: str = "stratified"
    max_rounds: int = 1200
    patience: int = 75
    top_k: int = 500
    probe_rounds: int = 150
    threads: int = 4
    fn_cost: float = 10.0
    fp_cost: float = 1.0
    families: tuple = ("lightgbm",)
    representations: tuple = ("raw", "application", "history")
    # A bounded, predeclared two-point search; no outer-label feedback.
    complexities: tuple = (15, 31)

    def __post_init__(self):
        for name in ["seed", "outer_folds", "inner_folds", "max_rounds", "patience", "top_k",
                     "probe_rounds", "threads"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0 or self.seed > 2**32 - 18:
            raise ValueError("seed must allow reproducible child seeds in uint32 range")
        if self.outer_folds < 2 or self.inner_folds < 3:
            raise ValueError("outer_folds >= 2 and inner_folds >= 3 required")
        if self.split_mode not in {"stratified", "group", "time"}:
            raise ValueError("Unknown split_mode")
        if min(self.max_rounds, self.patience, self.top_k,
               self.probe_rounds, self.threads) < 1:
            raise ValueError("Training limits must be positive")
        if not self.complexities or any(isinstance(c, bool) or not isinstance(c, int) or c < 2
                                       for c in self.complexities):
            raise ValueError("At least one complexity >= 2 required")
        if not self.families or set(self.families) - {"lightgbm", "xgboost", "catboost"}:
            raise ValueError("Unsupported model family")
        if len(set(self.families)) != len(self.families):
            raise ValueError("Duplicate model families")
        if "lightgbm" not in self.families:
            raise ValueError("LightGBM required for the predeclared ablation")
        if (not self.representations or "raw" not in self.representations
                or set(self.representations) - {"raw", "application", "history"}
                or len(set(self.representations)) != len(self.representations)):
            raise ValueError("Representations must be distinct known names and include raw")
        if "history" not in self.representations and self.families != ("lightgbm",):
            raise ValueError("Optional model families require the history representation")
        validate_costs(self.fn_cost, self.fp_cost)

    def as_dict(self):
        return asdict(self)


def binary(y):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected a nonempty binary target")
    return y.astype(np.int8)


def validate_costs(fn_cost, fp_cost):
    if not np.isfinite([fn_cost, fp_cost]).all() or min(fn_cost, fp_cost) < 0:
        raise ValueError("Costs must be finite and nonnegative")
    if fn_cost + fp_cost == 0:
        raise ValueError("At least one cost must be positive")


def scores(y, p):
    y = binary(y)
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected one finite probability in [0, 1] per label")
    return y, p


def choose_threshold(y, p, fn_cost=10.0, fp_cost=1.0):
    """Reject iff p >= threshold; move tied scores together, include both extremes.

    Ties in cost favor fewer approvals. Only calibration labels may enter here.
    """
    validate_costs(fn_cost, fp_cost)
    y, p = scores(y, p)
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)) + 1, len(p)]
    bad = np.r_[0, np.cumsum(ys)[ends - 1]]
    approved = np.r_[0, ends]
    rejected_good = (y == 0).sum() - (approved - bad)
    costs = (fn_cost * bad + fp_cost * rejected_good) / len(y)
    # Each threshold is the first rejected score, or +inf to approve everyone.
    thresholds = np.r_[-np.inf, ps[ends[:-1]], np.inf]
    return float(thresholds[np.argmin(costs)])


def decision_metrics(y, p, threshold, fn_cost=10.0, fp_cost=1.0):
    validate_costs(fn_cost, fp_cost)
    y, p = scores(y, p)
    if np.isnan(threshold):
        raise ValueError("Threshold cannot be NaN")
    approve = p < threshold
    fn = int(((y == 1) & approve).sum())
    fp = int(((y == 0) & ~approve).sum())
    return dict(cost=(fn_cost * fn + fp_cost * fp) / len(y),
                fn=fn, fp=fp, approval_rate=float(approve.mean()),
                approved_default_rate=float(y[approve].mean()) if approve.any() else None,
                approve_all_cost=float(fn_cost * y.mean()),
                reject_all_cost=float(fp_cost * (1 - y.mean())))


def metrics(y, p, threshold, config):
    result = decision_metrics(y, p, threshold, config.fn_cost, config.fp_cost)
    result.update(auc=float(roc_auc_score(y, p)),
                  ap=float(average_precision_score(y, p)),
                  brier=float(brier_score_loss(y, p)),
                  log_loss=float(log_loss(y, p, labels=[0, 1])), n=len(y))
    return result


def make_splits(y, n_splits, seed, mode="stratified", groups=None, times=None):
    """Time mode splits distinct timestamps into ordered expanding windows.

    Timestamps never straddle a boundary. If groups are also supplied in time
    mode, purge every training row whose group occurs in the validation window.
    """
    y = binary(y)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    groups = None if groups is None else np.asarray(groups)
    times = None if times is None else np.asarray(times)
    if groups is not None and (groups.ndim != 1 or len(groups) != len(y) or _missing(groups)):
        raise ValueError("Invalid groups")
    if mode == "group":
        if times is not None:
            raise ValueError("Timestamps supplied: use time mode with group purging")
        if groups is None:
            raise ValueError("Group mode requires verified borrower groups")
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
    elif mode == "time":
        if times is None or times.ndim != 1 or len(times) != len(y) or _missing(times):
            raise ValueError("Time mode requires complete application timestamps")
        unique = np.unique(times)
        if len(unique) < n_splits + 1:
            raise ValueError("Too few distinct timestamps")
        blocks = np.array_split(unique, n_splits + 1)
        splits = []
        for block in blocks[1:]:
            tr = np.flatnonzero(times < block[0])
            va = np.flatnonzero(np.isin(times, block))
            if groups is not None:
                tr = tr[~np.isin(groups[tr], groups[va])]
            splits.append((tr, va))
    elif mode == "stratified":
        if groups is not None and len(np.unique(groups)) != len(groups):
            raise ValueError("Repeated borrowers require group or time mode")
        if times is not None:
            raise ValueError("Timestamps supplied: explicitly choose time mode or remove them")
        splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                      .split(np.zeros(len(y)), y))
    else:
        raise ValueError("Unknown split mode")
    for tr, va in splits:
        if not len(tr) or not len(va) or np.intersect1d(tr, va).size:
            raise ValueError("Empty or overlapping split")
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError("Each partition needs both classes; reduce folds or add data")
        if mode in {"group", "time"} and groups is not None:
            if np.intersect1d(groups[tr], groups[va]).size:
                raise ValueError("Borrower overlap")
        if mode == "time" and not times[tr].max() < times[va].min():
            raise ValueError("Non-causal time split")
    return splits


def _missing(a):
    import pandas as pd
    return bool(pd.isna(a).any())


def inner_partitions(y, config, groups=None, times=None):
    """fit -> tune -> calibration, with identical grouping/time rules at all levels."""
    splits = make_splits(y, config.inner_folds, config.seed,
                         config.split_mode, groups, times)
    dev, calibration = splits[-1]
    subset = lambda a: None if a is None else np.asarray(a)[dev]
    nested = make_splits(np.asarray(y)[dev], config.inner_folds, config.seed + 1,
                         config.split_mode, subset(groups), subset(times))
    fit, tune = nested[-1]
    return dev[fit], dev[tune], calibration
