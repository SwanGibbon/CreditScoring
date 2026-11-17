"""A final holdout excluded from all nested CV, tuning and calibration.

The function freezes predictions before consuming holdout labels. Its returned
models remain trained on development only, so a subsequent submission can use
the exact assessed policy without refitting on the final holdout.
"""
import numpy as np
import pandas as pd

from .evaluation import encode_threshold, fit_policies, nested_evaluate, validate_frames
from .protocol import Config, make_splits, metrics


def reserve_holdout(frames, y, config=Config(), holdout_folds=5, groups=None, times=None):
    y = validate_frames(frames, y, config.representations)
    if not isinstance(holdout_folds, int) or isinstance(holdout_folds, bool) or holdout_folds < 2:
        raise ValueError("holdout_folds must be an integer >= 2")
    # For time mode take the latest complete timestamp block and purge borrowers.
    development, holdout = make_splits(y, holdout_folds, config.seed + 17,
                                       config.split_mode, groups, times)[-1]
    return development, holdout


def independent_evaluate(frames, y, config=Config(), holdout_folds=5,
                         groups=None, times=None, partitions=None):
    y = validate_frames(frames, y, config.representations)
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y) or pd.isna(groups).any():
            raise ValueError("Invalid groups")
        if config.split_mode == "stratified" and len(np.unique(groups)) != len(groups):
            raise ValueError("Repeated borrowers require group or time mode")
    if times is not None:
        times = np.asarray(times)
        if times.ndim != 1 or len(times) != len(y) or pd.isna(times).any():
            raise ValueError("Invalid timestamps")
        if config.split_mode != "time":
            raise ValueError("Timestamps require time mode")
    if partitions is None:
        partitions = reserve_holdout(frames, y, config, holdout_folds, groups, times)
    development, holdout = (np.asarray(a) for a in partitions)
    if not len(development) or not len(holdout) or np.intersect1d(development, holdout).size:
        raise ValueError("Independent development and holdout must be nonempty and disjoint")
    for idx in [development, holdout]:
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(np.unique(idx)) != len(idx):
            raise ValueError("Invalid independent partition positions")
        if idx.min() < 0 or idx.max() >= len(y):
            raise ValueError("Independent partition positions out of bounds")
    # Explicit partitions are also checked: importing a saved plan cannot bypass
    # the no-overlap, chronology or borrower-isolation contract.
    if groups is not None and config.split_mode in {"group", "time"}:
        g = np.asarray(groups)
        if np.intersect1d(g[development], g[holdout]).size:
            raise ValueError("Borrowers overlap the independent holdout")
    if config.split_mode == "time":
        if times is None or not np.asarray(times)[development].max() < np.asarray(times)[holdout].min():
            raise ValueError("Independent holdout must follow development in time")
    subset = lambda a: None if a is None else np.asarray(a)[development]
    training = {name: frame.iloc[development] for name, frame in frames.items()}
    cv = nested_evaluate(training, y[development], config, subset(groups), subset(times))
    policies = fit_policies(training, y[development], config, subset(groups), subset(times))
    # Only now may untouched holdout features enter a prediction call.
    held_frames = {name: frame.iloc[holdout] for name, frame in frames.items()}
    cache = {}
    for policy in policies.values():
        if id(policy) not in cache:
            cache[id(policy)] = policy.predict(held_frames)
    records, rows = [], []
    for name, policy in policies.items():
        p = cache[id(policy)]
        records.append(dict(policy=name, representation=policy.representation, family=policy.family,
                            threshold=encode_threshold(policy.threshold),
                            **metrics(y[holdout], p, policy.threshold, config)))
        rows.append(pd.DataFrame(dict(row=frames["raw"].index[holdout], policy=name,
                                      target=y[holdout], probability=p, approve=p < policy.threshold)))
    return dict(development=cv, holdout_metrics=pd.DataFrame(records).set_index("policy"),
                holdout_predictions=pd.concat(rows, ignore_index=True), policies=policies,
                holdout_audit=dict(
                    protocol="development nested CV then frozen independent holdout; no holdout refit",
                    holdout_folds=holdout_folds, split_seed=config.seed + 17,
                    development_rows=frames["raw"].index[development].tolist(),
                    holdout_rows=frames["raw"].index[holdout].tolist(),
                    final_policies={name: policy.metadata for name, policy in policies.items()}))
