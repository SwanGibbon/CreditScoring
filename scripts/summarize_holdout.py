"""Paired stratified bootstrap for frozen independent-holdout predictions.

No training, model/threshold selection or input-data modification. Intervals
condition on fitted models and holdout class counts; they do not cover drift
or uncertainty from retraining. Assumes independent applications.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rank_groups(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Need finite one-dimensional scores")
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    return order, starts


def weighted_auc(y, counts, ranks):
    y, counts = np.asarray(y), np.asarray(counts)
    if counts.shape != y.shape or np.any(counts < 0):
        raise ValueError("Invalid resampling weights")
    order, starts = ranks
    positive = np.add.reduceat((counts * y)[order], starts)
    negative = np.add.reduceat((counts * (1 - y))[order], starts)
    if not positive.sum() or not negative.sum():
        raise ValueError("Both classes must have positive weight")
    below = np.cumsum(negative) - negative
    return float(np.dot(positive.astype(float), below + .5 * negative) /
                 (float(positive.sum()) * float(negative.sum())))


def summarize(directory, replicates=1000, seed=20261117):
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    directory = Path(directory)
    source = directory / "holdout_predictions.csv"
    data = pd.read_csv(source, float_precision="round_trip")
    if data.duplicated(["row", "policy"]).any():
        raise ValueError("Duplicate policy/row predictions")
    if data.groupby("row").target.nunique().ne(1).any():
        raise ValueError("Labels disagree between policies")
    matrix = data.pivot(index="row", columns="policy", values="probability")
    if matrix.isna().any().any():
        raise ValueError("All policies must share the same holdout rows")
    y = data.drop_duplicates("row").set_index("row").target.reindex(matrix.index).to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary labels")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    ranks = {name: rank_groups(matrix[name].to_numpy()) for name in matrix}
    point = {name: weighted_auc(y, np.ones(len(y), dtype=np.int64), rank) for name, rank in ranks.items()}
    samples = {name: np.empty(replicates) for name in matrix}
    for iteration in range(replicates):
        indices = np.r_[rng.choice(positive, len(positive), replace=True),
                        rng.choice(negative, len(negative), replace=True)]
        counts = np.bincount(indices, minlength=len(y))
        for name, rank in ranks.items():
            samples[name][iteration] = weighted_auc(y, counts, rank)
    rows = []
    for name in matrix:
        lower, upper = np.quantile(samples[name], [.025, .975])
        rows.append(dict(policy=name, auc=point[name], lower_95=lower, upper_95=upper))
    differences = []
    for left, right in [("history_lightgbm", "application_lightgbm"),
                        ("application_lightgbm", "raw_lightgbm"),
                        ("history_lightgbm", "raw_lightgbm")]:
        if left in samples and right in samples:
            lower, upper = np.quantile(samples[left] - samples[right], [.025, .975])
            differences.append(dict(comparison=left + "_minus_" + right,
                                    delta_auc=point[left] - point[right], lower_95=lower, upper_95=upper))
    output = directory / "auc-bootstrap"
    output.mkdir(exist_ok=False)
    intervals = pd.DataFrame(rows)
    paired = pd.DataFrame(differences)
    intervals.to_csv(output / "auc_intervals.csv", index=False)
    paired.to_csv(output / "paired_auc_differences.csv", index=False)
    provenance = dict(method="paired stratified percentile bootstrap", replicates=replicates,
                      seed=seed, confidence_level=.95, rows=len(y), positive_rows=len(positive),
                      interpretation="Conditional on fixed fitted models and class counts; IID applications assumed; no drift/retraining coverage",
                      input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / "method.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return intervals, paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    intervals, paired = summarize(args.directory, args.replicates, args.seed)
    print(intervals.to_string(index=False))
    print(paired.to_string(index=False))
