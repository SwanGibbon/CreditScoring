"""Nested evaluation with a separate threshold-calibration holdout.

fit_policies receives ONLY outer-training data. No validation feature, label,
category, missingness rate, stopping signal, or threshold enters that function.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from .protocol import Config, binary, choose_threshold, inner_partitions, make_splits, metrics


class FoldTransform:
    """Fitted filtering, vocabulary and supervised ranking, all local to fit()."""
    def __init__(self, config):
        self.config = config

    def fit(self, x, y):
        import lightgbm as lgb
        if {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU", "TARGET"} & set(x.columns):
            raise ValueError("Identifiers/target cannot be model inputs")
        if not x.columns.is_unique:
            raise ValueError("Duplicate feature names")
        x = x.copy().replace([np.inf, -np.inf], np.nan)
        self.fit_index_ = x.index.to_numpy().copy()
        self.columns_ = list(x.columns[(x.nunique(dropna=False) > 1) & (x.isna().mean() <= .99)])
        if not self.columns_:
            raise ValueError("No nonconstant features in this training partition")
        num = list(x[self.columns_].select_dtypes(include="number").columns)
        cat = [c for c in self.columns_ if c not in num]
        # Constant imputation preserves a distinct missing value without learning
        # a median; tree splits can separate it. The exact rule is predeclared.
        self.encoder_ = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="constant", fill_value=-999,
                                      keep_empty_features=True), num),
            ("missing", MissingIndicator(features="all", sparse=True), num),
            ("category", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), cat),
        ], sparse_threshold=1.0)
        a = self.encoder_.fit_transform(self.clean(x[self.columns_]))
        probe = lgb.LGBMClassifier(
            n_estimators=self.config.probe_rounds, learning_rate=.05, num_leaves=15,
            n_jobs=self.config.threads, random_state=self.config.seed,
            verbosity=-1, deterministic=True, force_col_wise=True)
        probe.fit(a, y)
        gain = probe.booster_.feature_importance("gain")
        self.selected_ = np.argsort(-gain, kind="stable")[:self.config.top_k]
        self.names_ = self.encoder_.get_feature_names_out()[self.selected_].tolist()
        return self

    @staticmethod
    def clean(x):
        x = x.copy().replace([np.inf, -np.inf], np.nan)
        for col in x.select_dtypes(exclude="number"):
            # Prefix observed strings so an actual category named '__MISSING__'
            # or 'MISSING:' cannot collide with the encoded null state.
            x[col] = ("VALUE:" + x[col].astype("string")).fillna("MISSING:")
        return x

    def transform(self, x):
        a = self.encoder_.transform(self.clean(x[self.columns_]))
        return a[:, self.selected_].astype(np.float32)


def model_matrix(a, family):
    # CatBoost's sparse missing-value behavior differs by version; selected
    # numeric features are dense and have already been imputed explicitly.
    return a.toarray() if family == "catboost" and sparse.issparse(a) else a


def train_model(family, complexity, a, y, config, validation=None, rounds=None):
    rounds = config.max_rounds if rounds is None else rounds
    common = dict(n_estimators=rounds, learning_rate=.04, random_state=config.seed,
                  n_jobs=config.threads)
    fit_kwargs = {}
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**common, num_leaves=complexity, metric="auc",
                                   verbosity=-1, deterministic=True, force_col_wise=True)
        if validation is not None:
            fit_kwargs = dict(eval_X=validation[0], eval_y=validation[1], eval_metric="auc",
                              callbacks=[lgb.early_stopping(config.patience, first_metric_only=True,
                                                            verbose=False)])
    elif family == "xgboost":
        from xgboost import XGBClassifier
        extra = {"early_stopping_rounds": config.patience} if validation is not None else {}
        model = XGBClassifier(**common, max_depth=int(np.ceil(np.log2(complexity))),
                              tree_method="hist", eval_metric="auc", **extra)
        if validation is not None:
            fit_kwargs = dict(eval_set=[validation], verbose=False)
    elif family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=rounds, learning_rate=.04,
                                   depth=int(np.ceil(np.log2(complexity))),
                                   random_seed=config.seed, thread_count=config.threads,
                                   verbose=False, allow_writing_files=False)
        if validation is not None:
            fit_kwargs = dict(eval_set=validation, early_stopping_rounds=config.patience,
                              use_best_model=True)
    else:
        raise ValueError(f"Unknown model family: {family}")
    model.fit(a, y, **fit_kwargs)
    if validation is None:
        selected_rounds = rounds
    elif family == "lightgbm":
        selected_rounds = model.best_iteration_ or rounds
    elif family == "xgboost":
        selected_rounds = model.best_iteration + 1
    else:
        selected_rounds = model.get_best_iteration() + 1
    return model, max(1, int(selected_rounds))


@dataclass
class Policy:
    representation: str
    family: str
    transform: FoldTransform
    model: object
    threshold: float
    metadata: dict

    def predict(self, frames):
        a = self.transform.transform(frames[self.representation])
        return self.model.predict_proba(model_matrix(a, self.family))[:, 1]


def validate_frames(frames, y, required=("raw", "application", "history")):
    y = binary(y)
    if not set(required).issubset(frames):
        raise ValueError(f"Missing predeclared representations: {sorted(set(required)-set(frames))}")
    reference = frames["raw"].index
    if not reference.is_unique:
        raise ValueError("Each row needs a unique audit index")
    for x in frames.values():
        if len(x) != len(y) or not x.index.equals(reference):
            raise ValueError("Representation row alignment mismatch")
    return y


def fit_policies(frames, y, config=Config(), groups=None, times=None):
    """Tune on fit/tune; refit on their union; calibrate on disjoint labels.

    No refit after calibration: the threshold is used with exactly the model
    whose probabilities were calibrated. This deliberately reserves data.
    """
    y = validate_frames(frames, y, config.representations)
    fit, tune, calibration = inner_partitions(y, config, groups, times)
    development = np.sort(np.r_[fit, tune])
    candidates = []
    specifications = [(r, "lightgbm") for r in config.representations]
    specifications += [("history", f) for f in config.families if f != "lightgbm"]
    for representation, family in specifications:
        x = frames[representation]
        transform = FoldTransform(config).fit(x.iloc[fit], y[fit])
        a, v = (model_matrix(transform.transform(x.iloc[i]), family) for i in [fit, tune])
        options = []
        for complexity in config.complexities:
            model, rounds = train_model(family, complexity, a, y[fit], config,
                                         validation=(v, y[tune]))
            auc = float(roc_auc_score(y[tune], model.predict_proba(v)[:, 1]))
            options.append(dict(complexity=int(complexity), rounds=rounds, tune_auc=auc))
        # Stable ties follow the predeclared simpler-first configuration order.
        choice = max(options, key=lambda option: option["tune_auc"])
        candidates.append(dict(representation=representation, family=family,
                               options=options, **choice))
    selected = max(candidates, key=lambda c: c["tune_auc"])
    policies = {}
    for choice in candidates:
        representation, family = choice["representation"], choice["family"]
        x = frames[representation]
        transform = FoldTransform(config).fit(x.iloc[development], y[development])
        a = model_matrix(transform.transform(x.iloc[development]), family)
        model, _ = train_model(family, choice["complexity"], a, y[development], config,
                               rounds=choice["rounds"])
        p = model.predict_proba(model_matrix(transform.transform(x.iloc[calibration]), family))[:, 1]
        threshold = choose_threshold(y[calibration], p, config.fn_cost, config.fp_cost)
        metadata = dict(**choice, threshold=encode_threshold(threshold),
                        fit_rows=frames["raw"].index[fit].tolist(),
                        tune_rows=frames["raw"].index[tune].tolist(),
                        calibration_rows=frames["raw"].index[calibration].tolist(),
                        refit_rows=frames["raw"].index[development].tolist(),
                        selected_features=transform.names_)
        policy = Policy(representation, family, transform, model, threshold, metadata)
        policies[f"{representation}_{family}"] = policy
        if choice is selected:
            policies["selected"] = policy
    return policies


def encode_threshold(threshold):
    return "+inf" if threshold == np.inf else "-inf" if threshold == -np.inf else float(threshold)


def nested_evaluate(frames, y, config=Config(), groups=None, times=None):
    y = validate_frames(frames, y, config.representations)
    outer = make_splits(y, config.outer_folds, config.seed,
                        config.split_mode, groups, times)
    records, prediction_rows, audit = [], [], []
    for fold, (train, validation) in enumerate(outer, start=1):
        print(f"Outer fold {fold}/{len(outer)}: train={len(train)}, validation={len(validation)}", flush=True)
        subset = lambda a: None if a is None else np.asarray(a)[train]
        policies = fit_policies({k: x.iloc[train] for k, x in frames.items()}, y[train],
                                config, subset(groups), subset(times))
        # Freeze all predictions and thresholds before reading outer labels.
        validation_frames = {k: x.iloc[validation] for k, x in frames.items()}
        cache = {}
        for policy in policies.values():
            if id(policy) not in cache:
                cache[id(policy)] = policy.predict(validation_frames)
        for name, policy in policies.items():
            p = cache[id(policy)]
            record = dict(fold=fold, policy=name, family=policy.family,
                          representation=policy.representation,
                          threshold=encode_threshold(policy.threshold),
                          **metrics(y[validation], p, policy.threshold, config))
            records.append(record)
            prediction_rows.append(pd.DataFrame(dict(
                row=frames["raw"].index[validation], fold=fold, policy=name,
                target=y[validation], probability=p, approve=p < policy.threshold)))
        audit.append(dict(fold=fold, outer_train=frames["raw"].index[train].tolist(),
                          outer_validation=frames["raw"].index[validation].tolist(),
                          policies={k: v.metadata for k, v in policies.items()}))
    folds = pd.DataFrame(records)
    summary = folds.groupby("policy").agg(
        mean_fold_auc=("auc", "mean"), std_fold_auc=("auc", "std"),
        mean_fold_ap=("ap", "mean"), evaluated_rows=("n", "sum"))
    for name, rows in folds.groupby("policy"):
        for col in ["cost", "approval_rate", "approve_all_cost", "reject_all_cost", "brier", "log_loss"]:
            summary.loc[name, col] = np.average(rows[col], weights=rows.n)
    # Paired differences, same model family, splits, budget and feature selection rule.
    paired = folds.pivot(index="fold", columns="policy", values="auc")
    differences = {}
    for left, right in [("application", "raw"), ("history", "application"), ("history", "raw")]:
        a, b = f"{left}_lightgbm", f"{right}_lightgbm"
        if a in paired and b in paired:
            differences[f"{left}_minus_{right}"] = paired[a] - paired[b]
    ablation = pd.DataFrame(differences, index=paired.index)
    return dict(summary=summary, folds=folds, ablation=ablation,
                predictions=pd.concat(prediction_rows, ignore_index=True), audit=audit)
