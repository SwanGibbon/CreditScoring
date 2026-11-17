from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import credit_scoring.evaluation as evaluation
from credit_scoring.evaluation import FoldTransform, fit_policies, nested_evaluate
from credit_scoring.protocol import choose_threshold
from credit_scoring.run import save_evaluation


def fixed_partitions(n):
    return np.arange(n // 2), np.arange(n // 2, 3 * n // 4), np.arange(3 * n // 4, n)


def test_unknown_validation_category_and_signal_never_enter_transform(config):
    x = pd.DataFrame({"a": np.arange(200), "category": ["train", "other"] * 100,
                      "heldout_signal": [0.] * 200, "mostly_missing": [np.nan] * 199 + [1.]})
    transform = FoldTransform(config).fit(x, np.tile([0, 1], 100))
    heldout = x.iloc[:4].copy()
    heldout["category"], heldout["heldout_signal"] = "outer-only", 999
    before = list(transform.names_)
    assert transform.transform(heldout).shape[0] == 4
    assert transform.names_ == before
    assert "heldout_signal" not in transform.columns_
    assert "mostly_missing" not in transform.columns_
    assert not any("outer-only" in name for name in transform.names_)


def test_true_sentinels_are_distinct_from_missing_values(config):
    x = pd.DataFrame({"number": [-999., np.nan] * 80,
                      "category": ["__MISSING__", None] * 80})
    transform = FoldTransform(config).fit(x, np.tile([0, 1], 80))
    a = transform.transform(x)
    if hasattr(a, "toarray"):
        a = a.toarray()
    assert not np.array_equal(a[0], a[1])
    assert any("missing" in n for n in transform.names_)


def test_calibration_labels_only_change_threshold(config, frames, monkeypatch):
    x, y = frames
    fit, tune, cal = fixed_partitions(len(y))
    monkeypatch.setattr(evaluation, "inner_partitions", lambda *args: (fit, tune, cal))
    first = fit_policies(x, y, config)
    changed = y.copy()
    changed[cal] = 1 - changed[cal]
    second = fit_policies(x, changed, config)
    assert first["selected"].representation == second["selected"].representation
    for name, policy in first.items():
        other = second[name]
        assert policy.metadata["selected_features"] == other.metadata["selected_features"]
        assert policy.metadata["rounds"] == other.metadata["rounds"]
        np.testing.assert_array_equal(policy.predict(x), other.predict(x))
        p = policy.predict({k: v.iloc[cal] for k, v in x.items()})
        assert policy.threshold == choose_threshold(y[cal], p, config.fn_cost, config.fp_cost)
        assert other.threshold == choose_threshold(changed[cal], p, config.fn_cost, config.fp_cost)
        assert not set(policy.metadata["refit_rows"]) & set(cal)


def test_outer_label_changes_cannot_change_fit_or_predictions(config, frames, monkeypatch):
    x, y = frames
    train, val = np.arange(270), np.arange(270, 360)
    monkeypatch.setattr(evaluation, "make_splits", lambda *args: [(train, val)])
    first = nested_evaluate(x, y, config)
    changed = y.copy()
    changed[val] = 1 - changed[val]
    second = nested_evaluate(x, changed, config)
    assert first["audit"] == second["audit"]
    np.testing.assert_array_equal(first["predictions"].probability,
                                  second["predictions"].probability)
    np.testing.assert_array_equal(first["predictions"].approve, second["predictions"].approve)
    assert not first["folds"].auc.equals(second["folds"].auc)


def test_end_to_end_reproducibility_audit_and_outputs(config, frames, tmp_path):
    x, y = frames
    first, second = nested_evaluate(x, y, config), nested_evaluate(x, y, config)
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    assert first["audit"] == second["audit"]
    assert len(first["ablation"]) == config.outer_folds
    for _, predictions in first["predictions"].groupby("policy"):
        assert predictions.row.nunique() == len(y)
        assert len(predictions) == len(y)
    for fold in first["audit"]:
        outer = set(fold["outer_validation"])
        for policy in fold["policies"].values():
            for key in ["fit_rows", "tune_rows", "calibration_rows", "refit_rows"]:
                assert not outer & set(policy[key])
            fit, tune, cal = (set(policy[k]) for k in ["fit_rows", "tune_rows", "calibration_rows"])
            assert not fit & tune and not fit & cal and not tune & cal
            assert set(policy["refit_rows"]) == fit | tune
    out = tmp_path / "run"
    save_evaluation(first, out, config, {"synthetic": True}, {})
    report = json.loads((out / "report.json").read_text())
    assert report["data_audit"]["synthetic"]
    assert report["environment"]["packages"]["lightgbm"]
    assert len(list(out.iterdir())) == 5
    with pytest.raises(FileExistsError):
        save_evaluation(first, out, config, {}, {})


@pytest.mark.parametrize("family", ["xgboost", "catboost"])
def test_optional_family_inner_stopping_and_refit(family, config, frames, tmp_path):
    import joblib
    pytest.importorskip(family)
    config = replace(config, families=("lightgbm", family), complexities=(7,))
    x, y = frames
    policies = fit_policies(x, y, config)
    policy = policies[f"history_{family}"]
    p = policy.predict(x)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    assert 1 <= policy.metadata["rounds"] <= config.max_rounds
    model_path = tmp_path / "policy.joblib"
    joblib.dump(policy, model_path)
    np.testing.assert_array_equal(joblib.load(model_path).predict(x), p)
