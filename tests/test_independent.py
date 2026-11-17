from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from credit_scoring.independent import independent_evaluate, reserve_holdout
from credit_scoring.run import input_manifest, verify_inputs, write_plan


def test_holdout_labels_do_not_change_training_predictions_or_decisions(config, frames):
    x, y = frames
    partitions = reserve_holdout(x, y, config)
    development, holdout = partitions
    first = independent_evaluate(x, y, config, partitions=partitions)
    changed = y.copy()
    changed[holdout] = 1 - changed[holdout]
    second = independent_evaluate(x, changed, config, partitions=partitions)
    assert first["holdout_audit"] == second["holdout_audit"]
    pd.testing.assert_frame_equal(first["development"]["predictions"], second["development"]["predictions"])
    for col in ["probability", "approve"]:
        np.testing.assert_array_equal(first["holdout_predictions"][col], second["holdout_predictions"][col])
    assert not first["holdout_metrics"].auc.equals(second["holdout_metrics"].auc)
    for fold in first["development"]["audit"]:
        assert not set(holdout) & set(fold["outer_train"])
        assert not set(holdout) & set(fold["outer_validation"])
    for policy in first["policies"].values():
        for key in ["fit_rows", "tune_rows", "calibration_rows", "refit_rows"]:
            assert not set(holdout) & set(policy.metadata[key])
            assert set(policy.metadata[key]).issubset(set(development))


def test_holdout_plan_exclusive_and_label_free(config, frames, tmp_path):
    x, y = frames
    target = tmp_path / "run"
    partitions = write_plan(target, x, y, config, {"synthetic": True}, {})
    plan = json.loads((tmp_path / "run.plan.json").read_text())
    assert plan["primary_policy"] == "selected"
    assert plan["independent_holdout_rows"] == x["raw"].index[partitions[1]].tolist()
    assert "target" not in plan and "labels" not in plan
    assert not target.exists()
    with pytest.raises(FileExistsError):
        write_plan(target, x, y, config, {}, {})


def test_final_holdout_respects_time_and_borrowers(config, frames):
    x, y = frames
    # Constant timestamp blocks with both classes; last block also shares
    # borrowers with the first, which must be purged from development.
    y = np.tile([0, 1], len(y) // 2)
    times = np.repeat(np.arange(len(y) // 4), 4)
    groups = np.arange(len(y))
    groups[-4:] = groups[:4]
    cfg = replace(config, split_mode="time")
    dev, held = reserve_holdout(x, y, cfg, groups=groups, times=times)
    assert times[dev].max() < times[held].min()
    assert not set(groups[dev]) & set(groups[held])


def test_manual_holdout_cannot_bypass_borrower_isolation(config, frames):
    x, y = frames
    dev, held = np.arange(270), np.arange(270, 360)
    groups = np.arange(len(y))
    groups[-1] = groups[0]
    with pytest.raises(ValueError, match="Borrowers overlap"):
        independent_evaluate(x, y, replace(config, split_mode="group"), groups=groups,
                             partitions=(dev, held))


def test_source_mutation_invalidates_manifest(data_dir):
    manifest = input_manifest(data_dir)
    verify_inputs(data_dir, manifest)
    with (data_dir / "bureau.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="Input changed"):
        verify_inputs(data_dir, manifest)
