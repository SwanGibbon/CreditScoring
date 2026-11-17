import json

import joblib
import numpy as np
import pandas as pd
import pytest

from credit_scoring.run import main
from credit_scoring.features import build_features
from credit_scoring.application_development import main as development_main


def test_cli_missing_data_stops_without_metrics(tmp_path, capsys):
    out = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        main(["--data", str(tmp_path), "--output", str(out), "--check-data"])
    assert error.value.code == 2
    assert capsys.readouterr().err.count(".csv") == 8
    assert not out.exists()


@pytest.mark.parametrize("data_dir", [360], indirect=True)
def test_csv_to_cli_to_submission_full_smoke(data_dir):
    out = data_dir / "output"
    args = ["--data", str(data_dir), "--output", str(out), "--outer-folds", "2",
            "--max-rounds", "8", "--threads", "1", "--chunksize", "53", "--submission"]
    main(args)
    submission = pd.read_csv(out / "submission.csv")
    expected = pd.read_csv(data_dir / "application_test.csv")
    assert submission.SK_ID_CURR.tolist() == expected.SK_ID_CURR.tolist()
    assert submission.TARGET.between(0, 1).all()
    report = json.loads((out / "report.json").read_text())
    assert len(report["inputs"]) == 8
    assert all(len(v["sha256"]) == 64 for v in report["inputs"].values())
    policy = json.loads((out / "final_policy.json").read_text())["policy"]
    assert not set(policy["calibration_rows"]) & set(policy["refit_rows"])
    independent = report["independent_holdout"]
    assert not set(independent["holdout_rows"]) & set(policy["refit_rows"])
    assert policy == independent["final_policies"]["selected"]
    assert (data_dir / "output.plan.json").is_file()
    assert (out / "holdout_metrics.csv").is_file()
    restored = joblib.load(out / "final_policy.joblib")
    _, _, test_frames, _, _ = build_features(data_dir)
    np.testing.assert_allclose(restored.predict(test_frames), submission.TARGET.to_numpy(), atol=1e-12)
    with pytest.raises(SystemExit):
        main(args)  # never overwrite an existing report


@pytest.mark.parametrize("data_dir", [360], indirect=True)
def test_application_development_reserves_but_never_scores_final_holdout(data_dir):
    out = data_dir / "application-development"
    development_main(["--data", str(data_dir), "--output", str(out)])
    plan = json.loads((data_dir / "application-development.plan.json").read_text())
    held = set(plan["independent_holdout_rows"])
    predictions = pd.read_csv(out / "predictions.csv")
    assert not held & set(predictions.row)
    assert not (out / "holdout_metrics.csv").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["data_audit"]["independent_final_evaluation_completed"] is False
    assert not report["data_audit"]["history_available"]
    assert set(predictions.policy) == {"raw_lightgbm", "application_lightgbm", "selected"}
    assert list(pd.read_csv(out / "ablation.csv").columns) == ["fold", "application_minus_raw"]
