"""Export an explicit allowlist of aggregate results for a public repository.

No prediction rows, targets, split indices, model bytes, private links, or local
paths are exported. The input is an already completed local experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def records(path):
    return pd.read_csv(path, float_precision="round_trip").to_dict("records")


def export(run_directory, output, memory_gib=None):
    run = Path(run_directory)
    report = json.loads((run / "report.json").read_text())
    method = json.loads((run / "auc-bootstrap/method.json").read_text())
    source = run / "holdout_predictions.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != method["input_sha256"]:
        raise ValueError("Holdout predictions differ from the bootstrap input")
    holdout = records(run / "holdout_metrics.csv")
    intervals = records(run / "auc-bootstrap/auc_intervals.csv")
    interval_map = {r["policy"]: r for r in intervals}
    for row in holdout:
        if abs(row["auc"] - interval_map[row["policy"]]["auc"]) > 1e-12:
            raise ValueError("Point AUC differs from bootstrap provenance")
    audit_path = run.parent / "full-data-audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    process_path = run.parent / (run.name + ".process.json")
    process = json.loads(process_path.read_text()) if process_path.exists() else {}
    verification_path = run / "verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    final = report["independent_holdout"]
    result = {
        "schema_version": 1,
        "run_id": run.name,
        "run_date_utc": process.get("started_utc", "").split("T")[0] or None,
        "scope": "Home Credit Default Risk; fixed independent IID application holdout",
        "primary_policy": "selected",
        "selected_representation": final["final_policies"]["selected"]["representation"],
        "dataset": {
            "development_rows": len(final["development_rows"]),
            "holdout_rows": len(final["holdout_rows"]),
            "test_rows": report["data_audit"]["test_rows"],
            "final_model_fit_rows": len(final["final_policies"]["selected"]["refit_rows"]),
            "threshold_calibration_rows": len(final["final_policies"]["selected"]["calibration_rows"]),
            "known_row_counts": audit.get("known_row_counts", {}),
            "inputs": {name: {k: value[k] for k in ["bytes", "sha256"]}
                       for name, value in report["inputs"].items()},
            "unmatched_bureau_balance_rows": report["data_audit"]["unknown_owner_rows"]["bureau_balance"],
        },
        "config": report["config"],
        "environment": report["environment"],
        "runtime_seconds": process.get("elapsed_seconds"),
        "memory_gib": memory_gib,
        "verification": {k: verification[k] for k in ["tests_passed", "notebook_code_cells_executed",
            "holdout_metrics_recomputed_from_saved_predictions", "model_sha256_and_metadata_verified",
            "submission_original_id_order_verified"] if k in verification},
        "holdout_metrics": holdout,
        "auc_intervals": intervals,
        "paired_auc_differences": records(run / "auc-bootstrap/paired_auc_differences.csv"),
        "development_summary": records(run / "summary.csv"),
        "development_folds": records(run / "folds.csv"),
        "development_ablation": records(run / "ablation.csv"),
        "bootstrap": {k: method[k] for k in ["method", "replicates", "seed", "confidence_level",
                                             "rows", "positive_rows", "interpretation",
                                             "input_sha256", "analysis_code_sha256"]},
        "provenance": {
            "training_source_sha256": report["sources"],
            "model_sha256": report["model_sha256"],
            "local_report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
            "note": "Training snapshot predates a CLI help wording fix. Fitting code is unchanged; hashes identify the recorded training version.",
        },
        "limitations": [
            "No absolute application timestamps: forward-time performance is untested.",
            "Distinct application IDs cannot exclude hidden repeat borrowers.",
            "Bootstrap conditions on fixed fitted models and class counts; it excludes retraining uncertainty and drift.",
            "TARGET denotes competition payment difficulties, not necessarily legal default.",
            "Cost 10:1 is a hypothetical scenario, not measured monetary loss.",
            "Holdout is now observed; future changes need fresh independent confirmation.",
            "XGBoost and CatBoost had synthetic tests only, not this full real-data benchmark.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--output", default="reports/home-credit-full-001.json")
    parser.add_argument("--memory-gib", type=int, help="Verified RAM of the benchmark machine, when known")
    args = parser.parse_args()
    exported = export(args.run_directory, args.output, args.memory_gib)
    print(f"Exported aggregate report {exported['run_id']} to {args.output}")
