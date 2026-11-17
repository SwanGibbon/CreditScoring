"""CLI: preflight, bounded nested evaluation, optional final submission."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import pandas as pd

from .features import REQUIRED_FILES, build_features, check_files, check_schema
from .independent import independent_evaluate, reserve_holdout
from .protocol import Config


def environment():
    versions = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def input_manifest(directory, names=REQUIRED_FILES):
    manifest = {}
    for name in names:
        path = Path(directory) / f"{name}.csv"
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed while hashing: {path}")
        manifest[path.name] = dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                                   sha256=digest.hexdigest())
    return manifest


def verify_inputs(directory, manifest):
    for name, expected in manifest.items():
        stat = (Path(directory) / name).stat()
        if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise RuntimeError(f"Input changed after hashing: {name}; start a new experiment")


def source_manifest():
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "credit_scoring").glob("*.py")) + sorted(root.glob("requirements*.txt"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def write_plan(directory, frames, y, config, data_audit, manifest,
               holdout_folds=5, groups=None, times=None):
    """Exclusive, pre-training plan; never stores holdout labels."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Output already exists: {directory}")
    plan_path = directory.parent / (directory.name + ".plan.json")
    partitions = reserve_holdout(frames, y, config, holdout_folds, groups, times)
    development, holdout = partitions
    plan = dict(config=config.as_dict(), inputs=manifest, data_audit=data_audit,
                sources=source_manifest(), environment=environment(),
                primary_policy="selected", independent_holdout_folds=holdout_folds,
                split_seed=config.seed + 17,
                development_rows=frames["raw"].index[development].tolist(),
                independent_holdout_rows=frames["raw"].index[holdout].tolist())
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("x") as handle:
        handle.write(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    return partitions


def save_evaluation(result, directory, config, data_audit, manifest, independent=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for key in ["summary", "folds", "ablation", "predictions"]:
        result[key].to_csv(directory / f"{key}.csv", index=key in {"summary", "ablation"})
    report = dict(config=config.as_dict(), environment=environment(), sources=source_manifest(),
                  data_audit=data_audit, inputs=manifest, fold_audit=result["audit"],
                  interpretation="Evaluate the predeclared selected policy; do not pick a winner from outer scores")
    if independent is not None:
        independent["holdout_metrics"].to_csv(directory / "holdout_metrics.csv", index=True)
        independent["holdout_predictions"].to_csv(directory / "holdout_predictions.csv", index=False)
        report["independent_holdout"] = independent["holdout_audit"]
        policy = independent["policies"]["selected"]
        model_path = directory / "final_policy.joblib"
        joblib.dump(policy, model_path, compress=3)
        (directory / "final_policy.json").write_text(json.dumps(
            dict(config=config.as_dict(), policy=policy.metadata), indent=2, allow_nan=False) + "\n")
        report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="home-credit-default-risk")
    parser.add_argument("--output", default="artifacts/run-42")
    parser.add_argument("--check-data", action="store_true", help="Check required files without loading tables")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--holdout-folds", type=int, default=5,
                        help="Reserve 1/N as independent final holdout (default 1/5)")
    parser.add_argument("--max-rounds", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--probe-rounds", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--families", nargs="+", default=["lightgbm"])
    parser.add_argument("--submission", action="store_true", help="Predict test with the exact assessed development-only policy; never refit on final holdout")
    args = parser.parse_args(argv)
    try:
        check_files(args.data)
    except FileNotFoundError as error:
        parser.exit(2, f"{error}\nNo metrics calculated. Obtain data under your existing access rights.\n")
    if args.check_data:
        check_schema(args.data)
        print("All eight CSV files and required headers are present; content/ownership checks run during loading.")
        return
    if Path(args.output).exists():
        parser.error("Output directory already exists; choose a new --output to preserve the previous run")
    config = Config(seed=args.seed, outer_folds=args.outer_folds, max_rounds=args.max_rounds,
                    patience=args.patience, probe_rounds=args.probe_rounds, top_k=args.top_k,
                    threads=args.threads, families=tuple(args.families))
    print("Config:", config.as_dict(), flush=True)
    manifest = input_manifest(args.data)
    frames, y, test, ids, audit = build_features(args.data, args.chunksize)
    verify_inputs(args.data, manifest)
    partitions = write_plan(args.output, frames, y, config, audit, manifest, args.holdout_folds)
    experiment = independent_evaluate(frames, y, config, args.holdout_folds, partitions=partitions)
    result = experiment["development"]
    verify_inputs(args.data, manifest)
    save_evaluation(result, args.output, config, audit, manifest, independent=experiment)
    print("Development nested CV:")
    print(result["summary"].to_string())
    print("Paired LightGBM ablation:\n", result["ablation"].to_string())
    print("Independent final holdout (descriptive comparisons; primary policy = selected):")
    print(experiment["holdout_metrics"].to_string())
    if args.submission:
        # Use the exact assessed development-only policy; do not refit on holdout.
        policy = experiment["policies"]["selected"]
        pd.DataFrame({"SK_ID_CURR": ids, "TARGET": policy.predict(test)}).to_csv(
            Path(args.output) / "submission.csv", index=False)


if __name__ == "__main__":
    main()
