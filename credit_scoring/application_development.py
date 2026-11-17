"""Real application-only development CV while history/final evaluation is pending.

This explicitly reserves but NEVER scores the final holdout. It must not be
presented as the full Home Credit pipeline or as independent final performance.
"""
import argparse
from pathlib import Path

from .evaluation import nested_evaluate
from .features import build_application_features
from .protocol import Config
from .run import input_manifest, save_evaluation, verify_inputs, write_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="home-credit-default-risk")
    parser.add_argument("--output", default="artifacts/application-development-001")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Choose a new output path; existing results cannot be overwritten")
    config = Config(outer_folds=3, max_rounds=300, patience=30, probe_rounds=60,
                    threads=3, representations=("raw", "application"))
    manifest = input_manifest(args.data, ("application_train", "application_test"))
    frames, y, _, _, audit = build_application_features(args.data)
    verify_inputs(args.data, manifest)
    audit.update(scope="application-only development nested CV; final holdout never scored",
                 independent_final_evaluation_completed=False)
    development, holdout = write_plan(args.output, frames, y, config, audit, manifest)
    print("Reserved final holdout rows (not scored):", len(holdout), flush=True)
    print("Frozen development config:", config.as_dict(), flush=True)
    result = nested_evaluate({k: x.iloc[development] for k, x in frames.items()}, y[development], config)
    verify_inputs(args.data, manifest)
    save_evaluation(result, args.output, config, audit, manifest)
    print("DEVELOPMENT ONLY — not full history, not independent final performance")
    print(result["summary"].to_string())


if __name__ == "__main__":
    main()
