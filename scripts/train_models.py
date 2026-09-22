#!/usr/bin/env python
"""Train and compare LR/SVM/RF on a generated feature dataset, then freeze
and save the selected model as an inference-ready artifact.

Model-selection rule (documented up front, not chosen post hoc): among the
three candidates, select the one with the highest validation F1 score
(rounded to 3 decimal places so noise-level differences don't dominate),
breaking ties by validation ROC-AUC. The selected model is evaluated on the
test split exactly once, after selection is frozen. This mirrors
System-Design/04-ml-design.md section 7 and avoids tuning against the test
set.

Usage:
  python scripts/train_models.py \\
      --features-csv data/processed/features_payload_0.10.csv \\
      --seed 42 \\
      --model-out data/models/stegoshield_rf.joblib \\
      --report-out data/reports/model_comparison_payload_0.10.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.ml.training import NEEDS_SCALING, build_candidate_models, evaluate, fit_model, predict_proba


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", default="data/models/stegoshield_rf.joblib")
    parser.add_argument("--report-out", default=None, help="Defaults to data/reports/model_comparison_<features-csv-stem>.json")
    return parser.parse_args()


def _load_split(df: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    subset = df[df["split"] == split]
    x = subset[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = subset["label"].to_numpy(dtype=int)
    return x, y


def main() -> None:
    args = _parse_args()
    features_path = Path(args.features_csv)
    report_path = Path(args.report_out) if args.report_out else Path("data/reports") / f"model_comparison_{features_path.stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_path)
    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature CSV is missing expected columns: {missing_cols}")

    x_train, y_train = _load_split(df, "train")
    x_val, y_val = _load_split(df, "val")
    x_test, y_test = _load_split(df, "test")
    print(f"train={len(y_train)} (pos={int(y_train.sum())}) "
          f"val={len(y_val)} (pos={int(y_val.sum())}) "
          f"test={len(y_test)} (pos={int(y_test.sum())})")

    candidates = build_candidate_models(seed=args.seed)
    comparison: dict[str, dict] = {}
    trained_models = {}

    for name, estimator in candidates.items():
        trained = fit_model(name, estimator, x_train, y_train, needs_scaling=NEEDS_SCALING[name])
        trained_models[name] = trained

        train_proba = predict_proba(trained, x_train)
        val_proba = predict_proba(trained, x_val)
        train_metrics = evaluate(y_train, train_proba)
        val_metrics = evaluate(y_val, val_proba)

        comparison[name] = {
            "train_seconds": trained.train_seconds,
            "train_metrics": train_metrics.to_dict(),
            "val_metrics": val_metrics.to_dict(),
        }
        print(
            f"{name}: train_acc={train_metrics.accuracy:.3f} "
            f"val_acc={val_metrics.accuracy:.3f} val_f1={val_metrics.f1:.3f} "
            f"val_auc={val_metrics.roc_auc:.3f} (fit {trained.train_seconds:.2f}s)"
        )
        gap = train_metrics.accuracy - val_metrics.accuracy
        if gap > 0.15:
            print(f"  NOTE: train/val accuracy gap = {gap:.3f} -- possible overfitting, investigate before trusting this model.")

    # F1 is rounded before comparing so two candidates within ~0.001 of each
    # other (noise-level on a few thousand validation samples) are treated
    # as tied and the decision falls through to ROC-AUC, which is a more
    # robust, threshold-independent measure of separability. Comparing raw
    # floats would let a coin-flip-sized F1 difference override a
    # meaningfully better AUC.
    selected_name = max(
        comparison,
        key=lambda n: (round(comparison[n]["val_metrics"]["f1"], 3), comparison[n]["val_metrics"]["roc_auc"]),
    )
    print(f"\nSelected model (by validation F1 [rounded to 3dp], tie-break ROC-AUC): {selected_name}")

    selected = trained_models[selected_name]
    test_proba = predict_proba(selected, x_test)
    test_metrics = evaluate(y_test, test_proba)
    print(
        f"Frozen test evaluation for {selected_name}: "
        f"acc={test_metrics.accuracy:.3f} f1={test_metrics.f1:.3f} auc={test_metrics.roc_auc:.3f}"
    )
    if test_metrics.accuracy >= 0.98:
        print(
            "  WARNING: test accuracy >= 0.98. Per CLAUDE.md ML quality rules, investigate for "
            "leakage/duplication/artifacts before treating this as a credible result."
        )

    bundle = {
        "model": selected.estimator,
        "model_name": selected_name,
        "model_version": "v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "training_dataset_id": features_path.stem,
        "random_seed": args.seed,
        "scaler": selected.scaler,
        "metrics": {
            "val": comparison[selected_name]["val_metrics"],
            "test": test_metrics.to_dict(),
        },
        "package_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    joblib.dump(bundle, args.model_out)
    print(f"Saved model artifact to {args.model_out}")

    report = {
        "features_csv": str(features_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "seed": args.seed,
        "dataset_counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "candidates": comparison,
        "selected_model": selected_name,
        "selection_rule": "max validation F1, tie-break validation ROC-AUC",
        "test_metrics": test_metrics.to_dict(),
        "package_versions": bundle["package_versions"],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved comparison report to {report_path}")


if __name__ == "__main__":
    main()
