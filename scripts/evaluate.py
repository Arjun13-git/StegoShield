#!/usr/bin/env python
"""Evaluate a saved model artifact against a feature dataset split.

Standalone from train_models.py so a frozen artifact can be re-evaluated
against any dataset split (test, or later a robustness/external dataset)
without retraining. Also emits ROC-curve points for the frontend Research
tab.

Usage:
  python scripts/evaluate.py \\
      --model data/models/stegoshield_rf.joblib \\
      --features-csv data/processed/features_payload_0.10.csv \\
      --split test \\
      --report-out data/reports/evaluation_payload_0.10.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from app.ml.features import FEATURE_NAMES
from app.ml.training import evaluate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--report-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = joblib.load(args.model)

    bundle_features = bundle.get("feature_names")
    if bundle_features is not None and list(bundle_features) != list(FEATURE_NAMES):
        raise RuntimeError(
            "Model artifact's feature schema does not match the current "
            "extract_features() schema. Retrain before evaluating: "
            f"artifact has {len(bundle_features)} features, code expects {len(FEATURE_NAMES)}."
        )

    df = pd.read_csv(args.features_csv)
    subset = df[df["split"] == args.split]
    if subset.empty:
        raise ValueError(f"No rows found for split='{args.split}' in {args.features_csv}")

    x = subset[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = subset["label"].to_numpy(dtype=int)

    scaler = bundle.get("scaler")
    x_in = scaler.transform(x) if scaler is not None else x
    proba = bundle["model"].predict_proba(x_in)[:, 1]

    metrics = evaluate(y, proba)
    fpr, tpr, thresholds = roc_curve(y, proba)

    report = {
        "model": str(args.model),
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("model_version"),
        "features_csv": str(args.features_csv),
        "split": args.split,
        "metrics": metrics.to_dict(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }

    report_path = (
        Path(args.report_out)
        if args.report_out
        else Path("data/reports") / f"evaluation_{Path(args.features_csv).stem}_{args.split}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print(
        f"{args.split}: acc={metrics.accuracy:.3f} precision={metrics.precision:.3f} "
        f"recall={metrics.recall:.3f} f1={metrics.f1:.3f} auc={metrics.roc_auc:.3f}"
    )
    print(f"confusion_matrix={metrics.confusion_matrix}")
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
