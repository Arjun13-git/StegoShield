#!/usr/bin/env python
"""Payload-sensitivity experiment.

Takes ONE frozen, already-trained model artifact (trained at a single
reference payload level, as produced by train_models.py) and measures how
its test-split detection performance changes across every payload level's
feature file. This answers "how well does our one deployed detector do
against different embedding strengths?" rather than training a separate
model per payload level, which better matches how a single model is
actually deployed in the API. The trend across payload levels is measured,
not assumed.

Usage:
  python scripts/generate_payload_report.py \\
      --model data/models/stegoshield_rf.joblib \\
      --features-dir data/processed \\
      --payloads 0.01,0.05,0.10,0.20,0.40 \\
      --report-out data/reports/payload_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_NAMES
from app.ml.lsb import DEFAULT_PAYLOAD_LEVELS
from app.ml.training import evaluate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--payloads", default=",".join(str(p) for p in DEFAULT_PAYLOAD_LEVELS))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--report-out", default="data/reports/payload_sensitivity.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = joblib.load(args.model)

    bundle_features = bundle.get("feature_names")
    if bundle_features is not None and list(bundle_features) != list(FEATURE_NAMES):
        raise RuntimeError(
            "Model artifact's feature schema does not match the current extract_features() schema."
        )

    scaler = bundle.get("scaler")
    model = bundle["model"]
    payload_levels = [float(p) for p in args.payloads.split(",") if p]
    features_dir = Path(args.features_dir)

    results = []
    for payload in payload_levels:
        csv_path = features_dir / f"features_payload_{payload:.2f}.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping payload={payload}")
            continue

        df = pd.read_csv(csv_path)
        subset = df[df["split"] == args.split]
        x = subset[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        y = subset["label"].to_numpy(dtype=int)

        x_in = scaler.transform(x) if scaler is not None else x
        proba = model.predict_proba(x_in)[:, 1]
        metrics = evaluate(y, proba)

        results.append({"payload": payload, "features_csv": str(csv_path), **metrics.to_dict()})
        print(
            f"payload={payload:.2f}: acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
            f"recall={metrics.recall:.3f} auc={metrics.roc_auc:.3f} (n={metrics.n_samples})"
        )

    report = {
        "model": str(args.model),
        "model_name": bundle.get("model_name"),
        "split_evaluated": args.split,
        "note": (
            "A single frozen model artifact (trained once, at its own reference "
            "payload) was evaluated against each payload level's held-out test "
            "split. This measures deployed-model sensitivity to embedding "
            "strength, not a separately-trained-per-payload comparison."
        ),
        "results": results,
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved payload sensitivity report to {report_path}")


if __name__ == "__main__":
    main()
