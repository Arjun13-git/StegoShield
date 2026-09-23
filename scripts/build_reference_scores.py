#!/usr/bin/env python
"""Build the API's reference file from the Phase 1A dataset and the frozen model.

  * cover-score reference (risk percentiles): VALIDATION-split clean covers
  * per-feature reference statistics (explanations): TRAIN-split covers/stego
  * flag rates on the TEST split are stored for description only; nothing is
    tuned on them.

The frozen model is only used for inference (`predict_proba`); it is never
modified. The output is bound to the model by SHA-256.

Usage:
  python scripts/build_reference_scores.py \\
      --features-csv data/processed/features_payload_0.10.csv \\
      --model data/models/stegoshield_rf.joblib \\
      --out backend/app/resources/reference_scores_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.services.reference import build_reference


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-csv", default="data/processed/features_payload_0.10.csv")
    p.add_argument("--model", default="data/models/stegoshield_rf.joblib")
    p.add_argument("--out", default="backend/app/resources/reference_scores_v1.json")
    args = p.parse_args()

    bundle = joblib.load(args.model)
    if list(bundle["feature_names"]) != list(FEATURE_NAMES):
        raise RuntimeError("model feature schema does not match the current extractor")
    model, scaler = bundle["model"], bundle.get("scaler")

    def predict_proba(x: np.ndarray) -> np.ndarray:
        x_in = scaler.transform(x) if scaler is not None else x
        return model.predict_proba(x_in)[:, 1]

    df = pd.read_csv(args.features_csv)
    cols = list(FEATURE_NAMES)

    def part(split: str, label: int) -> np.ndarray:
        return df[(df["split"] == split) & (df["label"] == label)][cols].to_numpy(dtype=np.float64)

    ref = build_reference(
        model_sha256=_sha256(Path(args.model)),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=cols,
        predict_proba=predict_proba,
        train_x_cover=part("train", 0), train_x_stego=part("train", 1),
        val_x_cover=part("val", 0), val_x_stego=part("val", 1),
        test_x_cover=part("test", 0), test_x_stego=part("test", 1),
        payload_note="Stego reference = Phase 1A controlled LSB replacement at 0.10 bits per pixel (BOSSBase, grayscale).",
        description=("Reference for interpreting the uncalibrated StegoShield model score. Cover-score percentiles come from the "
                     "Phase 1A validation split, per-feature statistics from the train split; test-split flag rates are descriptive only."),
    )
    ref["source"] = {"features_csv": args.features_csv, "features_csv_sha256": _sha256(Path(args.features_csv)),
                     "generator": "scripts/build_reference_scores.py"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ref, indent=1, sort_keys=True))
    print(json.dumps({"out": args.out, "n": ref["n"], "operating_points": ref["operating_points"]}, indent=2))


if __name__ == "__main__":
    main()
