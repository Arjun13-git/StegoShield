#!/usr/bin/env python
"""Phase 1B pilot model-compatibility checks against the FROZEN Phase 1A
RandomForest artifact. No retraining, no feature/model/scaler changes.

Three experiments, kept strictly separate (never pooled):

  A. Clean-cover domain-shift control: ALASKA2 covers only, no embedding.
     Reports a SCORE distribution only -- there is no positive class here,
     so this is never reported as "accuracy".

  B. Matched-methodology cross-domain LSB test: ALASKA2 covers with the
     EXISTING, UNMODIFIED backend/app/ml/lsb.py embed_lsb() applied at
     payload=0.10 (Phase 1A's reference payload). ALASKA2 covers are
     genuine 3-channel RGB (unlike Phase 1A's grayscale-replicated
     BOSSBase), so embed_lsb() -- which requires a single 2D channel -- is
     called independently on each of R, G, B with distinct deterministic
     seeds. This is a documented methodological choice, not a modification
     to embed_lsb() itself.

  C. Dataset-provided JPEG steganography: JMiPOD, JUNIWARD, UERD, each
     evaluated as its OWN separate balanced cover-vs-stego comparison
     (35 cover / 35 stego each) -- never combined into one pooled score.

Usage:
  python scripts/evaluate_external_pilot.py \\
      --manifest data/external/ALASKA2/pilot_manifest.csv \\
      --model data/models/stegoshield_rf.joblib \\
      --report-out data/reports/phase1b_pilot_evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.ml.features import FEATURE_NAMES, extract_features
from app.ml.lsb import sample_seed, embed_lsb
from app.ml.model import ModelBundle
from app.ml.training import evaluate as compute_metrics

LSB_PAYLOAD = 0.10  # matches Phase 1A's reference training payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/external/ALASKA2/pilot_manifest.csv")
    parser.add_argument("--model", default="data/models/stegoshield_rf.joblib")
    parser.add_argument("--report-out", default="data/reports/phase1b_pilot_evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _score(bundle: ModelBundle, image_path: str) -> tuple[float | None, str | None]:
    try:
        with Image.open(image_path) as img:
            img.load()
            rgb = img.convert("RGB")
        vector, names = extract_features(rgb)
        if tuple(names) != FEATURE_NAMES:
            return None, "feature_schema_mismatch"
        if not np.isfinite(vector).all():
            return None, "non_finite_feature_vector"
        score = bundle.predict_score(vector)
        return score, None
    except Exception as exc:  # noqa: BLE001
        return None, f"error: {exc}"


def _score_lsb_variant(bundle: ModelBundle, cover_path: str, seed: int) -> tuple[float | None, str | None]:
    try:
        with Image.open(cover_path) as img:
            img.load()
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None, f"unexpected_shape_{rgb.shape}"

        stego = np.empty_like(rgb)
        for c, channel_label in enumerate(("R", "G", "B")):
            channel_seed = sample_seed(seed, f"{Path(cover_path).stem}_{channel_label}", LSB_PAYLOAD)
            stego[..., c] = embed_lsb(rgb[..., c], LSB_PAYLOAD, channel_seed)

        stego_img = Image.fromarray(stego)
        vector, names = extract_features(stego_img)
        if tuple(names) != FEATURE_NAMES:
            return None, "feature_schema_mismatch"
        if not np.isfinite(vector).all():
            return None, "non_finite_feature_vector"
        score = bundle.predict_score(vector)
        return score, None
    except Exception as exc:  # noqa: BLE001
        return None, f"error: {exc}"


def _distribution_stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    arr = np.array(scores)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.manifest, dtype={"source_id": str})

    bundle = ModelBundle(args.model)
    bundle.load()
    if not bundle.loaded:
        raise RuntimeError(f"Could not load frozen model artifact at {args.model}")

    bundle_features = bundle.bundle.get("feature_names")
    if bundle_features is not None and list(bundle_features) != list(FEATURE_NAMES):
        raise RuntimeError("Model artifact's feature schema does not match the current extract_features() schema.")

    cover_rows = df[df["variant"] == "cover"].sort_values("source_id")
    print(f"Loaded manifest: {len(df)} rows, {len(cover_rows)} cover sources")

    report: dict = {
        "model": str(args.model),
        "model_name": bundle.bundle.get("model_name"),
        "manifest": str(args.manifest),
        "lsb_payload": LSB_PAYLOAD,
    }

    # --- Experiment A: clean-cover domain-shift control -------------------
    a_scores: list[float] = []
    a_failures: list[dict] = []
    for _, row in cover_rows.iterrows():
        score, err = _score(bundle, row["filepath"])
        if err is not None:
            a_failures.append({"source_id": row["source_id"], "error": err})
        else:
            a_scores.append(score)
    report["experiment_A_clean_cover_domain_shift"] = {
        "note": "Score distribution only. No positive class exists here -- NOT an accuracy measurement.",
        "n_attempted": len(cover_rows),
        "n_scored": len(a_scores),
        "n_failed": len(a_failures),
        "failures": a_failures,
        "score_distribution": _distribution_stats(a_scores),
    }
    print(f"Experiment A: scored {len(a_scores)}/{len(cover_rows)} covers")

    # --- Experiment B: matched-methodology LSB (our embed_lsb, per-channel) ---
    b_cover_scores: list[float] = []
    b_stego_scores: list[float] = []
    b_failures: list[dict] = []
    for _, row in cover_rows.iterrows():
        cover_score, cover_err = _score(bundle, row["filepath"])
        stego_score, stego_err = _score_lsb_variant(bundle, row["filepath"], args.seed)
        if cover_err is not None or stego_err is not None:
            b_failures.append({"source_id": row["source_id"], "cover_error": cover_err, "stego_error": stego_err})
            continue
        b_cover_scores.append(cover_score)
        b_stego_scores.append(stego_score)

    if b_cover_scores and b_stego_scores:
        y_true = np.array([0] * len(b_cover_scores) + [1] * len(b_stego_scores))
        y_proba = np.array(b_cover_scores + b_stego_scores)
        b_metrics = compute_metrics(y_true, y_proba).to_dict()
    else:
        b_metrics = None

    report["experiment_B_matched_lsb"] = {
        "note": (
            f"Our own embed_lsb() (unmodified) applied independently per R/G/B channel at "
            f"payload={LSB_PAYLOAD} bpp/channel, since ALASKA2 covers are genuine 3-channel RGB "
            "(unlike Phase 1A's grayscale-replicated BOSSBase training data). This is a documented "
            "methodological choice, not a change to embed_lsb() itself."
        ),
        "n_attempted": len(cover_rows),
        "n_scored_pairs": len(b_cover_scores),
        "n_failed": len(b_failures),
        "failures": b_failures,
        "cover_score_distribution": _distribution_stats(b_cover_scores),
        "stego_score_distribution": _distribution_stats(b_stego_scores),
        "balanced_metrics": b_metrics,
    }
    print(f"Experiment B: scored {len(b_cover_scores)}/{len(cover_rows)} matched cover/stego pairs")

    # --- Experiment C: dataset-provided JPEG stego, kept separate ---------
    report["experiment_C_dataset_stego"] = {}
    for method in ("jmipod", "juniward", "uerd"):
        method_rows = df[df["variant"] == method].sort_values("source_id")
        cover_by_source = {row["source_id"]: row["filepath"] for _, row in cover_rows.iterrows()}

        cover_scores: list[float] = []
        stego_scores: list[float] = []
        failures: list[dict] = []

        for _, row in method_rows.iterrows():
            sid = row["source_id"]
            cover_path = cover_by_source.get(sid)
            if cover_path is None:
                failures.append({"source_id": sid, "error": "no_matching_cover_row_in_manifest"})
                continue
            cscore, cerr = _score(bundle, cover_path)
            sscore, serr = _score(bundle, row["filepath"])
            if cerr is not None or serr is not None:
                failures.append({"source_id": sid, "cover_error": cerr, "stego_error": serr})
                continue
            cover_scores.append(cscore)
            stego_scores.append(sscore)

        if cover_scores and stego_scores:
            y_true = np.array([0] * len(cover_scores) + [1] * len(stego_scores))
            y_proba = np.array(cover_scores + stego_scores)
            metrics = compute_metrics(y_true, y_proba).to_dict()
        else:
            metrics = None

        report["experiment_C_dataset_stego"][method] = {
            "n_attempted": len(method_rows),
            "n_scored_pairs": len(cover_scores),
            "n_failed": len(failures),
            "failures": failures,
            "cover_score_distribution": _distribution_stats(cover_scores),
            "stego_score_distribution": _distribution_stats(stego_scores),
            "balanced_metrics": metrics,
        }
        print(f"Experiment C [{method}]: scored {len(cover_scores)}/{len(method_rows)} balanced pairs")

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote evaluation report to {report_path}")


if __name__ == "__main__":
    main()
