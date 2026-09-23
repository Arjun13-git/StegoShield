#!/usr/bin/env python
"""Phase 1B robustness stress test of the FROZEN Phase 1A detector on the
BOSSBase TEST split.

Design (fixed before any run; not tuned against results):

  original cover  (BOSSBase test source, grayscale pixels)
  original stego  (Phase 1A embed_lsb at payload p, seed = sample_seed(42, source_id, p))
        |                                   |
        +-------- SAME transformation ------+
        |                                   |
   transformed cover                  transformed stego
        +---------- frozen RF (feature extractor + model unchanged) ----------+

  * The transformation is applied identically to the cover and its stego
    derivative. For Gaussian noise the same noise field (same seed) is used
    for both members of a pair.
  * Every transformed sample keeps its original source_id and split=test;
    a source contributes exactly one cover and one stego row per
    (transformation, payload) condition. Conditions are evaluated
    separately and never pooled, and no derived image is treated as a new
    independent source.
  * Nothing is retrained or refit. The artifact hash is recorded before
    and after.
  * The identity condition ("none") must reproduce the stored Phase 1A
    results; this is checked and reported, not assumed.

Transformation parameters (pre-declared, see backend/app/ml/robustness.py):
  jpeg_qf90, jpeg_qf75, resize 0.9x (bicubic, not resized back),
  gaussian noise sigma=2.0 (8-bit levels), crop box (13,13,493,493).

Payloads: 0.10 is the PRIMARY condition (Phase 1A reference payload).
0.40 is a pre-declared SUPPLEMENTARY condition: at 0.10 the detector is
already near chance in-domain (Phase 1A test ROC-AUC 0.553), leaving little
headroom to observe degradation; 0.40 is the largest Phase 1A payload
(in-domain ROC-AUC 0.695). This choice was made from Phase 1A results.

Usage:
  python scripts/robustness_eval.py --limit 30   # small pilot
  python scripts/robustness_eval.py               # all 1500 test sources
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.ml.external import select_pilot_sources
from app.ml.features import FEATURE_NAMES, extract_features
from app.ml.lsb import embed_lsb, sample_seed
from app.ml.model import ModelBundle
from app.ml.robustness import build_suite_v1
from app.ml.training import evaluate as compute_metrics

PAYLOADS = (0.10, 0.40)
IDENTITY = "none"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bossbase-root", default="data/raw/BOSSBase")
    p.add_argument("--split-manifest", default="data/processed/split_manifest.csv")
    p.add_argument("--features-dir", default="data/processed")
    p.add_argument("--model", default="data/models/stegoshield_rf.joblib")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="deterministic seeded subset of test sources (pilot runs)")
    p.add_argument("--report-out", default="data/reports/phase1b_robustness.json")
    p.add_argument("--scores-out", default="data/reports/phase1b_robustness_scores.csv")
    return p.parse_args()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _features(arr: np.ndarray) -> np.ndarray:
    # Mirrors Phase 1A's generate_dataset.py exactly: pixels -> RGB image -> extract_features
    vector, names = extract_features(Image.fromarray(arr).convert("RGB"))
    if tuple(names) != FEATURE_NAMES:
        raise RuntimeError("feature schema mismatch")
    if not np.isfinite(vector).all():
        raise ValueError("non-finite feature vector")
    return vector


def _predict(bundle: ModelBundle, x: np.ndarray) -> np.ndarray:
    scaler = bundle.bundle.get("scaler")
    x_in = scaler.transform(x) if scaler is not None else x
    return bundle.bundle["model"].predict_proba(x_in)[:, 1]


def main() -> None:
    args = _parse_args()
    wall_start = time.perf_counter()
    hash_before = _sha256(args.model)

    bundle = ModelBundle(args.model)
    bundle.load()
    if list(bundle.bundle["feature_names"]) != list(FEATURE_NAMES):
        raise RuntimeError("Model artifact feature schema does not match the current extractor.")

    sm = pd.read_csv(args.split_manifest, dtype={"source_id": str})
    test_ids = sorted(sm.loc[sm["split"] == "test", "source_id"])
    total_test_sources = len(test_ids)
    if args.limit is not None:
        test_ids = select_pilot_sources(test_ids, args.limit, args.seed)
    print(f"Test sources: {len(test_ids)} of {total_test_sources} (seed={args.seed})")

    transforms = [None] + build_suite_v1()  # None = identity
    names = [IDENTITY] + [t.name for t in transforms[1:]]

    feats_cover: dict[str, list[np.ndarray]] = {n: [] for n in names}
    feats_stego: dict[tuple[str, float], list[np.ndarray]] = {(n, p): [] for n in names for p in PAYLOADS}
    kept_ids: dict[str, list[str]] = {n: [] for n in names}
    failures: dict[str, list[dict]] = {n: [] for n in names}
    t_transform = {n: 0.0 for n in names}
    t_features = {n: 0.0 for n in names}
    content_hashes: dict[str, str] = {}

    for i, sid in enumerate(test_ids, start=1):
        with Image.open(Path(args.bossbase_root) / f"{sid}.pgm") as img:
            img.load()
            pixels = np.asarray(img, dtype=np.uint8).copy()
        content_hashes[sid] = hashlib.md5(pixels.tobytes()).hexdigest()
        stegos = {p: embed_lsb(pixels, p, sample_seed(args.seed, sid, p)) for p in PAYLOADS}
        noise_seed = sample_seed(args.seed, f"{sid}_robustness_noise", 0.0)

        for name, t in zip(names, transforms):
            try:
                t0 = time.perf_counter()
                if t is None:
                    cover_t, stego_t = pixels, stegos
                else:
                    cover_t = t.apply(pixels, noise_seed)
                    stego_t = {p: t.apply(stegos[p], noise_seed) for p in PAYLOADS}
                t1 = time.perf_counter()
                vc = _features(cover_t)
                vs = {p: _features(stego_t[p]) for p in PAYLOADS}
                t2 = time.perf_counter()
            except Exception as exc:  # noqa: BLE001
                failures[name].append({"source_id": sid, "error": str(exc)[:200]})
                continue
            t_transform[name] += t1 - t0
            t_features[name] += t2 - t1
            feats_cover[name].append(vc)
            for p in PAYLOADS:
                feats_stego[(name, p)].append(vs[p])
            kept_ids[name].append(sid)

        if i % 250 == 0 or i == len(test_ids):
            print(f"  processed {i}/{len(test_ids)} sources ({time.perf_counter() - wall_start:.0f}s)")

    # ---- batch inference, metrics, paired statistics -----------------------
    results: dict[str, dict] = {}
    score_rows: list[dict] = []
    infer_start = time.perf_counter()
    for name in names:
        if not kept_ids[name]:
            results[name] = {"n_source_pairs": 0}
            continue
        xc = np.vstack(feats_cover[name])
        pc = _predict(bundle, xc)
        per_payload = {}
        for p in PAYLOADS:
            xs = np.vstack(feats_stego[(name, p)])
            ps = _predict(bundle, xs)
            y = np.array([0] * len(pc) + [1] * len(ps))
            m = compute_metrics(y, np.concatenate([pc, ps])).to_dict()
            diff = ps - pc
            per_payload[f"{p:.2f}"] = {
                "metrics": m,
                "cover_fpr_at_0.5": float((pc >= 0.5).mean()),
                "stego_tpr_at_0.5": float((ps >= 0.5).mean()),
                "mean_cover_score": float(pc.mean()),
                "mean_stego_score": float(ps.mean()),
                "paired_mean_score_diff_stego_minus_cover": float(diff.mean()),
                "paired_fraction_stego_gt_cover": float((diff > 0).mean()),
                "paired_fraction_ties": float((diff == 0).mean()),
            }
            for sid, c, s in zip(kept_ids[name], pc, ps):
                score_rows.append({"source_id": sid, "transform": name, "payload": p,
                                   "cover_score": float(c), "stego_score": float(s)})
        results[name] = {
            "n_source_pairs": len(kept_ids[name]),
            "n_failed_sources": len(failures[name]),
            "failures": failures[name][:10],
            "seconds_transform": round(t_transform[name], 2),
            "seconds_feature_extraction": round(t_features[name], 2),
            "by_payload": per_payload,
        }
    infer_seconds = time.perf_counter() - infer_start

    # ---- reproduction of stored Phase 1A results (identity condition) ----
    reproduction: dict = {}
    if not failures[IDENTITY] and args.limit is None:
        stored = json.loads(Path("data/reports/evaluation_payload_0.10_test.json").read_text())["metrics"]
        got = results[IDENTITY]["by_payload"]["0.10"]["metrics"]
        reproduction["payload_0.10_max_abs_metric_diff"] = max(
            abs(got[k] - stored[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
        )
        reproduction["payload_0.10_confusion_matrix_equal"] = got["confusion_matrix"] == stored["confusion_matrix"]
        sens = {round(r["payload"], 2): r for r in json.loads(Path("data/reports/payload_sensitivity.json").read_text())["results"]}
        got4 = results[IDENTITY]["by_payload"]["0.40"]["metrics"]
        reproduction["payload_0.40_max_abs_metric_diff"] = max(
            abs(got4[k] - sens[0.4][k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
        )
        reproduction["payload_0.40_confusion_matrix_equal"] = got4["confusion_matrix"] == sens[0.4]["confusion_matrix"]
    # feature-level reproduction against Phase 1A CSV (identity condition)
    csv = pd.read_csv(Path(args.features_dir) / "features_payload_0.10.csv", dtype={"source_id": str})
    csv = csv[csv["split"] == "test"].set_index("sample_id")
    kept = kept_ids[IDENTITY]
    xc = np.vstack(feats_cover[IDENTITY])
    xs = np.vstack(feats_stego[(IDENTITY, 0.10)])
    ref_c = csv.loc[[f"{s}_cover" for s in kept], list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    ref_s = csv.loc[[f"{s}_stego_p0.10" for s in kept], list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    reproduction["identity_feature_max_abs_diff_vs_phase1a_csv"] = float(
        max(np.abs(xc.astype(np.float64) - ref_c).max(), np.abs(xs.astype(np.float64) - ref_s).max())
    )

    # ---- integrity checks -------------------------------------------------
    sm_split = sm.set_index("source_id")["split"]
    hashes = list(content_hashes.values())
    integrity = {
        "n_sources_evaluated": len(test_ids),
        "all_sources_in_test_split": bool((sm_split.loc[test_ids] == "test").all()),
        "sources_in_train_or_val": int((sm_split.loc[test_ids] != "test").sum()),
        "duplicate_content_within_evaluated_sources": len(hashes) - len(set(hashes)),
        "known_phase1a_duplicate_ids_present": sorted(set(test_ids) & {"5536", "5537", "5538", "5539"}),
        "rows_per_source_per_condition": "1 cover + 1 stego (conditions evaluated separately, never pooled)",
        "model_sha256_before": hash_before,
    }
    integrity["model_sha256_after"] = _sha256(args.model)
    integrity["model_unchanged"] = integrity["model_sha256_before"] == integrity["model_sha256_after"]

    report = {
        "model": args.model,
        "seed": args.seed,
        "payloads": list(PAYLOADS),
        "primary_payload": 0.10,
        "transforms": [{"name": IDENTITY, "kind": "identity", "params": {}}] + [t.describe() for t in transforms[1:]],
        "n_test_sources_total": total_test_sources,
        "n_test_sources_evaluated": len(test_ids),
        "results": results,
        "phase1a_reproduction_check": reproduction,
        "integrity": integrity,
        "runtime_seconds": {"total": round(time.perf_counter() - wall_start, 1), "batch_inference": round(infer_seconds, 2)},
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    pd.DataFrame(score_rows).to_csv(args.scores_out, index=False)

    print("\nreproduction:", json.dumps(reproduction))
    print("integrity:", json.dumps(integrity))
    for name in names:
        r = results[name]
        if not r.get("n_source_pairs"):
            print(f"{name}: no successful pairs")
            continue
        for p in PAYLOADS:
            m = r["by_payload"][f"{p:.2f}"]["metrics"]
            print(f"{name:<24} p={p:.2f} n={r['n_source_pairs']} acc={m['accuracy']:.3f} "
                  f"f1={m['f1']:.3f} auc={m['roc_auc']:.3f} rec={m['recall']:.3f}")
    print(f"\nWrote {args.report_out} and {args.scores_out}")


if __name__ == "__main__":
    main()
