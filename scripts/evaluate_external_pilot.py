#!/usr/bin/env python
"""Phase 1B external (ALASKA2) evaluation of the FROZEN Phase 1A RandomForest.
No retraining, no scaler refit, no feature/model/threshold changes, and
nothing here is tuned on ALASKA2 results.

Unit of analysis: the SOURCE (one numbered ALASKA2 cover identity). The four
variants of a source (Cover/JMiPOD/JUNIWARD/UERD) are NOT independent; all
significance tests and confidence intervals below either resample whole
sources (paired bootstrap) or use per-source paired differences.

Experiments (kept strictly separate, never pooled):

  A. Clean-cover score distribution (no positive class -> never "accuracy"),
     overall and by JPEG QF, alongside the frozen model's IN-DOMAIN BOSSBase
     test-cover scores as a reference baseline.

  B. Controlled synthetic LSB cross-domain test. This is OUR OWN embed_lsb()
     (unmodified), NOT ALASKA2's native steganography. ALASKA2 covers are
     genuine 3-channel RGB, so embed_lsb() (single 2D channel) is applied
     independently to R, G and B with distinct deterministic seeds.
     payload 0.10 = primary (Phase 1A reference). payload 0.40 = supplementary,
     declared before results because 0.10 is near-chance in-domain.

  C. Dataset-provided JPEG stego, each method its own balanced comparison:
     Cover vs JMiPOD, Cover vs JUNIWARD, Cover vs UERD.

Usage:
  python scripts/evaluate_external_pilot.py \\
      --manifest data/external/ALASKA2/pilot_manifest.csv \\
      --report-out data/reports/phase1b_pilot_evaluation.json
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
from scipy import stats
from sklearn.metrics import roc_auc_score

from app.ml.features import FEATURE_NAMES, extract_features
from app.ml.lsb import embed_lsb, sample_seed
from app.ml.model import ModelBundle
from app.ml.training import evaluate as compute_metrics

LSB_PAYLOADS = (0.10, 0.40)  # 0.10 primary, 0.40 supplementary
PRIMARY_LSB_PAYLOAD = 0.10
METHODS = ("jmipod", "juniward", "uerd")
MIN_SOURCES_FOR_QF_BREAKDOWN = 20
BOOTSTRAP_RESAMPLES = 2000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/external/ALASKA2/pilot_manifest.csv")
    p.add_argument("--model", default="data/models/stegoshield_rf.joblib")
    p.add_argument("--features-dir", default="data/processed")
    p.add_argument("--report-out", default="data/reports/phase1b_pilot_evaluation.json")
    p.add_argument("--scores-out", default="data/reports/phase1b_alaska2_scores.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _feature_vector(image: Image.Image) -> np.ndarray:
    vector, names = extract_features(image)
    if tuple(names) != FEATURE_NAMES:
        raise RuntimeError("feature schema mismatch")
    if not np.isfinite(vector).all():
        raise ValueError("non-finite feature vector")
    return vector


def _load_rgb_array(path: str) -> np.ndarray:
    with Image.open(path) as img:
        img.load()
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def _lsb_stego_rgb(rgb: np.ndarray, stem: str, payload: float, seed: int) -> np.ndarray:
    out = np.empty_like(rgb)
    for c, label in enumerate(("R", "G", "B")):
        out[..., c] = embed_lsb(rgb[..., c], payload, sample_seed(seed, f"{stem}_{label}", payload))
    return out


def _predict(bundle: ModelBundle, x: np.ndarray) -> np.ndarray:
    scaler = bundle.bundle.get("scaler")
    x_in = scaler.transform(x) if scaler is not None else x
    return bundle.bundle["model"].predict_proba(x_in)[:, 1]


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return [float(centre - half), float(centre + half)]


def _describe(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {"n": 0}
    k = int((scores >= 0.5).sum())
    return {
        "n": int(len(scores)),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "fraction_ge_0.5": k / len(scores),
        "fraction_ge_0.5_wilson95": _wilson(k, len(scores)),
    }


def _paired_analysis(cover: np.ndarray, stego: np.ndarray, rng: np.random.Generator) -> dict:
    """Balanced cover-vs-stego metrics plus source-level (paired) statistics."""
    n = len(cover)
    y = np.array([0] * n + [1] * n)
    proba = np.concatenate([cover, stego])
    metrics = compute_metrics(y, proba).to_dict()
    diff = stego - cover

    boot_auc, boot_acc = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)  # resample whole sources (paired)
        pr = np.concatenate([cover[idx], stego[idx]])
        boot_auc.append(roc_auc_score(y, pr))
        boot_acc.append(float(((pr >= 0.5).astype(int) == y).mean()))

    try:
        w = stats.wilcoxon(diff)
        wilcoxon = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
    except ValueError:
        wilcoxon = {"note": "all paired differences are zero"}

    return {
        "n_source_pairs": n,
        "n_cover": n,
        "n_stego": n,
        "metrics": metrics,
        "roc_auc_bootstrap95_over_sources": [float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5))],
        "accuracy_bootstrap95_over_sources": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
        "cover_fpr_at_0.5": float((cover >= 0.5).mean()),
        "stego_tpr_at_0.5": float((stego >= 0.5).mean()),
        "cover_score": _describe(cover),
        "stego_score": _describe(stego),
        "paired_score_diff_stego_minus_cover": {
            "mean": float(diff.mean()),
            "median": float(np.median(diff)),
            "fraction_stego_gt_cover": float((diff > 0).mean()),
            "fraction_ties": float((diff == 0).mean()),
            "wilcoxon_signed_rank": wilcoxon,
        },
    }


def _by_qf(qf: np.ndarray, cover: np.ndarray, stego: np.ndarray, rng: np.random.Generator) -> dict:
    out = {}
    for level in (75, 90, 95):
        mask = qf == level
        if mask.sum() >= MIN_SOURCES_FOR_QF_BREAKDOWN:
            out[f"QF{level}"] = _paired_analysis(cover[mask], stego[mask], rng)
        else:
            out[f"QF{level}"] = {"n_source_pairs": int(mask.sum()),
                                  "note": f"fewer than {MIN_SOURCES_FOR_QF_BREAKDOWN} sources; not reported"}
    return out


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    wall_start = time.perf_counter()
    hash_before = _sha256(args.model)

    bundle = ModelBundle(args.model)
    bundle.load()
    if not bundle.loaded:
        raise RuntimeError(f"Could not load frozen model artifact at {args.model}")
    if list(bundle.bundle["feature_names"]) != list(FEATURE_NAMES):
        raise RuntimeError("Model artifact's feature schema does not match the current extractor.")

    df = pd.read_csv(args.manifest, dtype={"source_id": str})
    covers = df[df["variant"] == "cover"].sort_values("source_id").reset_index(drop=True)
    paths = {m: df[df["variant"] == m].set_index("source_id")["filepath"].to_dict() for m in METHODS}
    print(f"Manifest: {len(df)} rows, {len(covers)} sources")

    # ---- one feature-extraction pass per image ---------------------------
    columns = ["cover", *METHODS, *[f"lsb_{p:.2f}" for p in LSB_PAYLOADS]]
    feats: dict[str, dict[str, np.ndarray]] = {c: {} for c in columns}
    failures: list[dict] = []
    t_feat = 0.0
    n_images = 0

    for _, row in covers.iterrows():
        sid = row["source_id"]
        try:
            t0 = time.perf_counter()
            rgb = _load_rgb_array(row["filepath"])
            local: dict[str, np.ndarray] = {"cover": _feature_vector(Image.fromarray(rgb))}
            for m in METHODS:
                local[m] = _feature_vector(Image.fromarray(_load_rgb_array(paths[m][sid])))
            for p in LSB_PAYLOADS:
                stego = _lsb_stego_rgb(rgb, Path(row["filepath"]).stem, p, args.seed)
                local[f"lsb_{p:.2f}"] = _feature_vector(Image.fromarray(stego))
            t_feat += time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_id": sid, "error": str(exc)[:200]})
            continue
        n_images += len(local)
        for c, v in local.items():
            feats[c][sid] = v

    kept = [sid for sid in covers["source_id"] if sid in feats["cover"]]
    meta = covers.set_index("source_id").loc[kept]
    qf = meta["jpeg_quality_factor"].to_numpy()
    splits = meta["split"].to_numpy()

    t_inf0 = time.perf_counter()
    scores = {c: _predict(bundle, np.vstack([feats[c][sid] for sid in kept])) for c in columns}
    t_inf = time.perf_counter() - t_inf0

    pd.DataFrame({"source_id": kept, "split": splits, "qf": qf, **scores}).to_csv(args.scores_out, index=False)

    report: dict = {
        "model": args.model,
        "model_name": bundle.bundle.get("model_name"),
        "manifest": args.manifest,
        "unit_of_analysis": "source (numbered ALASKA2 cover identity); variants of one source are not independent",
        "n_sources_in_manifest": int(len(covers)),
        "n_sources_scored": len(kept),
        "n_failed_sources": len(failures),
        "failures": failures,
        "qf_counts_sources": {str(k): int(v) for k, v in pd.Series(qf).value_counts().sort_index().items()},
        "split_counts_sources": {k: int(v) for k, v in pd.Series(splits).value_counts().items()},
        "note_on_splits": ("No model is trained or tuned on ALASKA2, so all sources are evaluated; the "
                           "split column exists only to keep any future ALASKA2 calibration experiment leakage-safe."),
    }

    # ---- in-domain reference (BOSSBase test split, frozen model) ---------
    boss = pd.read_csv(Path(args.features_dir) / "features_payload_0.10.csv")
    boss_test = boss[boss["split"] == "test"]
    boss_cover_scores = _predict(bundle, boss_test[boss_test["label"] == 0][list(FEATURE_NAMES)].to_numpy(dtype=np.float64))
    boss_stego_scores = _predict(bundle, boss_test[boss_test["label"] == 1][list(FEATURE_NAMES)].to_numpy(dtype=np.float64))
    report["in_domain_reference_bossbase_test_payload_0.10"] = {
        "cover_score": _describe(boss_cover_scores),
        "stego_score": _describe(boss_stego_scores),
        "note": "Same frozen model on the Phase 1A held-out BOSSBase test split (in-domain).",
    }

    # ---- Experiment A -----------------------------------------------------
    a = scores["cover"]
    ks = stats.ks_2samp(a, boss_cover_scores)
    mw = stats.mannwhitneyu(a, boss_cover_scores, alternative="two-sided")
    groups = {f"QF{lv}": a[qf == lv] for lv in (75, 90, 95)}
    kw = stats.kruskal(*[g for g in groups.values() if len(g) > 0])
    pair = {}
    for x, y_ in (("QF75", "QF90"), ("QF75", "QF95"), ("QF90", "QF95")):
        u = stats.mannwhitneyu(groups[x], groups[y_], alternative="two-sided")
        pair[f"{x}_vs_{y_}"] = {"mann_whitney_U": float(u.statistic), "p_value_uncorrected": float(u.pvalue),
                                "common_language_effect_size_P(first>second)": float(u.statistic / (len(groups[x]) * len(groups[y_])))}
    rho = stats.spearmanr(qf.astype(float), a)
    report["experiment_A_clean_cover_domain_shift"] = {
        "note": "Score distribution only; there is no positive class, so this is NOT accuracy.",
        "overall": _describe(a),
        "by_qf": {k: _describe(v) for k, v in groups.items()},
        "vs_bossbase_test_covers": {
            "alaska2_n": int(len(a)), "bossbase_n": int(len(boss_cover_scores)),
            "ks_2samp_statistic": float(ks.statistic), "ks_p_value": float(ks.pvalue),
            "mann_whitney_p_value": float(mw.pvalue),
            "median_diff_alaska2_minus_bossbase": float(np.median(a) - np.median(boss_cover_scores)),
        },
        "qf_association": {
            "kruskal_wallis_H": float(kw.statistic), "kruskal_wallis_p": float(kw.pvalue),
            "pairwise_mann_whitney_p_uncorrected_3_comparisons": pair,
            "spearman_rho_qf_vs_score": float(rho.statistic), "spearman_p": float(rho.pvalue),
            "caveat": "Descriptive; n per QF group is ~50 sources, p-values are uncorrected, no causal claim.",
        },
    }

    # ---- Experiment B (controlled synthetic LSB; not ALASKA2-native) -----
    report["experiment_B_controlled_synthetic_lsb"] = {
        "note": ("Generated by OUR embed_lsb() (unmodified), applied independently per R/G/B channel at the "
                 "stated bits-per-pixel PER CHANNEL. This is NOT ALASKA2's native steganography."),
        "primary_payload": PRIMARY_LSB_PAYLOAD,
        "by_payload": {},
    }
    for p in LSB_PAYLOADS:
        s = scores[f"lsb_{p:.2f}"]
        report["experiment_B_controlled_synthetic_lsb"]["by_payload"][f"{p:.2f}"] = {
            "role": "primary" if p == PRIMARY_LSB_PAYLOAD else "supplementary (declared before results)",
            "overall": _paired_analysis(a, s, rng),
            "by_qf": _by_qf(qf, a, s, rng),
        }

    # ---- Experiment C (dataset-provided stego, separate) ------------------
    report["experiment_C_dataset_stego"] = {}
    for m in METHODS:
        report["experiment_C_dataset_stego"][m] = {
            "overall": _paired_analysis(a, scores[m], rng),
            "by_qf": _by_qf(qf, a, scores[m], rng),
        }

    # ---- feature-level shift vs BOSSBase (descriptive only) ---------------
    boss_cov = boss[(boss["label"] == 0) & (boss["split"] == "train")]
    alaska_cov = np.vstack([feats["cover"][sid] for sid in kept]).astype(np.float64)
    importances = bundle.bundle["model"].feature_importances_
    shift = {}
    for j, name in enumerate(FEATURE_NAMES):
        bm, bs = float(boss_cov[name].mean()), float(boss_cov[name].std())
        am, as_ = float(alaska_cov[:, j].mean()), float(alaska_cov[:, j].std(ddof=1))
        shift[name] = {
            "bossbase_train_cover_mean": bm, "bossbase_std": bs,
            "alaska2_cover_mean": am, "alaska2_std": as_,
            "standardized_mean_diff_in_bossbase_std": (am - bm) / bs if bs > 0 else None,
            "rf_feature_importance": float(importances[j]),
            **{f"alaska2_mean_QF{lv}": float(alaska_cov[qf == lv, j].mean()) for lv in (75, 90, 95)},
        }
    report["feature_shift_descriptive"] = {
        "note": "Descriptive comparison only. Feature importance alone does not establish what drives scores.",
        "features": shift,
    }

    report["runtime_seconds"] = {
        "feature_extraction_incl_decode_and_lsb": round(t_feat, 2),
        "images_feature_extracted": n_images,
        "seconds_per_image": round(t_feat / max(n_images, 1), 4),
        "batch_inference": round(t_inf, 3),
        "total_wall": round(time.perf_counter() - wall_start, 1),
    }
    report["integrity"] = {
        "model_sha256_before": hash_before,
        "model_sha256_after": _sha256(args.model),
    }
    report["integrity"]["model_unchanged"] = report["integrity"]["model_sha256_before"] == report["integrity"]["model_sha256_after"]

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2, default=str))
    print(f"Scored {len(kept)}/{len(covers)} sources; failures={len(failures)}")
    print(f"Wrote {args.report_out} and {args.scores_out}")


if __name__ == "__main__":
    main()
