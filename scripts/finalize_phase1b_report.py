#!/usr/bin/env python
"""Close Phase 1B: verify, independently recompute, audit, and render the
final report from the generated artifacts. Read-only with respect to the
model, the datasets and the earlier reports; it starts no new experiment.

  1. Verifies the frozen Phase 1A artifact and that Phase 1A source files are
     byte-identical to git HEAD.
  2. Re-derives the ALASKA2 dataset integrity facts from the files on disk
     (decode, dimensions, JPEG structure, SHA-256 duplicates, QF, pairing,
     split integrity).
  3. Independently recomputes the ALASKA2 and robustness metrics from the
     saved per-source scores (rank-based AUC, Wilcoxon, KS, Kruskal-Wallis)
     and compares them with the stored reports.
  4. Applies Holm multiple-comparison correction and Bonferroni-level
     intervals to the robustness deltas.
  5. Spot-checks end-to-end determinism of the robustness pipeline on a
     small deterministic subset.
  6. Re-runs two DESCRIPTIVE diagnostics that were previously executed
     ad hoc (paired feature changes; univariate feature AUC under resize)
     so the report can cite reproducible numbers.
  7. Writes data/reports/phase1b_final_report.{md,json}. Exits non-zero if
     any verification check fails.

Usage (from the repository root, project venv):
  python scripts/finalize_phase1b_report.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import PIL
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from PIL import Image
from scipy import stats

from app.ml.evalstats import auc_rank, holm_adjust, paired_auc_bootstrap
from app.ml.external import MANIFEST_COLUMNS, inspect_jpeg_quality, select_pilot_sources
from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features
from app.ml.lsb import embed_lsb, sample_seed
from app.ml.robustness import build_suite_v1
from app.ml.split import verify_split_integrity

EXPECTED_SHA256 = "34707f83dc385e0846be3df7c56da930aad951a23fd3daa032a050449d02523d"
EXPECTED_SIZE = 1695049
MODEL = Path("data/models/stegoshield_rf.joblib")
REPORTS = Path("data/reports")
ALASKA = Path("data/external/ALASKA2")
MANIFEST = ALASKA / "pilot_manifest.csv"
VARIANT_DIRS = {"cover": "Cover", "jmipod": "JMiPOD", "juniward": "JUNIWARD", "uerd": "UERD"}
FROZEN_PHASE1A_FILES = [
    "backend/app/ml/features.py", "backend/app/ml/lsb.py", "backend/app/ml/dataset.py",
    "backend/app/ml/split.py", "backend/app/ml/training.py", "backend/app/ml/model.py",
    "scripts/generate_dataset.py", "scripts/train_models.py", "scripts/evaluate.py",
    "scripts/generate_payload_report.py",
]
PAYLOADS = (0.10, 0.40)
N_BOOT_CI = 4000

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
    return bool(ok)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _feat(arr: np.ndarray) -> np.ndarray:
    return extract_features(Image.fromarray(arr).convert("RGB"))[0]


# --------------------------------------------------------------------------
# A. Frozen model and Phase 1A code
# --------------------------------------------------------------------------


def verify_model() -> tuple[dict, object]:
    sha = sha256_file(MODEL)
    check("model SHA-256 equals audited Phase 1A hash", sha == EXPECTED_SHA256, sha)
    check("model size is 1,695,049 bytes", MODEL.stat().st_size == EXPECTED_SIZE, str(MODEL.stat().st_size))
    bundle = joblib.load(MODEL)
    params = bundle["model"].get_params()
    expect = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20, "random_state": 42, "class_weight": "balanced"}
    check("RandomForest hyperparameters as documented", all(params[k] == v for k, v in expect.items()), json.dumps({k: params[k] for k in expect}))
    check("scaler is None (no scaler refit possible)", bundle["scaler"] is None)
    check("artifact feature_names == current FEATURE_NAMES", list(bundle["feature_names"]) == list(FEATURE_NAMES))
    check("artifact n_features_in_ == 22", int(bundle["model"].n_features_in_) == 22)
    check("artifact training_dataset_id is features_payload_0.10", bundle["training_dataset_id"] == "features_payload_0.10")
    tracked = git("ls-files", "--error-unmatch", *FROZEN_PHASE1A_FILES)
    diff = git("diff", "--quiet", "HEAD", "--", *FROZEN_PHASE1A_FILES)
    check("Phase 1A source files are tracked and identical to git HEAD", tracked.returncode == 0 and diff.returncode == 0)
    import re

    phase1b_code = [
        "backend/app/ml/external.py", "backend/app/ml/robustness.py", "backend/app/ml/evalstats.py",
        "scripts/download_alaska2_pilot.py", "scripts/inspect_alaska2.py", "scripts/prepare_external_dataset.py",
        "scripts/evaluate_external_pilot.py", "scripts/robustness_eval.py", "scripts/analyze_robustness.py",
    ]
    fit_hits = [f for f in phase1b_code if re.search(r"\.fit(_transform)?\(", Path(f).read_text())]
    check("no fit()/fit_transform() call in any Phase 1B module or script (cannot retrain or refit a scaler)", not fit_hits, str(fit_hits))
    info = {
        "sha256": sha, "size_bytes": MODEL.stat().st_size, "model_name": bundle["model_name"],
        "model_version": bundle["model_version"], "feature_schema_version": bundle["feature_schema_version"],
        "random_seed": bundle["random_seed"], "training_dataset_id": bundle["training_dataset_id"],
        "hyperparameters": {k: params[k] for k in expect}, "n_features": 22,
        "feature_names_sha256": hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest(),
        "git_head": git("rev-parse", "HEAD").stdout.strip(),
    }
    return info, bundle


# --------------------------------------------------------------------------
# B. ALASKA2 dataset integrity (re-derived from files on disk)
# --------------------------------------------------------------------------


def verify_dataset() -> dict:
    disk = {v: sorted(p.stem for p in (ALASKA / d).glob("*.jpg")) for v, d in VARIANT_DIRS.items()}
    sources = sorted(disk["cover"])
    check("each variant directory holds the same source IDs", all(disk[v] == sources for v in disk), {v: len(disk[v]) for v in disk}.__str__())
    check("151 complete source groups on disk", len(sources) == 151)

    fmt, mode, size, ntab, sub = {}, {}, {}, {}, {}
    raw_hashes, pix_hashes, failures = [], [], []
    qf_by_source: dict[str, set] = {s: set() for s in sources}
    n_files = 0
    from PIL import JpegImagePlugin

    for v, d in VARIANT_DIRS.items():
        for sid in sources:
            p = ALASKA / d / f"{sid}.jpg"
            n_files += 1
            try:
                with Image.open(p) as img:
                    img.load()
                    fmt[img.format] = fmt.get(img.format, 0) + 1
                    mode[img.mode] = mode.get(img.mode, 0) + 1
                    size[f"{img.size[0]}x{img.size[1]}"] = size.get(f"{img.size[0]}x{img.size[1]}", 0) + 1
                    ntab[len(img.quantization)] = ntab.get(len(img.quantization), 0) + 1
                    code = JpegImagePlugin.get_sampling(img)
                    sub[code] = sub.get(code, 0) + 1
                    pix_hashes.append(hashlib.sha256(np.asarray(img).tobytes()).hexdigest())
            except Exception as exc:  # noqa: BLE001
                failures.append((str(p), str(exc)[:80]))
                continue
            raw_hashes.append(sha256_file(p))
            qf_by_source[sid].add(inspect_jpeg_quality(p).matched_known_qf)

    check("604 image files, 0 decode failures", n_files == 604 and not failures, f"files={n_files} failures={len(failures)}")
    check("all files JPEG / RGB / 512x512", fmt == {"JPEG": 604} and mode == {"RGB": 604} and size == {"512x512": 604}, f"{fmt} {mode} {size}")
    check("all files carry exactly 2 JPEG quantization tables", ntab == {2: 604}, str(ntab))
    check("no exact byte-duplicate files (SHA-256 of file bytes)", len(set(raw_hashes)) == len(raw_hashes) == 604)
    check("no exact decoded-pixel duplicates (SHA-256 of pixels)", len(set(pix_hashes)) == len(pix_hashes) == 604)
    consistent = all(len(s) == 1 and None not in s for s in qf_by_source.values())
    check("every source has one QF (75/90/95) across its 4 variants, none unmatched", consistent)
    qf_counts = {int(q): sum(1 for s in qf_by_source.values() if q in s) for q in (75, 90, 95)}
    check("QF source counts sum to 151", sum(qf_counts.values()) == 151, str(qf_counts))

    m = pd.read_csv(MANIFEST, dtype={"source_id": str})
    check("manifest columns equal MANIFEST_COLUMNS", tuple(m.columns) == MANIFEST_COLUMNS)
    check("manifest has 604 rows = 151 sources x 4 variants", len(m) == 604 and m.source_id.nunique() == 151)
    g = m.groupby("source_id")
    check("each source has exactly the 4 distinct variants", bool((g.variant.nunique() == 4).all() and (g.size() == 4).all()))
    check("each source lies in exactly one split (no crossover)", bool((g.split.nunique() == 1).all()))
    check("verify_split_integrity reports no problems", verify_split_integrity(m[["source_id", "split"]].to_dict("records")) == [])
    check("manifest source set equals on-disk source set", sorted(m.source_id.unique()) == sources)
    check("all manifest filepaths exist", all(Path(p).exists() for p in m.filepath))
    mqf = m.groupby("source_id").jpeg_quality_factor.first()
    check("manifest QF equals independently recomputed QF", all({int(mqf[s])} == qf_by_source[s] for s in sources))
    dl = json.loads((REPORTS / "phase1b_scale_download.json").read_text())
    check("download report: 151 complete groups equal on-disk sources", sorted(dl["complete_group_ids"]) == sources)
    split_counts = m[m.variant == "cover"].split.value_counts().to_dict()
    qf_x_split = pd.crosstab(m[m.variant == "cover"].split, m[m.variant == "cover"].jpeg_quality_factor).to_dict()
    total_bytes = sum(p.stat().st_size for d in VARIANT_DIRS.values() for p in (ALASKA / d).glob("*.jpg"))
    return {
        "sources": sources, "n_files": n_files, "formats": fmt, "modes": mode, "sizes": size,
        "quant_table_counts": {str(k): v for k, v in ntab.items()}, "chroma_subsampling_codes(0=4:4:4)": {str(k): v for k, v in sub.items()},
        "decode_failures": len(failures), "qf_source_counts": qf_counts, "split_source_counts": split_counts,
        "qf_x_split_sources": {str(k): v for k, v in qf_x_split.items()}, "total_bytes": total_bytes,
        "download": {k: dl[k] for k in ("candidate_pool_size", "complete_group_count", "id_gap_count", "id_gap_ids",
                                        "incomplete_group_count", "files_reused_from_local", "files_downloaded_this_run",
                                        "bytes_downloaded_this_run", "rate_limit_429_events", "wall_seconds",
                                        "seconds_inside_download_calls", "pacing", "seed")},
    }


# --------------------------------------------------------------------------
# C. ALASKA2 evaluation: independent recomputation
# --------------------------------------------------------------------------


def verify_alaska_eval(bundle: dict, sources: list[str]) -> dict:
    ev = json.loads((REPORTS / "phase1b_pilot_evaluation.json").read_text())
    check("ALASKA2 evaluation scored 151/151 sources, 0 failures", ev["n_sources_scored"] == 151 and ev["n_failed_sources"] == 0)
    check("evaluation recorded model hash unchanged before/after", ev["integrity"]["model_unchanged"] and ev["integrity"]["model_sha256_after"] == EXPECTED_SHA256)

    sc = pd.read_csv(REPORTS / "phase1b_alaska2_scores.csv", dtype={"source_id": str}, float_precision="round_trip").set_index("source_id").loc[sources]
    qf = sc["qf"].to_numpy()
    cover = sc["cover"].to_numpy()
    out: dict = {"stored": ev}

    # in-domain BOSSBase reference, recomputed with the frozen model
    boss = pd.read_csv("data/processed/features_payload_0.10.csv")
    bt = boss[(boss.split == "test") & (boss.label == 0)]
    boss_cover = bundle["model"].predict_proba(bt[list(FEATURE_NAMES)].to_numpy(dtype=np.float64))[:, 1]
    ref = ev["in_domain_reference_bossbase_test_payload_0.10"]["cover_score"]
    check("BOSSBase in-domain cover scores reproduce the stored reference", len(boss_cover) == 1500 and close(boss_cover.mean(), ref["mean"]) and close((boss_cover >= 0.5).mean(), ref["fraction_ge_0.5"]))

    # Experiment A
    ks = stats.ks_2samp(cover, boss_cover)
    mw = stats.mannwhitneyu(cover, boss_cover, alternative="two-sided")
    A = ev["experiment_A_clean_cover_domain_shift"]
    check("Exp A: recomputed mean/median/std/fraction>=0.5 match stored", close(cover.mean(), A["overall"]["mean"]) and close(np.median(cover), A["overall"]["median"]) and close(cover.std(ddof=1), A["overall"]["std"]) and close((cover >= 0.5).mean(), A["overall"]["fraction_ge_0.5"]))
    check("Exp A: recomputed KS statistic and p match stored", close(ks.statistic, A["vs_bossbase_test_covers"]["ks_2samp_statistic"]) and close(ks.pvalue, A["vs_bossbase_test_covers"]["ks_p_value"]))
    groups = {lv: cover[qf == lv] for lv in (75, 90, 95)}
    kw = stats.kruskal(*groups.values())
    rho = stats.spearmanr(qf.astype(float), cover)
    pair_names = [(75, 90), (75, 95), (90, 95)]
    pair_p = [float(stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue) for a, b in pair_names]
    pair_holm = holm_adjust(pair_p)
    check("Exp A: recomputed Kruskal-Wallis and Spearman match stored", close(kw.statistic, A["qf_association"]["kruskal_wallis_H"]) and close(rho.statistic, A["qf_association"]["spearman_rho_qf_vs_score"]))
    out["A"] = {
        "alaska_n": len(cover), "boss_n": len(boss_cover),
        "alaska": {"mean": float(cover.mean()), "median": float(np.median(cover)), "std": float(cover.std(ddof=1)), "min": float(cover.min()), "max": float(cover.max()), "frac_ge_0.5": float((cover >= 0.5).mean()), "wilson95": A["overall"]["fraction_ge_0.5_wilson95"]},
        "boss": {"mean": float(boss_cover.mean()), "median": float(np.median(boss_cover)), "std": float(boss_cover.std(ddof=1)), "min": float(boss_cover.min()), "max": float(boss_cover.max()), "frac_ge_0.5": float((boss_cover >= 0.5).mean())},
        "ks": [float(ks.statistic), float(ks.pvalue)], "mannwhitney_p": float(mw.pvalue),
        "median_diff": float(np.median(cover) - np.median(boss_cover)),
        "by_qf": {f"QF{lv}": {"n": int(len(g)), "mean": float(g.mean()), "median": float(np.median(g)), "std": float(g.std(ddof=1)), "min": float(g.min()), "max": float(g.max()), "frac_ge_0.5": float((g >= 0.5).mean())} for lv, g in groups.items()},
        "kruskal": [float(kw.statistic), float(kw.pvalue)], "spearman": [float(rho.statistic), float(rho.pvalue)],
        "pairwise_mw": {f"QF{a}_vs_QF{b}": {"p_raw": p, "p_holm": h} for (a, b), p, h in zip(pair_names, pair_p, pair_holm)},
    }

    # Experiments B and C
    comps = {
        "B_lsb_0.10": ("lsb_0.10", ev["experiment_B_controlled_synthetic_lsb"]["by_payload"]["0.10"]),
        "B_lsb_0.40": ("lsb_0.40", ev["experiment_B_controlled_synthetic_lsb"]["by_payload"]["0.40"]),
        "C_jmipod": ("jmipod", ev["experiment_C_dataset_stego"]["jmipod"]),
        "C_juniward": ("juniward", ev["experiment_C_dataset_stego"]["juniward"]),
        "C_uerd": ("uerd", ev["experiment_C_dataset_stego"]["uerd"]),
    }
    res: dict = {}
    all_ok = True
    for key, (col, blk) in comps.items():
        stego = sc[col].to_numpy()
        stored = blk["overall"]
        auc = auc_rank(cover, stego)
        y = np.array([0] * len(cover) + [1] * len(stego))
        acc = float(((np.concatenate([cover, stego]) >= 0.5).astype(int) == y).mean())
        diff = stego - cover
        w = stats.wilcoxon(diff)
        all_ok &= close(auc, stored["metrics"]["roc_auc"]) and close(acc, stored["metrics"]["accuracy"]) and close(w.pvalue, stored["paired_score_diff_stego_minus_cover"]["wilcoxon_signed_rank"]["p_value"], 1e-12)
        all_ok &= stored["roc_auc_bootstrap95_over_sources"][0] <= auc <= stored["roc_auc_bootstrap95_over_sources"][1]
        entry = {
            "n": len(cover), "auc": auc, "auc_ci": stored["roc_auc_bootstrap95_over_sources"], "accuracy": acc,
            "accuracy_ci": stored["accuracy_bootstrap95_over_sources"], "precision": stored["metrics"]["precision"],
            "recall": stored["metrics"]["recall"], "f1": stored["metrics"]["f1"], "confusion_matrix": stored["metrics"]["confusion_matrix"],
            "cover_fpr": stored["cover_fpr_at_0.5"], "stego_tpr": stored["stego_tpr_at_0.5"],
            "cover_mean": float(cover.mean()), "stego_mean": float(stego.mean()),
            "diff_mean": float(diff.mean()), "diff_median": float(np.median(diff)),
            "frac_stego_gt_cover": float((diff > 0).mean()), "frac_ties": float((diff == 0).mean()),
            "wilcoxon_p_raw": float(w.pvalue), "by_qf": {},
        }
        for lv in (75, 90, 95):
            mk = qf == lv
            wq = stats.wilcoxon(stego[mk] - cover[mk])
            sq = blk["by_qf"][f"QF{lv}"]
            aq = auc_rank(cover[mk], stego[mk])
            all_ok &= close(aq, sq["metrics"]["roc_auc"]) and close(wq.pvalue, sq["paired_score_diff_stego_minus_cover"]["wilcoxon_signed_rank"]["p_value"], 1e-12)
            entry["by_qf"][f"QF{lv}"] = {"n": int(mk.sum()), "auc": aq, "auc_ci": sq["roc_auc_bootstrap95_over_sources"], "accuracy": sq["metrics"]["accuracy"], "wilcoxon_p_raw": float(wq.pvalue), "diff_mean": float((stego[mk] - cover[mk]).mean())}
        res[key] = entry
    check("Exp B/C: rank-based AUC, accuracy and Wilcoxon p (overall and per QF) match stored values; point AUC lies inside its CI", bool(all_ok))

    keys = list(res)
    holm_overall = holm_adjust([res[k]["wilcoxon_p_raw"] for k in keys])
    for k, h in zip(keys, holm_overall):
        res[k]["wilcoxon_p_holm_overall_family5"] = h
    qf_keys = [(k, q) for k in keys for q in ("QF75", "QF90", "QF95")]
    holm_qf = holm_adjust([res[k]["by_qf"][q]["wilcoxon_p_raw"] for k, q in qf_keys])
    for (k, q), h in zip(qf_keys, holm_qf):
        res[k]["by_qf"][q]["wilcoxon_p_holm_family15"] = h
    out["BC"] = res

    # cross-domain AUC difference (independent bootstrap over sources in each domain)
    rs = pd.read_csv(REPORTS / "phase1b_robustness_scores.csv", dtype={"source_id": str}, float_precision="round_trip")
    rng = np.random.default_rng(42)
    xdom = {}
    for p, key in ((0.10, "B_lsb_0.10"), (0.40, "B_lsb_0.40")):
        bi = rs[(rs["transform"] == "none") & (np.isclose(rs.payload, p))].sort_values("source_id")
        bc, bs_ = bi.cover_score.to_numpy(), bi.stego_score.to_numpy()
        ac, as_ = cover, sc[comps[key][0]].to_numpy()
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(ac), len(ac)); j = rng.integers(0, len(bc), len(bc))
            diffs.append(auc_rank(ac[i], as_[i]) - auc_rank(bc[j], bs_[j]))
        xdom[f"{p:.2f}"] = {"alaska_auc": res[key]["auc"], "bossbase_auc": auc_rank(bc, bs_), "delta": res[key]["auc"] - auc_rank(bc, bs_),
                            "delta_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    out["cross_domain_auc_delta"] = xdom
    out["feature_shift"] = ev["feature_shift_descriptive"]["features"]
    out["runtime"] = ev["runtime_seconds"]

    # supplementary descriptive diagnostic S1: paired feature change, ALASKA2 stego minus cover
    m = pd.read_csv(MANIFEST, dtype={"source_id": str})
    piv = m.pivot(index="source_id", columns="variant", values="filepath").loc[sources]
    names = list(FEATURE_NAMES)
    F = {k: [] for k in ("cover", "jmipod", "juniward", "uerd", "lsb0.10", "lsb0.40")}
    for sid, row in piv.iterrows():
        a = np.asarray(Image.open(row["cover"]).convert("RGB"), dtype=np.uint8)
        F["cover"].append(_feat(a))
        for v in ("jmipod", "juniward", "uerd"):
            F[v].append(_feat(np.asarray(Image.open(row[v]).convert("RGB"), dtype=np.uint8)))
        for p in PAYLOADS:
            o = np.empty_like(a)
            for c, lab in enumerate("RGB"):
                o[..., c] = embed_lsb(a[..., c], p, sample_seed(42, f"{sid}_{lab}", p))
            F[f"lsb{p:.2f}"].append(_feat(o))
    F = {k: np.vstack(v).astype(np.float64) for k, v in F.items()}
    s1 = {}
    for f in ("lsb_transition_r", "lsb_local_variance_r"):
        j = names.index(f); sd = F["cover"][:, j].std(ddof=1)
        s1[f] = {}
        for k in ("jmipod", "juniward", "uerd", "lsb0.10", "lsb0.40"):
            d = F[k][:, j] - F["cover"][:, j]
            s1[f][k] = {"paired_mean_change_in_cover_std_units": float(d.mean() / sd), "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
    out["S1_paired_feature_change"] = s1
    return out


# --------------------------------------------------------------------------
# D. Robustness: independent recomputation, multiplicity, determinism
# --------------------------------------------------------------------------


def verify_robustness(bundle: dict) -> dict:
    rj = json.loads((REPORTS / "phase1b_robustness.json").read_text())
    rci = json.loads((REPORTS / "phase1b_robustness_ci.json").read_text())
    rs = pd.read_csv(REPORTS / "phase1b_robustness_scores.csv", dtype={"source_id": str}, float_precision="round_trip")
    sm = pd.read_csv("data/processed/split_manifest.csv", dtype={"source_id": str})
    test_ids = sorted(sm[sm.split == "test"].source_id)
    names = ["none", "jpeg_qf90", "jpeg_qf75", "resize_0.9x", "gaussian_noise_sigma2", "crop_480_offset13"]

    check("robustness: 18,000 score rows = 1,500 sources x 6 conditions x 2 payloads", len(rs) == 18000)
    check("robustness: score sources equal the Phase 1A TEST split (1,500), none from train/val", sorted(rs.source_id.unique()) == test_ids)
    grp = rs.groupby(["transform", "payload"]).source_id
    check("robustness: every (transform, payload) has each test source exactly once (cover/stego pairing intact)", bool((grp.nunique() == 1500).all() and (grp.size() == 1500).all()))
    check("robustness: 0 failed sources in every condition", all(rj["results"][n]["n_failed_sources"] == 0 for n in names))
    declared = json.loads(json.dumps([t.describe() for t in build_suite_v1()]))
    check("robustness: parameters recorded in the report equal the pre-declared suite in code", rj["transforms"][1:] == declared)
    check("robustness: model hash unchanged inside the run", rj["integrity"]["model_unchanged"] and rj["integrity"]["model_sha256_after"] == EXPECTED_SHA256)
    check("robustness: report flags zero duplicate content and zero train/val sources", rj["integrity"]["duplicate_content_within_evaluated_sources"] == 0 and rj["integrity"]["sources_in_train_or_val"] == 0)

    # identity condition versus the stored Phase 1A evaluation
    p1a = json.loads((REPORTS / "evaluation_payload_0.10_test.json").read_text())["metrics"]
    sens = {round(r["payload"], 2): r for r in json.loads((REPORTS / "payload_sensitivity.json").read_text())["results"]}
    ident_ok = True
    for p, stored in ((0.10, p1a), (0.40, sens[0.4])):
        d = rs[(rs["transform"] == "none") & np.isclose(rs.payload, p)].sort_values("source_id")
        c, s = d.cover_score.to_numpy(), d.stego_score.to_numpy()
        tn, fp = int((c < 0.5).sum()), int((c >= 0.5).sum()); fn, tp = int((s < 0.5).sum()), int((s >= 0.5).sum())
        acc = (tp + tn) / 3000; prec = tp / (tp + fp); rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec)
        ident_ok &= [[tn, fp], [fn, tp]] == stored["confusion_matrix"] and close(acc, stored["accuracy"]) and close(prec, stored["precision"]) and close(rec, stored["recall"]) and close(f1, stored["f1"]) and close(auc_rank(c, s), stored["roc_auc"])
    check("robustness: identity condition reproduces Phase 1A stored metrics and confusion matrices (payloads 0.10 and 0.40)", bool(ident_ok))

    # AUC per condition recomputed vs CI report; Bonferroni-level intervals for the 10 deltas
    conds: dict = {}
    auc_ok = True
    non_identity = [n for n in names if n != "none"]
    for p in PAYLOADS:
        base = rs[(rs["transform"] == "none") & np.isclose(rs.payload, p)].sort_values("source_id")
        for n in names:
            d = rs[(rs["transform"] == n) & np.isclose(rs.payload, p)].sort_values("source_id")
            c, s = d.cover_score.to_numpy(), d.stego_score.to_numpy()
            stored = rci["conditions"][f"{n}|payload={p:.2f}"]
            auc_ok &= close(auc_rank(c, s), stored["roc_auc"])
            entry = {"auc": auc_rank(c, s), "auc_ci95": stored["roc_auc_bootstrap95"], "cover_fpr": float((c >= 0.5).mean()),
                     "stego_tpr": float((s >= 0.5).mean()), "mean_cover_score": float(c.mean()), "mean_stego_score": float(s.mean()),
                     "accuracy": float((((np.concatenate([c, s]) >= 0.5).astype(int)) == np.array([0] * 1500 + [1] * 1500)).mean())}
            if n != "none":
                bc, bs_ = base.cover_score.to_numpy(), base.stego_score.to_numpy()
                b95 = paired_auc_bootstrap(c, s, baseline_cover=bc, baseline_stego=bs_, n_resamples=N_BOOT_CI, alpha=0.05, seed=42)
                bbon = paired_auc_bootstrap(c, s, baseline_cover=bc, baseline_stego=bs_, n_resamples=N_BOOT_CI, alpha=0.05 / len(non_identity), seed=42)
                entry.update({"delta_auc": b95["delta_auc"], "delta_ci95": b95["delta_auc_ci"], "delta_ci_bonferroni10": bbon["delta_auc_ci"],
                              "delta_ci95_stored": stored["delta_auc_bootstrap95"]})
                entry["delta_survives_bonferroni10"] = not (bbon["delta_auc_ci"][0] <= 0.0 <= bbon["delta_auc_ci"][1])
            conds[f"{n}|{p:.2f}"] = entry
    check("robustness: AUC of all 12 conditions recomputed (rank-based) equals the stored CI report", bool(auc_ok))
    mem_ok = all(close(conds[f"{n}|{p:.2f}"]["auc"], rj["results"][n]["by_payload"][f"{p:.2f}"]["metrics"]["roc_auc"], 1e-9) and
                 close(conds[f"{n}|{p:.2f}"]["accuracy"], rj["results"][n]["by_payload"][f"{p:.2f}"]["metrics"]["accuracy"], 1e-12)
                 for n in names for p in PAYLOADS)
    check("robustness: recomputed AUC and accuracy of all 12 conditions equal the in-memory metrics stored by robustness_eval.py", bool(mem_ok))

    # end-to-end determinism spot check on a small deterministic subset
    subset = test_ids[:25]
    suite = [None] + build_suite_v1()
    max_diff = 0.0
    rows = []
    for sid in subset:
        px = np.asarray(Image.open(f"data/raw/BOSSBase/{sid}.pgm"), dtype=np.uint8).copy()
        stg = {p: embed_lsb(px, p, sample_seed(42, sid, p)) for p in PAYLOADS}
        ns = sample_seed(42, f"{sid}_robustness_noise", 0.0)
        for n, t in zip(names, suite):
            cv = px if t is None else t.apply(px, ns)
            rows.append((sid, n, None, _feat(cv)))
            for p in PAYLOADS:
                sv = stg[p] if t is None else t.apply(stg[p], ns)
                rows.append((sid, n, p, _feat(sv)))
    X = np.vstack([r[3] for r in rows])
    pr = bundle["model"].predict_proba(X)[:, 1]
    lookup = rs.set_index(["source_id", "transform", "payload"])
    for (sid, n, p, _), score in zip(rows, pr):
        if p is None:
            for pp in PAYLOADS:
                max_diff = max(max_diff, abs(score - float(lookup.loc[(sid, n, pp)].cover_score)))
        else:
            max_diff = max(max_diff, abs(score - float(lookup.loc[(sid, n, p)].stego_score)))
    check("robustness: independent re-execution of 25 sources x 6 conditions x 2 payloads reproduces the saved scores", max_diff < 1e-9, f"max|diff|={max_diff:.2e}")

    # supplementary descriptive diagnostic S2: univariate feature AUC under resize / crop (payload 0.40)
    sub = select_pilot_sources(test_ids, 300, 7)
    P = 0.40
    fc = {k: [] for k in ("none", "resize_0.9x", "crop_480_offset13")}
    fs = {k: [] for k in fc}
    tmap = {"none": None, "resize_0.9x": suite[names.index("resize_0.9x")], "crop_480_offset13": suite[names.index("crop_480_offset13")]}
    for sid in sub:
        px = np.asarray(Image.open(f"data/raw/BOSSBase/{sid}.pgm"), dtype=np.uint8).copy()
        st = embed_lsb(px, P, sample_seed(42, sid, P))
        for k, t in tmap.items():
            fc[k].append(_feat(px if t is None else t.apply(px, 0)))
            fs[k].append(_feat(st if t is None else t.apply(st, 0)))
    s2 = {}
    for f in ("lsb_transition_r", "lsb_local_variance_r"):
        j = list(FEATURE_NAMES).index(f)
        s2[f] = {k: auc_rank(np.array(fc[k])[:, j].astype(np.float64), np.array(fs[k])[:, j].astype(np.float64)) for k in fc}
    return {"stored": rj, "conditions": conds, "spot_check_max_abs_score_diff": max_diff, "S2_univariate_feature_auc_payload_0.40_n300": s2,
            "runtime": rj["runtime_seconds"], "n_test_sources": len(test_ids)}


# --------------------------------------------------------------------------
# E. Report rendering
# --------------------------------------------------------------------------


def f4(x: float) -> str:
    return f"{x:.4f}"


def fp(p: float) -> str:
    return f"{p:.3g}"


def ci(c: list[float]) -> str:
    return f"[{c[0]:.4f}, {c[1]:.4f}]"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render(R: dict) -> str:
    M, D, A, BC, RB = R["model"], R["dataset"], R["alaska"], R["alaska"]["BC"], R["robustness"]
    a = A["A"]
    boss_stored = json.loads((REPORTS / "evaluation_payload_0.10_test.json").read_text())["metrics"]
    xd = A["cross_domain_auc_delta"]
    cond = RB["conditions"]

    def incl0(c: list[float]) -> str:
        return "includes 0" if c[0] <= 0.0 <= c[1] else "excludes 0"

    xd_txt = "; ".join(
        f"{p} bpp: ALASKA2 {v['alaska_auc']:.4f} vs BOSSBase {v['bossbase_auc']:.4f}, difference {v['delta']:+.4f} (95% CI {ci(v['delta_ci95'])}, {incl0(v['delta_ci95'])})"
        for p, v in xd.items()
    )

    def dlt(name: str, p: str) -> str:
        c = cond[f"{name}|{p}"]
        return f"{c['delta_auc']:+.3f} (Bonferroni-10 CI {incl0(c['delta_ci_bonferroni10'])})"

    L: list[str] = []
    w = L.append

    w("# StegoShield — Phase 1B Final Report (external generalization and robustness)\n")
    w(f"Generated by `scripts/finalize_phase1b_report.py` from the saved artifacts on {R['generated']} (git HEAD `{M['git_head'][:10]}`). "
      "Every number below was recomputed from raw per-source scores or files and cross-checked against the stored experiment reports; "
      f"{R['n_checks_passed']}/{R['n_checks']} verification checks passed. Tags: **[VERIFIED]** re-derived fact, **[MEASURED]** computed result, "
      "**[INTERPRETATION]** reading of a measurement, **[HYPOTHESIS]** untested explanation, **[LIMITATION]**.\n")
    w("> Scope statement. This is a research evaluation of a prototype LSB-statistics detector. It is not a claim that the detector is production-ready, "
      "that ALASKA2 is representative of real-world steganography, or that cross-dataset / cross-steganography numbers are equivalent to the in-domain Phase 1A task.\n")

    w("## 1. Objective\n")
    w("Close Phase 1B by answering, with the Phase 1A detector frozen: (RQ5) does it generalize to an independent image corpus (ALASKA2: JPEG, RGB, 40+ cameras, QF 75/90/95); "
      "(a) with a controlled LSB embedding matched to what it was trained to detect, (b) against ALASKA2's own realistic DCT-domain steganography (JMiPOD, J-UNIWARD, UERD); "
      "and (RQ4/FR-10) how stable is in-domain detection after common image transformations.\n")

    w("## 2. Experimental setup\n")
    w("- Detector: the frozen Phase 1A RandomForest and the unchanged 22-feature extractor. No retraining, no scaler (the artifact has none), no threshold or hyper-parameter change, nothing tuned on ALASKA2 or on robustness results.")
    w("- Unit of analysis: the **source** (one numbered cover identity). The four variants of a source are not independent; intervals resample whole sources and tests use per-source paired differences.")
    w("- Decision threshold used wherever a hard label is needed: 0.5 (fixed in advance; never tuned).")
    w("- Experiments: A clean-cover score control; B controlled synthetic LSB (0.10 bpp/channel primary, 0.40 supplementary declared before results); C JMiPOD / JUNIWARD / UERD, each its own balanced comparison (never pooled); "
      "R robustness on the BOSSBase TEST split, the same transformation applied to cover and stego.\n")

    w("## 3. Frozen Phase 1A model\n")
    w(table(["property", "value"], [
        ["SHA-256", f"`{M['sha256']}`"], ["size (bytes)", str(M["size_bytes"])], ["model", f"{M['model_name']} {M['model_version']}"],
        ["hyper-parameters", f"n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42, class_weight=balanced"],
        ["feature schema", f"{M['feature_schema_version']}, 22 features, names hash `{M['feature_names_sha256'][:16]}…`"],
        ["scaler", "None"], ["trained on", "BOSSBase, payload 0.10, source-aware train split (Phase 1A)"],
    ]))
    w("")
    w(f"[MEASURED] Phase 1A in-domain BOSSBase TEST (n=1500 cover + 1500 stego, payload 0.10): accuracy {boss_stored['accuracy']:.4f}, precision {boss_stored['precision']:.4f}, "
      f"recall {boss_stored['recall']:.4f}, F1 {boss_stored['f1']:.4f}, ROC-AUC {boss_stored['roc_auc']:.4f}, confusion matrix {boss_stored['confusion_matrix']}. "
      f"At payload 0.40 the same frozen model reaches ROC-AUC {cond['none|0.40']['auc']:.4f}. These are weak-to-moderate detection scores and are the baseline for everything below.\n")

    w("## 4. ALASKA2 dataset / sample construction\n")
    dl = D["download"]
    w("- Source: Kaggle competition `alaska2-image-steganalysis`, obtained by selective per-file download (never the full ~300k-image corpus).")
    w(f"- Selection: deterministic seed {dl['seed']}; `select_pilot_sources()` permutes the ID list with a seeded RNG and slices a prefix, so a larger pool is a superset of a smaller one "
      f"(the original 35-source pilot is contained in the final sample). Candidates were drawn only from IDs 1–12,813, the range observed in the earlier bounded listing pass.")
    w(f"- Pool of {dl['candidate_pool_size']} candidates -> **{dl['complete_group_count']} complete groups**; {dl['id_gap_count']} IDs had no Cover file (404: `{', '.join(dl['id_gap_ids'])}`); "
      f"{dl['incomplete_group_count']} existing covers lacked a stego variant.")
    w(f"- Download: {dl['files_downloaded_this_run']} files ({dl['bytes_downloaded_this_run']/1e6:.1f} MB) fetched this run, {dl['files_reused_from_local']} reused from the pilot, "
      f"{dl['rate_limit_429_events']} HTTP-429 events, pacing {dl['pacing']['delay_seconds']} s between calls with exponential backoff, wall {dl['wall_seconds']:.0f} s.")
    w(f"- On disk: 4 variant directories x 151 files = {D['n_files']} JPEGs, {D['total_bytes']/1e6:.1f} MB (mean {D['total_bytes']/D['n_files']/1e3:.1f} KB per image).\n")

    w("## 5. Data integrity validation\n")
    w("[VERIFIED] (independently re-derived from the files on disk for this report)\n")
    w(table(["check", "result"], [
        ["complete source groups (Cover+JMiPOD+JUNIWARD+UERD)", "151"],
        ["image files / decode failures", f"{D['n_files']} / {D['decode_failures']}"],
        ["format / mode / dimensions", f"{D['formats']} / {D['modes']} / {D['sizes']}"],
        ["JPEG quantization tables per file", str(D["quant_table_counts"])],
        ["chroma subsampling (0 = 4:4:4)", str(D["chroma_subsampling_codes(0=4:4:4)"])],
        ["exact-duplicate files (SHA-256 of file bytes / of decoded pixels)", "0 / 0 across all 604"],
        ["pairing (same IDs in all four directories)", "151 common, 0 variant-only"],
        ["split by source (train / val / test)", f"{D['split_source_counts'].get('train')} / {D['split_source_counts'].get('val')} / {D['split_source_counts'].get('test')}"],
        ["split violations (source in >1 split) / duplicate crossovers", "0 / 0 (no duplicates exist)"],
    ]))
    w("")
    w("[LIMITATION] Camera / source metadata is not exposed, so 'source-aware' means per-image identity (a cover and all its variants stay together), not camera-level separation. "
      f"The split is not QF-stratified (test sources by QF75/90/95: {D['qf_x_split_sources']}). No model is trained on ALASKA2, so all 151 sources are evaluated; "
      "the split exists only to keep any future ALASKA2 calibration experiment leakage-safe.\n")

    w("## 6. QF validation\n")
    q = D["qf_source_counts"]
    w(f"[VERIFIED] Every file's luma quantization table is a byte-for-byte match to Pillow's own standard table for QF 75, 90 or 95 (tables generated locally, not transcribed). "
      f"Sources: QF75 = {q[75]}, QF90 = {q[90]}, QF95 = {q[95]}; 0 unmatched; every source has the same QF in all four variants (0 inconsistencies), so QF is not a cover-vs-stego shortcut inside any comparison. "
      "[LIMITATION] This validates 'matches the standard IJG-scaled table for QF=N', not ALASKA2's internal labelling.\n")

    w("## 7. Experiment A — clean-cover control\n")
    w("**Score distribution only. There is no positive class, so this is not classification accuracy.**\n")
    w(table(["set", "n", "mean", "median", "std", "min", "max", "fraction ≥ 0.5"], [
        ["ALASKA2 covers", a["alaska_n"], f4(a["alaska"]["mean"]), f4(a["alaska"]["median"]), f4(a["alaska"]["std"]), f4(a["alaska"]["min"]), f4(a["alaska"]["max"]), f"{a['alaska']['frac_ge_0.5']:.4f} (Wilson 95% {ci(a['alaska']['wilson95'])})"],
        ["BOSSBase TEST covers (in-domain)", a["boss_n"], f4(a["boss"]["mean"]), f4(a["boss"]["median"]), f4(a["boss"]["std"]), f4(a["boss"]["min"]), f4(a["boss"]["max"]), f"{a['boss']['frac_ge_0.5']:.4f}"],
    ]))
    w("")
    w(f"[MEASURED] ALASKA2 vs in-domain cover scores: KS D = {a['ks'][0]:.4f} (p = {fp(a['ks'][1])}), Mann-Whitney p = {fp(a['mannwhitney_p'])}, median difference {a['median_diff']:+.4f}. "
      "No difference in score location or shape is detectable at this sample size (absence of evidence, not proof of equivalence).\n")
    w("**By QF (unit = source):**\n")
    w(table(["QF", "n", "mean", "median", "std", "min", "max", "fraction ≥ 0.5"],
            [[k, v["n"], f4(v["mean"]), f4(v["median"]), f4(v["std"]), f4(v["min"]), f4(v["max"]), f"{v['frac_ge_0.5']:.4f}"] for k, v in a["by_qf"].items()]))
    w("")
    pw = a["pairwise_mw"]
    w(f"[MEASURED] Kruskal-Wallis H = {a['kruskal'][0]:.3f}, p = {fp(a['kruskal'][1])}; Spearman ρ(QF, score) = {a['spearman'][0]:.3f}, p = {fp(a['spearman'][1])}; "
      "pairwise Mann-Whitney (raw p -> Holm p): " + "; ".join(f"{k}: {fp(v['p_raw'])} -> {fp(v['p_holm'])}" for k, v in pw.items()) + ". "
      "The QF75 group has a visibly larger spread (std above) but variance equality was not tested.\n")
    w("**Distinguishing what was and was not shown:**\n")
    w(f"- [MEASURED] score distribution: ALASKA2 covers are statistically indistinguishable from BOSSBase test covers ({a['alaska']['frac_ge_0.5']*100:.1f}% vs {a['boss']['frac_ge_0.5']*100:.1f}% at or above 0.5).")
    w("- [INTERPRETATION] The 'fraction ≥ 0.5' statistic is therefore **not** evidence of domain shift. It reflects the model's own calibration/bias: this detector's scores cluster around 0.5 with the cover median just above it, in-domain as well. "
      "An earlier reading of the 35-source pilot (74.3% ≥ 0.5 taken as domain-shift false positives) was withdrawn once the in-domain rate (71.9%) was compared.")
    fs = A["feature_shift"]
    top = sorted(fs.items(), key=lambda kv: -kv[1]["rf_feature_importance"])[:6]
    w("- [MEASURED] Raw feature distributions do differ (descriptive): " + "; ".join(f"`{k}` {v['standardized_mean_diff_in_bossbase_std']:+.2f}σ (RF importance {v['rf_feature_importance']:.3f})" for k, v in top)
      + f"; brightness features `mean_r/g/b` shift by {fs['mean_r']['standardized_mean_diff_in_bossbase_std']:+.2f}σ / {fs['mean_g']['standardized_mean_diff_in_bossbase_std']:+.2f}σ / {fs['mean_b']['standardized_mean_diff_in_bossbase_std']:+.2f}σ "
      "(σ = BOSSBase training-cover std) and, unlike BOSSBase, R, G and B differ from each other.")
    w("- [INTERPRETATION] Feature-level shift exists, but the model's scores did not move detectably; that is all this experiment supports. Importance alone does not show which features determine a given score.")
    w("- [HYPOTHESIS, untested] JPEG decompression alters low-order-bit statistics in ways that partly offset or mask other shifts. No experiment here isolates that mechanism.\n")

    w("## 8. Experiment B — controlled LSB generalization (synthetic; not ALASKA2's native steganography)\n")
    w("The unmodified Phase 1A `embed_lsb()` is applied independently to the R, G and B channels of each decoded ALASKA2 cover (distinct deterministic seeds), at the stated bits per pixel **per channel**. "
      "The result is a pixel-domain modification of a decoded JPEG; it is not re-encoded to JPEG.\n")
    rows = []
    for k in ("B_lsb_0.10", "B_lsb_0.40"):
        e = BC[k]
        rows.append([k.replace("B_lsb_", ""), e["n"], f4(e["accuracy"]), f4(e["precision"]), f4(e["recall"]), f4(e["f1"]), f"{f4(e['auc'])} {ci(e['auc_ci'])}", str(e["confusion_matrix"]), f4(e["cover_fpr"]), f4(e["stego_tpr"])])
    w(table(["payload (bpp/ch)", "n pairs", "acc", "prec", "recall", "F1", "ROC-AUC [95% CI, sources]", "confusion [[TN,FP],[FN,TP]]", "cover FPR@0.5", "stego TPR@0.5"], rows))
    w("")
    w(table(["payload", "mean cover score", "mean stego score", "paired mean diff (stego−cover)", "fraction stego > cover", "Wilcoxon p raw", "Holm p (family of 5)"],
            [[k.replace("B_lsb_", ""), f4(BC[k]["cover_mean"]), f4(BC[k]["stego_mean"]), f"{BC[k]['diff_mean']:+.4f}", f4(BC[k]["frac_stego_gt_cover"]), fp(BC[k]["wilcoxon_p_raw"]), fp(BC[k]["wilcoxon_p_holm_overall_family5"])] for k in ("B_lsb_0.10", "B_lsb_0.40")]))
    w("")
    w("**By QF (n ≈ 50 sources each; intervals are wide):**\n")
    rows = []
    for k in ("B_lsb_0.10", "B_lsb_0.40"):
        for qk, v in BC[k]["by_qf"].items():
            rows.append([k.replace("B_lsb_", ""), qk, v["n"], f"{f4(v['auc'])} {ci(v['auc_ci'])}", fp(v["wilcoxon_p_raw"]), fp(v["wilcoxon_p_holm_family15"])])
    w(table(["payload", "QF", "n", "ROC-AUC [95% CI]", "Wilcoxon p raw", "Holm p (family of 15)"], rows))
    w("")
    w("[MEASURED] Cross-domain AUC difference (ALASKA2 minus BOSSBase test, independent bootstrap over sources in each domain):\n")
    w(table(["payload", "ALASKA2 AUC", "BOSSBase AUC", "difference", "95% CI"],
            [[p, f4(v["alaska_auc"]), f4(v["bossbase_auc"]), f"{v['delta']:+.4f}", ci(v["delta_ci95"])] for p, v in xd.items()]))
    w("")
    w("[INTERPRETATION] A small, statistically detectable signal transfers at both payloads (paired Wilcoxon survives Holm correction), but the detector remains weak in absolute terms, as in-domain. "
      "Cross-domain AUC difference by payload — " + xd_txt + ". A bound within 0.01 of zero should be read as borderline; these are uncorrected bootstrap comparisons of non-identical embedding settings. "
      "[LIMITATION] BOSSBase used one embedded plane replicated to three identical channels; here three channels are embedded independently, so the two AUCs are not measured under identical conditions.\n")

    w("## 9. Experiment C — JMiPOD / JUNIWARD / UERD (each method separate, balanced 151 vs 151)\n")
    rows = []
    for k in ("C_jmipod", "C_juniward", "C_uerd"):
        e = BC[k]
        rows.append([k[2:], e["n"], f4(e["accuracy"]), f4(e["precision"]), f4(e["recall"]), f4(e["f1"]), f"{f4(e['auc'])} {ci(e['auc_ci'])}", str(e["confusion_matrix"]), f"{e['diff_mean']:+.5f}", f4(e["frac_stego_gt_cover"]), fp(e["wilcoxon_p_raw"]), fp(e["wilcoxon_p_holm_overall_family5"])])
    w(table(["method", "n pairs", "acc", "prec", "recall*", "F1*", "ROC-AUC [95% CI, sources]", "confusion", "paired mean diff", "frac stego>cover", "Wilcoxon p raw", "Holm p (family of 5)"], rows))
    w("\n*Recall and F1 here only reflect the base rate of scores ≥ 0.5 (cover FPR "
      f"{BC['C_jmipod']['cover_fpr']:.3f} vs stego TPR {BC['C_jmipod']['stego_tpr']:.3f} for JMiPOD); they are not evidence of detection.\n")
    rows = []
    for k in ("C_jmipod", "C_juniward", "C_uerd"):
        for qk, v in BC[k]["by_qf"].items():
            rows.append([k[2:], qk, v["n"], f"{f4(v['auc'])} {ci(v['auc_ci'])}", fp(v["wilcoxon_p_raw"]), fp(v["wilcoxon_p_holm_family15"])])
    w("**By QF:**\n")
    w(table(["method", "QF", "n", "ROC-AUC [95% CI]", "Wilcoxon p raw", "Holm p (family of 15)"], rows))
    w("")
    s1 = A["S1_paired_feature_change"]
    w("[MEASURED] Descriptive check on the model's two most important LSB features (paired change from cover to stego, in units of the across-source std of the cover feature): " +
      "; ".join(f"`{f}`: " + ", ".join(f"{k} {v['paired_mean_change_in_cover_std_units']:+.3f}σ" for k, v in d.items()) for f, d in s1.items()) + ".")
    w("[INTERPRETATION] For all three methods the AUC interval includes 0.5, no result survives Holm correction, and the features the detector relies on barely move (≈0.01σ or less) compared with the synthetic LSB embedding (≈0.1–0.3σ). "
      "This is consistent with the detector's pixel-domain LSB-statistic design not registering adaptive DCT-domain embedding, which is the limitation the design documents anticipated. "
      "It is a negative result about this detector; it says nothing about detectors designed for JPEG-domain steganography. "
      "Uncorrected per-QF p-values below 0.05 (if any) belong to a family of 15 tests and are not treated as findings.\n")

    w("## 10. BOSSBase robustness evaluation\n")
    w("Design (fixed in code before the run): frozen detector on all **1,500** Phase 1A TEST sources; for each source the original cover and its stego (payload p, seed `sample_seed(42, source_id, p)`) receive the **same** transformation, "
      "then feature extraction and the frozen model. Payload 0.10 is primary; 0.40 is supplementary (chosen from Phase 1A results because 0.10 is near chance in-domain). "
      "Each source contributes one cover and one stego row per condition; conditions are never pooled.\n")
    tp = R["robustness"]["stored"]["transforms"]
    w(table(["condition", "parameters"], [[t["name"], json.dumps(t["params"])] for t in tp]))
    w("")
    w("[VERIFIED] 0 failed sources in every condition; all sources belong to the Phase 1A test split; the identity condition reproduces the stored Phase 1A metrics and confusion matrices exactly for both payloads; "
      f"an independent re-execution of 25 sources x 6 conditions x 2 payloads reproduced the saved scores (max |score difference| {RB['spot_check_max_abs_score_diff']:.1e}); the model hash was unchanged.\n")
    for p in ("0.10", "0.40"):
        w(f"**Payload {p}** ({'primary' if p == '0.10' else 'supplementary'}; n = 1,500 source pairs per condition)\n")
        rows = []
        for n in ["none", "jpeg_qf90", "jpeg_qf75", "resize_0.9x", "gaussian_noise_sigma2", "crop_480_offset13"]:
            c = cond[f"{n}|{p}"]
            if n == "none":
                rows.append([n, f"{f4(c['auc'])} {ci(c['auc_ci95'])}", "—", "—", f4(c["accuracy"]), f4(c["cover_fpr"]), f4(c["stego_tpr"]), "—"])
            else:
                rows.append([n, f"{f4(c['auc'])} {ci(c['auc_ci95'])}", f"{c['delta_auc']:+.4f}", f"{ci(c['delta_ci95'])} ; {ci(c['delta_ci_bonferroni10'])}", f4(c["accuracy"]), f4(c["cover_fpr"]), f4(c["stego_tpr"]), "yes" if c["delta_survives_bonferroni10"] else "no"])
        w(table(["condition", "ROC-AUC [95% CI]", "ΔAUC vs none", "ΔAUC 95% CI ; Bonferroni-10 CI (99.5%)", "accuracy@0.5", "cover FPR@0.5", "stego TPR@0.5", "Δ excludes 0 after Bonferroni-10"], rows))
        w("")
    w("Full precision, precision/recall/F1 and confusion matrices per condition are in `data/reports/phase1b_robustness.json`; per-source scores in `phase1b_robustness_scores.csv`.\n")
    s2 = RB["S2_univariate_feature_auc_payload_0.40_n300"]
    w("[MEASURED] Descriptive diagnostic (300 test sources, payload 0.40): univariate AUC of the two dominant features, none / resize / crop = " +
      "; ".join(f"`{f}` {v['none']:.3f} / {v['resize_0.9x']:.3f} / {v['crop_480_offset13']:.3f}" for f, v in s2.items()) + ".")
    w("[INTERPRETATION] ΔAUC versus the untransformed condition (0.10 / 0.40 bpp): "
      f"JPEG QF90 {dlt('jpeg_qf90', '0.10')} / {dlt('jpeg_qf90', '0.40')}; JPEG QF75 {dlt('jpeg_qf75', '0.10')} / {dlt('jpeg_qf75', '0.40')}; "
      f"Gaussian noise σ=2 {dlt('gaussian_noise_sigma2', '0.10')} / {dlt('gaussian_noise_sigma2', '0.40')}; "
      f"small crop {dlt('crop_480_offset13', '0.10')} / {dlt('crop_480_offset13', '0.40')}; 0.9× resize {dlt('resize_0.9x', '0.10')} / {dlt('resize_0.9x', '0.40')}. "
      "JPEG re-compression and noise leave the detector at AUC ≈ 0.5 (signal lost). They also move scores in different directions "
      "(JPEG lowers cover FPR and stego TPR together; noise pushes both to 1.0, giving the degenerate 'everything positive' rows with accuracy 0.5 and recall 1.0), so recall/F1 alone would mislead; AUC and FPR are the informative columns. "
      "Small crop changes AUC by less than 0.01 in either payload. Mild 0.9× resize did **not** degrade detection; that counter-intuitive result is preserved exactly as measured. "
      "The univariate diagnostic shows the same dominant feature (`lsb_transition_r`) carrying the signal after resize, which describes where the signal is, not why it survives. "
      "[HYPOTHESIS, untested] Interpolation followed by 8-bit rounding may keep low-order-bit transition statistics dependent on the embedded perturbation. No experiment here tests this.\n")

    w("## 11. Statistical methodology\n")
    w(f"- ROC-AUC: computed both with scikit-learn (stored reports) and by an independent rank-based (Mann-Whitney U, average ranks) implementation; all values agree to 1e-9.")
    w("- Uncertainty: percentile bootstrap over **sources** (a cover and its stego are resampled together); 2,000 resamples for ALASKA2 (seed 42), "
      f"{N_BOOT_CI:,} for the robustness deltas (seed 42). For a robustness delta both conditions use the same resampled sources.")
    w("- Paired tests: Wilcoxon signed-rank on per-source score differences (stego − cover). Group comparison of clean-cover scores: Kruskal-Wallis, Spearman (QF as ordinal), Mann-Whitney; two-sample KS versus BOSSBase.")
    w("- Multiplicity: Holm-Bonferroni within each family — overall paired tests for B and C (5), per-QF paired tests (15), QF pairwise cover comparisons (3). The 10 robustness ΔAUC comparisons are also shown with Bonferroni-level (99.5%) intervals. "
      "The single-run 95% intervals elsewhere are uncorrected.")
    w("- Cross-domain difference: independent bootstrap over ALASKA2 sources (n=151) and BOSSBase test sources (n=1500).")
    w("- Overlapping intervals are not a formal test; where a difference matters a difference interval is reported.\n")

    w("## 12. Results tables\n")
    w("The per-experiment tables are in sections 7–10. Consolidated ROC-AUC summary (95% source-bootstrap CI):\n")
    rows = [["BOSSBase in-domain (Phase 1A)", "0.10 bpp, single plane", f"{f4(cond['none|0.10']['auc'])} {ci(cond['none|0.10']['auc_ci95'])}"],
            ["BOSSBase in-domain (Phase 1A)", "0.40 bpp, single plane", f"{f4(cond['none|0.40']['auc'])} {ci(cond['none|0.40']['auc_ci95'])}"]]
    for k, lab in (("B_lsb_0.10", "ALASKA2 synthetic LSB 0.10 bpp/ch"), ("B_lsb_0.40", "ALASKA2 synthetic LSB 0.40 bpp/ch"), ("C_jmipod", "ALASKA2 JMiPOD (native)"), ("C_juniward", "ALASKA2 J-UNIWARD (native)"), ("C_uerd", "ALASKA2 UERD (native)")):
        rows.append([lab, "n=151 pairs", f"{f4(BC[k]['auc'])} {ci(BC[k]['auc_ci'])}"])
    w(table(["evaluation", "setting", "ROC-AUC"], rows))
    w("")

    w("## 13. Interpretation\n")
    w("1. [MEASURED → INTERPRETATION] In-domain the prototype is a weak LSB detector (AUC 0.553 at 0.10 bpp, 0.695 at 0.40 bpp) whose scores cluster near the 0.5 threshold.")
    w("2. [MEASURED → INTERPRETATION] Clean ALASKA2 covers receive scores indistinguishable from in-domain covers; the earlier 'domain-shift false-positive' reading is withdrawn. Raw features do shift, but that shift is not visible in the scores.")
    w("3. [MEASURED → INTERPRETATION] With matched (synthetic LSB) embedding a small signal transfers to ALASKA2 (" + xd_txt + "). The two settings differ (per-channel vs replicated embedding), so this compares related, not identical, tasks.")
    w("4. [MEASURED → INTERPRETATION] No evidence the detector separates ALASKA2's native JMiPOD, J-UNIWARD or UERD stego from cover. This is expected for a pixel-domain LSB-statistics design and does not indicate a pipeline fault.")
    w("5. [MEASURED → INTERPRETATION] The detector is not robust to JPEG re-compression or mild Gaussian noise (signal lost). It keeps its signal under small crop (|ΔAUC| < 0.01) and 0.9× downscaling (no decrease; " + f"{dlt('resize_0.9x', '0.10')} at 0.10 and {dlt('resize_0.9x', '0.40')} at 0.40); the cause of the resize behaviour is not established.")
    w("6. [LIMITATION] Cross-dataset and cross-steganography results are not equivalent to the in-domain Phase 1A task (different sensor pipelines, colour handling, compression, and embedding paradigms), and none of them supports a claim about steganography in general.\n")

    w("## 14. Limitations\n")
    for s in [
        "About 50 sources per QF group; per-QF intervals are wide and per-QF p-values sit in a family of 15.",
        "ALASKA2 sample is drawn from IDs 1–12,813 only; whether ID range correlates with camera, content or QF is unknown. Camera/source identity is unavailable, so leakage at camera level cannot be ruled out or measured.",
        "The QF label means an exact match to Pillow's standard table, not ALASKA2's documentation.",
        "The in-domain baseline is itself weak (AUC 0.553 at 0.10 bpp), leaving little headroom to measure degradation; the 0.40 supplementary payload was added for that reason.",
        "Experiment B embeds independently in three channels of a decoded JPEG and does not re-encode; it is a controlled synthetic scenario, not ALASKA2's steganography and not a realistic JPEG-domain attack.",
        "Robustness used one parameter per transformation, one noise realisation per source, and grayscale images; it does not characterise other strengths or transformation orders.",
        "Single fixed threshold 0.5; the model's scores are not calibrated probabilities, so 'likelihood score' wording only is justified.",
        "Only the pixel-domain detector was evaluated; no JPEG-domain features, deep detectors or retraining on ALASKA2 were tried (a possible separate future experiment).",
        "Per-source score CSVs must be read with pandas `float_precision='round_trip'`: the default fast parser perturbs some values by ~1e-16, which breaks exact cover/stego score ties and moved AUC by up to ~2e-5 and a Wilcoxon p-value by ~0.005 in a check made during finalization (no conclusion changes).",
        "Results depend on the package versions listed in section 15; RandomForest with n_jobs=-1 is metric-reproducible but not bit-identical in raw probabilities (differences ~1e-16 observed).",
        "ALASKA2's license forbids commercial use and redistribution of derived material; all ALASKA2 files and derived artifacts stay local and git-ignored.",
    ]:
        w(f"- {s}")
    w("")

    w("## 15. Reproducibility information\n")
    w(f"- Environment: Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, Pillow {PIL.__version__}. No dependency lock file exists.")
    w("- Seeds: 42 (split, LSB per-sample seeds via SHA-256, model, bootstrap, pilot selection). Robustness noise seed `sample_seed(42, '<id>_robustness_noise', 0.0)`; ALASKA2 LSB per-channel seeds `sample_seed(42, '<stem>_<R|G|B>', payload)`.")
    w("- Commands (repository root, project venv):")
    w("```text")
    for c in [
        "python scripts/download_alaska2_pilot.py --candidate-pool 160 --target-groups 160 --out-dir data/external/ALASKA2 --seed 42 --report-out data/reports/phase1b_scale_download.json",
        "python scripts/inspect_alaska2.py --root data/external/ALASKA2 --full-validate --report-out data/reports/phase1b_dataset_inspection.json",
        "python scripts/prepare_external_dataset.py --cover-dir data/external/ALASKA2/Cover --variant-dirs jmipod=data/external/ALASKA2/JMiPOD,juniward=data/external/ALASKA2/JUNIWARD,uerd=data/external/ALASKA2/UERD --pilot-size 1000 --require-complete --seed 42 --manifest-out data/external/ALASKA2/pilot_manifest.csv",
        "python scripts/evaluate_external_pilot.py --manifest data/external/ALASKA2/pilot_manifest.csv",
        "python scripts/robustness_eval.py && python scripts/analyze_robustness.py",
        "python scripts/finalize_phase1b_report.py",
    ]:
        w(c)
    w("```")
    w("- Runtime: ALASKA2 evaluation " + f"{A['runtime']['total_wall']:.0f} s ({A['runtime']['images_feature_extracted']} images, {A['runtime']['seconds_per_image']*1000:.1f} ms/image incl. decode and embedding, batch inference {A['runtime']['batch_inference']:.2f} s); "
      f"robustness {RB['runtime']['total']:.0f} s; download {dl['wall_seconds']:.0f} s.")
    w("- Superseded artifact: `data/reports/phase1b_pilot_download.json` is from the earlier 35-source pilot; `phase1b_scale_download.json` is the final 151-source run.\n")

    w("## 16. Final Phase 1B conclusion\n")
    w("Phase 1B is complete within its scope: the frozen Phase 1A detector was evaluated on 151 complete ALASKA2 source groups (604 images) with three separated experiments, and on the full 1,500-source BOSSBase test split under five pre-declared transformations, "
      "closing the FR-10 robustness requirement. The honest summary is: **the prototype is a weak, LSB-focused pixel-statistics detector.** It shows a small transferable signal against controlled LSB embedding, "
      "no detectable signal against ALASKA2's DCT-domain steganography, and loses its signal under JPEG re-compression and mild noise while tolerating small crop and mild downscaling. "
      "It is not production-ready, and these experiments do not support any claim beyond the tested conditions.\n")

    w("## 17. Fingerprints\n")
    rows = [["model", str(MODEL), R["fingerprints"]["model"]]]
    for name, path in R["fingerprints_paths"].items():
        rows.append([name, path, R["fingerprints"][name]])
    w(table(["artifact", "path", "SHA-256"], rows))
    w(f"\nFeature-names hash `{M['feature_names_sha256']}`; git HEAD `{M['git_head']}`.\n")

    w("## Appendix A — verification checks\n")
    w(table(["#", "check", "result", "detail"], [[i + 1, c["check"], "PASS" if c["passed"] else "**FAIL**", c["detail"]] for i, c in enumerate(CHECKS)]))
    w("")
    w("## Appendix B — scientific sanity audit\n")
    w(table(["item", "status", "evidence"], R["audit"]))
    return "\n".join(L) + "\n"


def main() -> int:
    t0 = time.perf_counter()
    model_info, bundle = verify_model()
    dataset = verify_dataset()
    alaska = verify_alaska_eval(bundle, dataset["sources"])
    robustness = verify_robustness(bundle)

    # extra claims the narrative relies on -> checked, not assumed
    A = alaska["A"]; BC = alaska["BC"]
    check("narrative: ALASKA2 vs in-domain cover-score difference is not significant (KS p > 0.05)", A["ks"][1] > 0.05, f"p={A['ks'][1]:.3g}")
    check("narrative: Exp C AUC 95% CIs all include 0.5", all(BC[k]["auc_ci"][0] <= 0.5 <= BC[k]["auc_ci"][1] for k in ("C_jmipod", "C_juniward", "C_uerd")))
    check("narrative: no Exp C overall paired test is significant after Holm (family of 5)", all(BC[k]["wilcoxon_p_holm_overall_family5"] > 0.05 for k in ("C_jmipod", "C_juniward", "C_uerd")))
    check("narrative: Exp B paired tests survive Holm (family of 5)", all(BC[k]["wilcoxon_p_holm_overall_family5"] < 0.05 for k in ("B_lsb_0.10", "B_lsb_0.40")))
    check("narrative: no per-QF Exp C test survives Holm (family of 15)", all(BC[k]["by_qf"][q]["wilcoxon_p_holm_family15"] > 0.05 for k in ("C_jmipod", "C_juniward", "C_uerd") for q in ("QF75", "QF90", "QF95")))
    cnd = robustness["conditions"]
    check("narrative: JPEG and noise AUC deltas are large and survive Bonferroni-10", all(cnd[f"{n}|{p}"]["delta_survives_bonferroni10"] for n in ("jpeg_qf90", "jpeg_qf75", "gaussian_noise_sigma2") for p in ("0.10", "0.40")))
    check("narrative: resize at 0.40 and crop at 0.10 are unchanged within uncertainty (Bonferroni-10 CI includes 0)", (not cnd["resize_0.9x|0.40"]["delta_survives_bonferroni10"]) and (not cnd["crop_480_offset13|0.10"]["delta_survives_bonferroni10"]))
    check("narrative: small crop changes AUC by less than 0.01 at both payloads", all(abs(cnd[f"crop_480_offset13|{p}"]["delta_auc"]) < 0.01 for p in ("0.10", "0.40")))
    check("narrative: Exp C dominant-feature change is <= 0.05 sigma for every native method", all(abs(alaska["S1_paired_feature_change"]["lsb_transition_r"][m]["paired_mean_change_in_cover_std_units"]) <= 0.05 for m in ("jmipod", "juniward", "uerd")))

    audit = [
        ["data leakage / source overlap", "checked", "0 sources in >1 split (ALASKA2 manifest); robustness sources are exactly the Phase 1A test split; each source once per condition"],
        ["duplicate contamination", "checked", "0 duplicates across 604 ALASKA2 files (SHA-256 bytes and pixels); Phase 1A excluded the 4 BOSSBase duplicate pairs; exact cross-corpus duplicates are impossible (RGB JPEG vs grayscale PGM)"],
        ["train/test contamination", "checked", "robustness evaluates only test-split sources; ALASKA2 was never used for fitting; identity condition reproduces Phase 1A test metrics exactly"],
        ["accidental retraining / scaler refit", "none", "artifact SHA-256 unchanged before/after every script; artifact scaler is None; no fit() call exists in Phase 1B scripts"],
        ["feature-schema change", "none", "artifact feature_names equal current FEATURE_NAMES; Phase 1A source files identical to git HEAD"],
        ["hidden test-set use in model selection", "none in Phase 1B", "no model selection occurs in Phase 1B; Phase 1A selection used validation only (previous audit)"],
        ["JPEG-format shortcuts", "controlled", "cover and stego share format, size and QF in every ALASKA2 pair (0 QF inconsistencies); B stego is decoded-JPEG + LSB (pixel-domain) — a synthetic scenario, stated as such"],
        ["camera / source bias", "unresolved (limitation)", "no camera metadata; sample drawn from IDs 1–12,813; BOSSBase is a narrow corpus"],
        ["small-sample effects", "acknowledged", "n=151 sources (≈50 per QF); source-bootstrap CIs reported; per-QF results not treated as findings"],
        ["multiple comparisons", "addressed", "Holm within families (5, 15, 3) and Bonferroni-10 intervals for robustness deltas"],
        ["unsupported causal claims", "avoided", "mechanisms for resize and JPEG effects are explicitly hypotheses; feature importance is not used as causal evidence"],
        ["rounded metrics changing interpretation", "checked", "conclusions use full-precision recomputed values (4 dp shown); no conclusion depends on the 4th decimal"],
        ["earlier misinterpretation", "corrected", "the 35-source pilot's '74% ≥ 0.5 = domain-shift FPR' reading was withdrawn after comparing with the in-domain 71.9%"],
    ]

    fp_paths = {
        "alaska2_manifest": str(MANIFEST), "alaska2_scores_csv": str(REPORTS / "phase1b_alaska2_scores.csv"),
        "alaska2_evaluation_json": str(REPORTS / "phase1b_pilot_evaluation.json"), "robustness_json": str(REPORTS / "phase1b_robustness.json"),
        "robustness_scores_csv": str(REPORTS / "phase1b_robustness_scores.csv"), "robustness_ci_json": str(REPORTS / "phase1b_robustness_ci.json"),
        "dataset_inspection_json": str(REPORTS / "phase1b_dataset_inspection.json"), "download_report_json": str(REPORTS / "phase1b_scale_download.json"),
        "phase1a_split_manifest": "data/processed/split_manifest.csv",
    }
    fingerprints = {"model": sha256_file(MODEL), **{k: sha256_file(Path(v)) for k, v in fp_paths.items()}}
    check("model hash still equals the audited value at the end of finalization", fingerprints["model"] == EXPECTED_SHA256)

    R = {
        "generated": time.strftime("%Y-%m-%d %H:%M %Z"), "model": model_info, "dataset": dataset, "alaska": alaska, "robustness": robustness,
        "audit": audit, "fingerprints": fingerprints, "fingerprints_paths": fp_paths,
        "n_checks": len(CHECKS), "n_checks_passed": sum(c["passed"] for c in CHECKS),
    }
    (REPORTS / "phase1b_final_report.md").write_text(render(R))
    slim = {k: v for k, v in R.items() if k not in ()}
    slim["alaska"] = {k: v for k, v in alaska.items() if k != "stored"}
    slim["robustness"] = {k: v for k, v in robustness.items() if k != "stored"}
    slim["dataset"] = {k: v for k, v in dataset.items() if k != "sources"}
    slim["checks"] = CHECKS
    (REPORTS / "phase1b_final_report.json").write_text(json.dumps(slim, indent=2, default=str))
    failed = [c for c in CHECKS if not c["passed"]]
    print(f"\n{R['n_checks_passed']}/{R['n_checks']} checks passed in {time.perf_counter() - t0:.0f}s; wrote phase1b_final_report.md/.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
