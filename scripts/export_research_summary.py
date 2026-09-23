"""Export the aggregate Phase 1A/1B results the Research page displays.

Reads the generated (git-ignored) reports under data/reports/ and writes ONE
small JSON file, frontend/src/data/research-summary.json. Nothing is typed in by
hand: every number in that file is copied from a report, so the UI cannot show a
metric that no experiment produced. Only aggregate statistics are exported; no
images, per-image scores or manifests (ALASKA2 must not be redistributed).

Usage (repository root):
    python scripts/export_research_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "data" / "reports"
OUT = REPO / "frontend" / "src" / "data" / "research-summary.json"

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
ROBUSTNESS_ORDER = ("jpeg_qf90", "jpeg_qf75", "gaussian_noise_sigma2", "crop_480_offset13", "resize_0.9x")


def load(name: str) -> dict:
    path = REPORTS / name
    if not path.is_file():
        raise SystemExit(f"missing report: {path} (run the Phase 1 scripts first)")
    return json.loads(path.read_text())


def pick(metrics: dict) -> dict:
    return {k: metrics[k] for k in METRICS}


def main() -> None:
    comparison = load("model_comparison_payload_0.10.json")
    payload = load("payload_sensitivity.json")
    final = load("phase1b_final_report.json")

    counts = comparison["dataset_counts"]  # samples = 2 x sources (one cover + one stego per source)
    sources = {split: n // 2 for split, n in counts.items()}

    candidates = {
        name: {"validation": pick(c["val_metrics"]), "validation_confusion_matrix": c["val_metrics"]["confusion_matrix"]}
        for name, c in comparison["candidates"].items()
    }

    alaska = final["alaska"]["BC"]
    external = {
        key: {"n_sources": v["n"], "roc_auc": v["auc"], "roc_auc_ci95": v["auc_ci"]}
        for key, v in alaska.items()
    }
    cover_scores = final["alaska"]["A"]

    conditions = final["robustness"]["conditions"]
    robustness = []
    for cond in ROBUSTNESS_ORDER:
        row = {"condition": cond}
        for bpp in ("0.10", "0.40"):
            c = conditions[f"{cond}|{bpp}"]
            row[bpp] = {"roc_auc": c["auc"], "delta_auc": c["delta_auc"], "delta_ci95": c["delta_ci95"]}
        robustness.append(row)

    summary = {
        "generated_from": sorted(
            ["model_comparison_payload_0.10.json", "payload_sensitivity.json", "phase1b_final_report.json"]
        ),
        "model": {
            "sha256": final["model"]["sha256"],
            "name": final["model"]["model_name"],
            "version": final["model"]["model_version"],
            "feature_schema_version": final["model"]["feature_schema_version"],
            "n_features": final["model"]["n_features"],
            "random_seed": final["model"]["random_seed"],
            "hyperparameters": final["model"]["hyperparameters"],
        },
        "phase1a": {
            "feature_schema_version": comparison["feature_schema_version"],
            "seed": comparison["seed"],
            "samples_per_split_at_0.10_bpp": counts,
            "sources_per_split": sources,
            "candidates": candidates,
            "selected_model": comparison["selected_model"],
            "selection_rule": comparison["selection_rule"],
            "test": {**pick(comparison["test_metrics"]), "confusion_matrix": comparison["test_metrics"]["confusion_matrix"],
                     "n_samples": comparison["test_metrics"]["n_samples"]},
        },
        "payload_sensitivity": {
            "note": payload["note"],
            "split": payload["split_evaluated"],
            "results": [{"payload_bpp": r["payload"], **pick(r)} for r in payload["results"]],
        },
        "phase1b": {
            "n_source_groups": final["dataset"]["n_files"] // 4,
            "n_files": final["dataset"]["n_files"],
            "formats": final["dataset"]["formats"],
            "sources_per_quality_factor": final["dataset"]["qf_source_counts"],
            "external": external,
            "clean_cover_fraction_score_ge_0.5": {
                "alaska2": cover_scores["alaska"]["frac_ge_0.5"],
                "bossbase_test": cover_scores["boss"]["frac_ge_0.5"],
            },
            "robustness_n_test_sources": final["robustness"]["n_test_sources"],
            "robustness": robustness,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
