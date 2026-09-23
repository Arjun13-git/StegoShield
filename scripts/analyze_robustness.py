#!/usr/bin/env python
"""Source-level (paired) uncertainty for the Phase 1B robustness results.

Reads the per-source scores written by scripts/robustness_eval.py and, for
every (transformation, payload) condition, reports ROC-AUC with a paired
bootstrap 95% CI (whole sources are resampled, so a cover and its stego are
never separated) and the paired difference in AUC versus the untransformed
condition, using the SAME resampled sources for both conditions. Also
reports cover false-positive rate and stego true-positive rate at the fixed
0.5 threshold, because accuracy/F1/recall alone can hide a degenerate
"predict everything positive" behaviour.

Usage:
  python scripts/analyze_robustness.py \\
      --scores data/reports/phase1b_robustness_scores.csv \\
      --report-out data/reports/phase1b_robustness_ci.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RESAMPLES = 2000


def _auc(cover: np.ndarray, stego: np.ndarray) -> float:
    y = np.array([0] * len(cover) + [1] * len(stego))
    return float(roc_auc_score(y, np.concatenate([cover, stego])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", default="data/reports/phase1b_robustness_scores.csv")
    p.add_argument("--report-out", default="data/reports/phase1b_robustness_ci.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.scores, dtype={"source_id": str}, float_precision="round_trip")  # exact ties matter for AUC
    rng = np.random.default_rng(args.seed)
    out: dict = {"unit": "source (paired cover/stego); bootstrap resamples whole sources", "conditions": {}}

    for payload in sorted(df["payload"].unique()):
        sub = df[df["payload"] == payload]
        base = sub[sub["transform"] == "none"].set_index("source_id")
        for name in sub["transform"].unique():
            cur = sub[sub["transform"] == name].set_index("source_id")
            common = base.index.intersection(cur.index)
            b_c, b_s = base.loc[common, "cover_score"].to_numpy(), base.loc[common, "stego_score"].to_numpy()
            c_c, c_s = cur.loc[common, "cover_score"].to_numpy(), cur.loc[common, "stego_score"].to_numpy()
            n = len(common)
            aucs, deltas = [], []
            for _ in range(RESAMPLES):
                idx = rng.integers(0, n, size=n)
                a_cur = _auc(c_c[idx], c_s[idx])
                aucs.append(a_cur)
                deltas.append(a_cur - _auc(b_c[idx], b_s[idx]))
            out["conditions"][f"{name}|payload={payload:.2f}"] = {
                "n_source_pairs": n,
                "roc_auc": _auc(c_c, c_s),
                "roc_auc_bootstrap95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
                "delta_auc_vs_untransformed": _auc(c_c, c_s) - _auc(b_c, b_s),
                "delta_auc_bootstrap95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
                "cover_fpr_at_0.5": float((c_c >= 0.5).mean()),
                "stego_tpr_at_0.5": float((c_s >= 0.5).mean()),
                "mean_cover_score": float(c_c.mean()),
                "mean_stego_score": float(c_s.mean()),
            }

    Path(args.report_out).write_text(json.dumps(out, indent=2))
    for k, v in out["conditions"].items():
        print(f"{k:<40} n={v['n_source_pairs']} AUC={v['roc_auc']:.3f} {np.round(v['roc_auc_bootstrap95'], 3)} "
              f"dAUC={v['delta_auc_vs_untransformed']:+.3f} {np.round(v['delta_auc_bootstrap95'], 3)} "
              f"FPR@.5={v['cover_fpr_at_0.5']:.3f} TPR@.5={v['stego_tpr_at_0.5']:.3f}")
    print(f"Wrote {args.report_out}")


if __name__ == "__main__":
    main()
