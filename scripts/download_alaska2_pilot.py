#!/usr/bin/env python
"""Download a small, deterministic ALASKA2 pilot via the Kaggle API,
selecting only source IDs verified (by actual download success/response,
not assumption) to have all four variants: Cover, JMiPOD, JUNIWARD, UERD.

Candidate source IDs are drawn deterministically (seed=42) from the ID
range 1-12813, which was directly observed to exist via a prior Kaggle
file-listing pagination pass (this script does not re-derive that range).
Each candidate is only kept if Cover AND all three stego variants download
successfully; anything less is logged as an incomplete group and skipped,
never silently assumed complete.

Usage:
  python scripts/download_alaska2_pilot.py --target-groups 30 --candidate-pool 80 \\
      --out-dir data/external/ALASKA2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from app.ml.external import select_pilot_sources

COMPETITION = "alaska2-image-steganalysis"
VARIANTS = ("Cover", "JMiPOD", "JUNIWARD", "UERD")
OBSERVED_ID_RANGE = range(1, 12814)  # verified via prior listing pagination (IDs 1-12813 seen)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-groups", type=int, default=30)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/external/ALASKA2")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between download calls")
    return parser.parse_args()


def _download_one(api: KaggleApi, variant: str, source_id: str, out_dir: Path, delay: float) -> tuple[bool, str]:
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    remote_name = f"{variant}/{source_id}.jpg"
    for attempt in range(2):
        try:
            api.competition_download_file(COMPETITION, remote_name, path=str(variant_dir), quiet=True)
            time.sleep(delay)
            downloaded = variant_dir / f"{source_id}.jpg"
            if downloaded.exists() and downloaded.stat().st_size > 0:
                return True, "ok"
            return False, "downloaded_but_empty_or_missing"
        except Exception as exc:  # noqa: BLE001 - we need to classify Kaggle's varied exception types
            msg = str(exc)
            if "429" in msg and attempt == 0:
                time.sleep(3.0)
                continue
            return False, msg[:200]
    return False, "retry_exhausted"


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    candidate_ids = select_pilot_sources(
        [f"{i:05d}" for i in OBSERVED_ID_RANGE], n=args.candidate_pool, seed=args.seed
    )
    print(f"Candidate pool: {len(candidate_ids)} deterministically-selected IDs (seed={args.seed})")

    complete_groups: list[str] = []
    incomplete_groups: dict[str, dict[str, str]] = {}
    per_variant_success = {v: 0 for v in VARIANTS}
    per_variant_failure = {v: 0 for v in VARIANTS}

    for source_id in candidate_ids:
        if len(complete_groups) >= args.target_groups:
            break

        results: dict[str, str] = {}
        cover_ok, cover_reason = _download_one(api, "Cover", source_id, out_dir, args.delay)
        results["Cover"] = "ok" if cover_ok else cover_reason
        if cover_ok:
            per_variant_success["Cover"] += 1
        else:
            per_variant_failure["Cover"] += 1
            incomplete_groups[source_id] = results
            print(f"  {source_id}: Cover missing/failed ({cover_reason}) -- skipping remaining variants")
            continue

        all_ok = True
        for variant in ("JMiPOD", "JUNIWARD", "UERD"):
            ok, reason = _download_one(api, variant, source_id, out_dir, args.delay)
            results[variant] = "ok" if ok else reason
            if ok:
                per_variant_success[variant] += 1
            else:
                per_variant_failure[variant] += 1
                all_ok = False

        if all_ok:
            complete_groups.append(source_id)
            print(f"  {source_id}: COMPLETE (Cover+JMiPOD+JUNIWARD+UERD) [{len(complete_groups)}/{args.target_groups}]")
        else:
            incomplete_groups[source_id] = results
            print(f"  {source_id}: INCOMPLETE {results}")

    summary = {
        "candidate_pool_size": len(candidate_ids),
        "target_groups": args.target_groups,
        "complete_group_count": len(complete_groups),
        "complete_group_ids": complete_groups,
        "incomplete_group_count": len(incomplete_groups),
        "incomplete_groups": incomplete_groups,
        "per_variant_success": per_variant_success,
        "per_variant_failure": per_variant_failure,
        "out_dir": str(out_dir),
    }
    report_path = Path("data/reports/phase1b_pilot_download.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote download summary to {report_path}")


if __name__ == "__main__":
    main()
