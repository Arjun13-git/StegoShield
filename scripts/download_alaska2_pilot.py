#!/usr/bin/env python
"""Download a small, deterministic ALASKA2 pilot via the Kaggle API,
keeping only source IDs verified (by actual download success AND a
successful decode, not assumption) to have all four variants: Cover,
JMiPOD, JUNIWARD, UERD.

Candidate source IDs are drawn deterministically (seed=42) from the ID
range 1-12813, which was directly observed to exist via a prior Kaggle
file-listing pagination pass (this script never re-paginates the listing).
Because select_pilot_sources() permutes the full ID list with a seeded RNG
and slices a prefix, a larger --candidate-pool is always a superset of a
smaller one for the same seed, so scaling up never reshuffles sources that
were already selected.

Files already present locally (and decodable) are REUSED, not re-downloaded.
A source group counts as complete only after Cover AND all three stego
variants exist locally and decode successfully. Each variant lives in its
own directory, so identical basenames can never overwrite one another.

Kaggle pacing: a fixed delay between calls plus exponential backoff on
HTTP 429. 404 (ID gap) is treated as "source does not exist", not retried.

Usage:
  python scripts/download_alaska2_pilot.py --candidate-pool 160 --target-groups 160 \\
      --out-dir data/external/ALASKA2 --seed 42 --dry-run     # inspect first
  python scripts/download_alaska2_pilot.py --candidate-pool 160 --target-groups 160 \\
      --out-dir data/external/ALASKA2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.ml.external import select_pilot_sources

COMPETITION = "alaska2-image-steganalysis"
VARIANTS = ("Cover", "JMiPOD", "JUNIWARD", "UERD")
OBSERVED_ID_RANGE = range(1, 12814)  # verified via prior listing pagination (IDs 1-12813 seen)
MAX_429_RETRIES = 5
MAX_OTHER_RETRIES = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-groups", type=int, default=30, help="stop once this many complete groups exist")
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/external/ALASKA2")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between download calls")
    parser.add_argument("--backoff-base", type=float, default=5.0, help="first 429 backoff in seconds (doubles each retry)")
    parser.add_argument("--report-out", default="data/reports/phase1b_pilot_download.json")
    parser.add_argument("--dry-run", action="store_true", help="report local coverage of the candidate pool; no API calls")
    return parser.parse_args()


def _valid_local(path: Path) -> bool:
    """True only if the file exists and fully decodes as an image."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False


class Counters:
    def __init__(self) -> None:
        self.reused = 0
        self.downloaded = 0
        self.rate_limit_events = 0
        self.download_seconds = 0.0
        self.downloaded_bytes = 0


def _fetch_one(api, variant: str, source_id: str, out_dir: Path, args, counters: Counters) -> tuple[str, str]:
    """Return (status, detail); status in {reused, downloaded, not_found, error}."""
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    target = variant_dir / f"{source_id}.jpg"

    if _valid_local(target):
        counters.reused += 1
        return "reused", "valid_local_file"

    remote_name = f"{variant}/{source_id}.jpg"
    other_attempts = 0
    for attempt in range(MAX_429_RETRIES + 1):
        start = time.perf_counter()
        try:
            api.competition_download_file(COMPETITION, remote_name, path=str(variant_dir), quiet=True)
            counters.download_seconds += time.perf_counter() - start
            time.sleep(args.delay)
            if _valid_local(target):
                counters.downloaded += 1
                counters.downloaded_bytes += target.stat().st_size
                return "downloaded", "ok"
            if target.exists():
                target.unlink()  # our own corrupt/partial download; safe to remove
            return "error", "downloaded_but_not_decodable"
        except Exception as exc:  # noqa: BLE001 - Kaggle raises varied exception types
            counters.download_seconds += time.perf_counter() - start
            msg = str(exc)
            if "404" in msg:
                return "not_found", "404 Not Found"
            if "429" in msg:
                counters.rate_limit_events += 1
                if attempt < MAX_429_RETRIES:
                    time.sleep(args.backoff_base * (2**attempt))
                    continue
                return "error", "rate_limited_retries_exhausted"
            other_attempts += 1
            if other_attempts <= MAX_OTHER_RETRIES:
                time.sleep(args.backoff_base)
                continue
            return "error", msg[:200]
    return "error", "retry_exhausted"


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_ids = select_pilot_sources(
        [f"{i:05d}" for i in OBSERVED_ID_RANGE], n=args.candidate_pool, seed=args.seed
    )
    print(f"Candidate pool: {len(candidate_ids)} deterministically-selected IDs (seed={args.seed})")

    locally_complete = [
        sid for sid in candidate_ids if all(_valid_local(out_dir / v / f"{sid}.jpg") for v in VARIANTS)
    ]
    print(f"Already complete locally (all 4 variants decodable): {len(locally_complete)}")

    if args.dry_run:
        need = len(candidate_ids) - len(locally_complete)
        print(f"DRY RUN: {need} candidate IDs would still need download attempts; no API calls made.")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    counters = Counters()
    wall_start = time.perf_counter()
    complete_groups: list[str] = []
    gap_ids: list[str] = []  # Cover returned 404: source ID does not exist
    error_groups: dict[str, dict[str, str]] = {}
    incomplete_groups: dict[str, dict[str, str]] = {}  # Cover ok but a stego variant missing/failed

    for source_id in candidate_ids:
        if len(complete_groups) >= args.target_groups:
            break

        results: dict[str, str] = {}
        status, detail = _fetch_one(api, "Cover", source_id, out_dir, args, counters)
        results["Cover"] = f"{status}:{detail}"
        if status == "not_found":
            gap_ids.append(source_id)
            print(f"  {source_id}: ID gap (Cover 404) -- skipped")
            continue
        if status == "error":
            error_groups[source_id] = results
            print(f"  {source_id}: Cover ERROR ({detail}) -- skipped")
            continue

        all_ok = True
        for variant in ("JMiPOD", "JUNIWARD", "UERD"):
            vstatus, vdetail = _fetch_one(api, variant, source_id, out_dir, args, counters)
            results[variant] = f"{vstatus}:{vdetail}"
            if vstatus not in ("reused", "downloaded"):
                all_ok = False

        if all_ok:
            complete_groups.append(source_id)
            print(f"  {source_id}: COMPLETE [{len(complete_groups)}]")
        else:
            incomplete_groups[source_id] = results
            print(f"  {source_id}: INCOMPLETE {results}")

    summary = {
        "candidate_pool_size": len(candidate_ids),
        "target_groups": args.target_groups,
        "seed": args.seed,
        "complete_group_count": len(complete_groups),
        "complete_group_ids": complete_groups,
        "id_gap_count": len(gap_ids),
        "id_gap_ids": gap_ids,
        "cover_error_groups": error_groups,
        "incomplete_group_count": len(incomplete_groups),
        "incomplete_groups": incomplete_groups,
        "files_reused_from_local": counters.reused,
        "files_downloaded_this_run": counters.downloaded,
        "bytes_downloaded_this_run": counters.downloaded_bytes,
        "rate_limit_429_events": counters.rate_limit_events,
        "seconds_inside_download_calls": round(counters.download_seconds, 1),
        "wall_seconds": round(time.perf_counter() - wall_start, 1),
        "pacing": {"delay_seconds": args.delay, "backoff_base_seconds": args.backoff_base},
        "out_dir": str(out_dir),
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "complete_group_ids"}, indent=2))
    print(f"\nWrote download summary to {report_path}")


if __name__ == "__main__":
    main()
