#!/usr/bin/env python
"""Generate the leakage-safe BOSSBase cover/stego feature dataset.

Produces:
  <output-dir>/split_manifest.csv        one row per source_id: (source_id, split)
  <output-dir>/features_payload_<p>.csv  one row per sample (the shared cover
                                          row plus one stego row per payload
                                          level), holding the full
                                          extract_features() vector

Raw stego pixel data is never written to disk: each stego image is built in
memory from the original BOSSBase cover, immediately reduced to its feature
vector, and discarded. This keeps disk usage independent of dataset size and
payload count while remaining fully reproducible -- the same
(seed, source_id, payload) always regenerates identical embedded pixels.

The train/val/test split is assigned once, by source_id, before any stego
image is generated, and is shared by every payload level so a given cover
image and all of its stego derivatives always land in the same split.

Usage:
  python scripts/generate_dataset.py \\
      --bossbase-root data/raw/BOSSBase \\
      --output-dir data/processed \\
      --seed 42 \\
      --payloads 0.01,0.05,0.10,0.20,0.40 \\
      --n-sources 200   # omit (or pass a value >= dataset size) for the full run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.ml.dataset import find_duplicate_source_ids, index_bossbase
from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features
from app.ml.lsb import DEFAULT_PAYLOAD_LEVELS, embed_lsb, sample_seed
from app.ml.split import assign_splits, verify_split_integrity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bossbase-root", default="data/raw/BOSSBase")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--payloads", default=",".join(str(p) for p in DEFAULT_PAYLOAD_LEVELS))
    parser.add_argument(
        "--n-sources",
        type=int,
        default=None,
        help="Subsample to this many source images (for pilot runs). Omit for the full dataset.",
    )
    parser.add_argument("--split-ratios", default="0.70,0.15,0.15", help="train,val,test")
    return parser.parse_args()


def _select_sources(source_ids: list[str], n_sources: int | None, seed: int) -> list[str]:
    if n_sources is None or n_sources >= len(source_ids):
        return source_ids
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(source_ids))
    return sorted(source_ids[i] for i in order[:n_sources])


def _feature_row(
    sample_id: str,
    source_id: str,
    split: str,
    label: int,
    payload: float,
    width: int,
    height: int,
    vector: np.ndarray,
) -> dict:
    row = {
        "sample_id": sample_id,
        "source_id": source_id,
        "split": split,
        "label": label,
        "payload": payload,
        "width": width,
        "height": height,
    }
    row.update({name: float(v) for name, v in zip(FEATURE_NAMES, vector)})
    return row


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_levels = [float(p) for p in args.payloads.split(",") if p]
    ratio_parts = [float(x) for x in args.split_ratios.split(",")]
    ratios = {"train": ratio_parts[0], "val": ratio_parts[1], "test": ratio_parts[2]}

    print(f"Indexing BOSSBase at {args.bossbase_root} ...")
    index_result = index_bossbase(args.bossbase_root)
    print(f"  valid images: {index_result.valid_count}, errors: {index_result.error_count}")
    for err in index_result.errors[:10]:
        print(f"  WARNING: {err}")
    if index_result.error_count > 10:
        print(f"  ... and {index_result.error_count - 10} more warnings")

    duplicates = find_duplicate_source_ids(index_result.records)
    if duplicates:
        print(
            f"Found {len(duplicates)} exact pixel-content duplicate(s) among indexed covers; "
            f"excluding the duplicate copy to prevent cross-split content leakage:"
        )
        for dup_id, original_id in duplicates.items():
            print(f"  excluding {dup_id} (identical content to {original_id})")

    all_source_ids = [r.source_id for r in index_result.records if r.source_id not in duplicates]
    selected_ids = _select_sources(all_source_ids, args.n_sources, args.seed)
    records_by_id = {r.source_id: r for r in index_result.records}
    print(
        f"Using {len(selected_ids)} of {len(all_source_ids)} indexed source images "
        f"({'full dataset' if args.n_sources is None else 'pilot subset'})."
    )

    split_map = assign_splits(selected_ids, seed=args.seed, ratios=ratios)
    split_counts = {"train": 0, "val": 0, "test": 0}
    for split in split_map.values():
        split_counts[split] += 1
    print(f"Split assignment (by source_id): {split_counts}")

    split_manifest_path = output_dir / "split_manifest.csv"
    pd.DataFrame(
        [{"source_id": sid, "split": split_map[sid]} for sid in selected_ids]
    ).to_csv(split_manifest_path, index=False)
    print(f"Wrote {split_manifest_path}")

    cover_rows: list[dict] = []
    stego_rows: dict[float, list[dict]] = {p: [] for p in payload_levels}

    start = time.perf_counter()
    for i, source_id in enumerate(selected_ids, start=1):
        record = records_by_id[source_id]
        split = split_map[source_id]
        with Image.open(record.path) as img:
            pixels = np.asarray(img, dtype=np.uint8)

        cover_vector, _ = extract_features(Image.fromarray(pixels).convert("RGB"))
        cover_rows.append(
            _feature_row(
                f"{source_id}_cover", source_id, split, 0, 0.0, record.width, record.height, cover_vector
            )
        )

        for payload in payload_levels:
            seed = sample_seed(args.seed, source_id, payload)
            stego_pixels = embed_lsb(pixels, payload, seed)
            stego_vector, _ = extract_features(Image.fromarray(stego_pixels).convert("RGB"))
            stego_rows[payload].append(
                _feature_row(
                    f"{source_id}_stego_p{payload:.2f}",
                    source_id,
                    split,
                    1,
                    payload,
                    record.width,
                    record.height,
                    stego_vector,
                )
            )

        if i % 500 == 0 or i == len(selected_ids):
            elapsed = time.perf_counter() - start
            print(f"  processed {i}/{len(selected_ids)} source images ({elapsed:.1f}s elapsed)")

    integrity_rows = [{"source_id": r["source_id"], "split": r["split"]} for r in cover_rows]
    for rows in stego_rows.values():
        integrity_rows.extend({"source_id": r["source_id"], "split": r["split"]} for r in rows)
    integrity_problems = verify_split_integrity(integrity_rows)
    if integrity_problems:
        raise RuntimeError(f"Split integrity check FAILED: {integrity_problems[:5]}")
    print("Split integrity check passed: no source_id crosses splits.")

    cover_df = pd.DataFrame(cover_rows)
    written_paths = []
    for payload in payload_levels:
        combined = pd.concat([cover_df, pd.DataFrame(stego_rows[payload])], ignore_index=True)
        out_path = output_dir / f"features_payload_{payload:.2f}.csv"
        combined.to_csv(out_path, index=False)
        written_paths.append(out_path)
        by_split_label = combined.groupby(["split", "label"]).size().to_dict()
        print(f"Wrote {out_path} ({len(combined)} rows) -- {by_split_label}")

    print(f"\nDone. feature_schema_version={FEATURE_SCHEMA_VERSION}, seed={args.seed}")
    print(f"Files: {split_manifest_path}, " + ", ".join(str(p) for p in written_paths))


if __name__ == "__main__":
    main()
