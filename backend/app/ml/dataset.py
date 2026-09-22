"""Deterministic dataset indexing for the BOSSBase cover-image source.

Produces a manifest of the available cover images keyed by a stable
`source_id` derived from each file's name stem, never from filesystem
iteration order (which is not guaranteed stable across platforms).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXPECTED_MODE = "L"


@dataclass(frozen=True)
class CoverRecord:
    source_id: str
    path: str
    width: int
    height: int
    content_hash: str


@dataclass(frozen=True)
class DatasetIndexResult:
    records: list[CoverRecord]
    errors: list[str]

    @property
    def valid_count(self) -> int:
        return len(self.records)

    @property
    def error_count(self) -> int:
        return len(self.errors)


def _sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    # Numeric filenames (BOSSBase's convention) sort naturally; anything
    # else falls back to lexicographic order. Either way the ordering is a
    # pure function of the filename, never of directory iteration order.
    if stem.isdigit():
        return (0, f"{int(stem):020d}")
    return (1, stem)


def index_bossbase(
    root: str | Path,
    *,
    expected_dimensions: tuple[int, int] | None = (512, 512),
    expected_count: int | None = None,
) -> DatasetIndexResult:
    """Index a BOSSBase-style directory of `.pgm` grayscale cover images.

    Validates that each file decodes successfully, is grayscale, and (when
    `expected_dimensions` is given) matches the expected size. Corrupt or
    mismatched files are collected into `errors` rather than aborting the
    whole scan, so a handful of bad files doesn't block the entire dataset.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"BOSSBase directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"BOSSBase path is not a directory: {root}")

    paths = sorted(root.glob("*.pgm"), key=_sort_key)
    if not paths:
        raise ValueError(f"No .pgm files found under {root}")

    records: list[CoverRecord] = []
    errors: list[str] = []

    for path in paths:
        source_id = path.stem
        try:
            with Image.open(path) as img:
                img.load()  # forces full decode; raises on truncated/corrupt data
                width, height = img.size
                mode = img.mode
                pixels = np.asarray(img, dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            errors.append(f"{path.name}: unreadable/corrupt image ({exc})")
            continue

        if mode != EXPECTED_MODE:
            errors.append(f"{path.name}: unexpected mode '{mode}', expected grayscale 'L'")
            continue
        if expected_dimensions is not None and (width, height) != expected_dimensions:
            errors.append(
                f"{path.name}: unexpected dimensions {width}x{height}, "
                f"expected {expected_dimensions[0]}x{expected_dimensions[1]}"
            )
            continue

        content_hash = hashlib.md5(pixels.tobytes()).hexdigest()
        records.append(
            CoverRecord(source_id=source_id, path=str(path), width=width, height=height, content_hash=content_hash)
        )

    if not records:
        raise ValueError(
            f"No valid images indexed under {root}; first error: "
            f"{errors[0] if errors else 'unknown'}"
        )

    if expected_count is not None and len(records) < expected_count:
        errors.append(
            f"Expected at least {expected_count} valid images, found only {len(records)}"
        )

    id_counts = Counter(r.source_id for r in records)
    duplicate_ids = [sid for sid, count in id_counts.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate source_id values detected in BOSSBase index: {sorted(duplicate_ids)}")

    return DatasetIndexResult(records=records, errors=errors)


def find_duplicate_source_ids(records: list[CoverRecord]) -> dict[str, str]:
    """Detect exact pixel-content duplicates among indexed cover images.

    Two different filenames can carry byte-for-byte identical pixel data.
    Source-aware splitting only guards against a *single* source_id
    appearing in multiple splits -- it has no way to know that two
    different source_ids are secretly the same image. If such a pair were
    split across train/test, the "held-out" evaluation would partly be
    evaluating on an image the model effectively already saw, silently
    inflating measured performance.

    Returns {duplicate_source_id: original_source_id}, in index order, so
    the first occurrence of a given pixel-content hash is treated as the
    canonical copy and every later occurrence is reported as a duplicate to
    exclude.
    """
    seen_hash_to_id: dict[str, str] = {}
    duplicates: dict[str, str] = {}
    for record in records:
        original = seen_hash_to_id.get(record.content_hash)
        if original is not None:
            duplicates[record.source_id] = original
        else:
            seen_hash_to_id[record.content_hash] = record.source_id
    return duplicates
