"""Deterministic, source-aware train/validation/test splitting.

The unit of splitting is the ORIGINAL COVER IMAGE identity (`source_id`),
never the generated sample. Every stego derivative of a cover image must be
assigned to the same split as its cover, so a cover/stego pair can never
leak across train/validation/test partitions.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
VALID_SPLITS = frozenset({"train", "val", "test"})


def assign_splits(
    source_ids: list[str],
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Deterministically assign each source_id to train/val/test.

    Splitting is done by shuffling the *sorted* set of unique source_ids
    with a seeded RNG, then slicing by ratio. Sorting before shuffling
    removes any dependency on input ordering or Python's hash-randomized
    set iteration, so the same (source_ids, seed, ratios) always produces
    the same assignment.
    """
    ratios = ratios or DEFAULT_RATIOS
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if set(ratios) != VALID_SPLITS:
        raise ValueError(f"ratios must have exactly the keys {sorted(VALID_SPLITS)}, got {sorted(ratios)}")

    unique_ids = sorted(set(source_ids))
    if len(unique_ids) != len(source_ids):
        raise ValueError("Duplicate source_id values passed to assign_splits")
    if not unique_ids:
        raise ValueError("source_ids must not be empty")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_ids))
    shuffled = [unique_ids[i] for i in order]

    n = len(shuffled)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    # test absorbs the rounding remainder so every id is assigned exactly once
    n_val = min(n_val, n - n_train)

    assignment: dict[str, str] = {}
    for sid in shuffled[:n_train]:
        assignment[sid] = "train"
    for sid in shuffled[n_train:n_train + n_val]:
        assignment[sid] = "val"
    for sid in shuffled[n_train + n_val:]:
        assignment[sid] = "test"
    return assignment


def verify_split_integrity(rows: list[dict]) -> list[str]:
    """Return a list of integrity problems; empty means the manifest is clean.

    Checks that every row's split is one of train/val/test, and that a
    given source_id never appears under more than one split value (which
    would indicate a cover/stego leakage across partitions).
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    for row in rows:
        sid = row["source_id"]
        split = row["split"]
        if split not in VALID_SPLITS:
            problems.append(f"{sid}: unknown split value '{split}'")
            continue
        prior = seen.get(sid)
        if prior is not None and prior != split:
            problems.append(f"source_id {sid} appears in multiple splits: {prior} and {split}")
        seen[sid] = split
    return problems
