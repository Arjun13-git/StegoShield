import pytest

from app.ml.split import assign_splits, verify_split_integrity


def _ids(n: int) -> list[str]:
    return [f"cover_{i:05d}" for i in range(n)]


def test_split_covers_every_source_id_exactly_once() -> None:
    ids = _ids(200)
    assignment = assign_splits(ids, seed=42)
    assert set(assignment.keys()) == set(ids)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_split_ratios_are_approximately_correct() -> None:
    ids = _ids(1000)
    assignment = assign_splits(ids, seed=42)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    assert counts["train"] == 700
    assert counts["val"] == 150
    assert counts["test"] == 150


def test_split_is_deterministic_for_fixed_seed() -> None:
    ids = _ids(300)
    a1 = assign_splits(ids, seed=42)
    a2 = assign_splits(ids, seed=42)
    assert a1 == a2


def test_split_changes_with_different_seed() -> None:
    ids = _ids(300)
    a1 = assign_splits(ids, seed=42)
    a2 = assign_splits(ids, seed=99)
    assert a1 != a2


def test_split_is_independent_of_input_order() -> None:
    ids = _ids(200)
    shuffled = list(reversed(ids))
    a1 = assign_splits(ids, seed=42)
    a2 = assign_splits(shuffled, seed=42)
    assert a1 == a2


def test_split_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        assign_splits(["a", "a", "b"], seed=42)


def test_split_rejects_bad_ratios() -> None:
    with pytest.raises(ValueError):
        assign_splits(_ids(10), seed=42, ratios={"train": 0.5, "val": 0.2, "test": 0.2})


def test_verify_split_integrity_flags_source_crossing_splits() -> None:
    rows = [
        {"source_id": "a", "split": "train"},
        {"source_id": "a", "split": "test"},  # same cover, different split: leakage
        {"source_id": "b", "split": "val"},
        {"source_id": "b", "split": "val"},
    ]
    problems = verify_split_integrity(rows)
    assert len(problems) == 1
    assert "source_id a" in problems[0]


def test_verify_split_integrity_flags_unknown_split_value() -> None:
    rows = [{"source_id": "a", "split": "bogus"}]
    problems = verify_split_integrity(rows)
    assert len(problems) == 1
    assert "unknown split value" in problems[0]


def test_verify_split_integrity_clean_manifest_has_no_problems() -> None:
    rows = [
        {"source_id": "a", "split": "train"},
        {"source_id": "a", "split": "train"},
        {"source_id": "b", "split": "test"},
    ]
    assert verify_split_integrity(rows) == []


def test_cover_and_stego_rows_stay_together_end_to_end() -> None:
    """Simulates the real manifest shape: one cover row + several stego
    rows (different payload levels) per source_id. None should cross splits.
    """
    ids = _ids(50)
    split_map = assign_splits(ids, seed=42)
    rows = []
    for sid in ids:
        rows.append({"source_id": sid, "split": split_map[sid], "label": 0})
        for payload in (0.01, 0.05, 0.10, 0.20, 0.40):
            rows.append({"source_id": sid, "split": split_map[sid], "label": 1, "payload": payload})
    assert verify_split_integrity(rows) == []
