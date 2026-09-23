from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.external import (
    MANIFEST_COLUMNS,
    ManifestRow,
    content_hash,
    discover_root,
    find_duplicate_files,
    inspect_jpeg_quality,
    pair_across_variants,
    select_pilot_sources,
)
from app.ml.features import FEATURE_NAMES
from app.ml.split import assign_splits, verify_split_integrity


def _write_jpeg(path: Path, seed: int = 0, size: tuple[int, int] = (32, 32), quality: int = 90) -> None:
    arr = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=quality)


# ---------------------------------------------------------------------------
# discover_root
# ---------------------------------------------------------------------------


def test_discover_root_missing_directory_reports_not_exists(tmp_path: Path) -> None:
    result = discover_root(tmp_path / "does_not_exist")
    assert result.root_exists is False
    assert result.subdirectories == []


def test_discover_root_finds_subdirectories_and_extensions(tmp_path: Path) -> None:
    cover = tmp_path / "Cover"
    cover.mkdir()
    for i in range(3):
        _write_jpeg(cover / f"{i}.jpg", seed=i)

    result = discover_root(tmp_path)
    assert result.root_exists is True
    assert len(result.subdirectories) == 1
    assert result.subdirectories[0].name == "Cover"
    assert result.subdirectories[0].file_count == 3
    assert result.subdirectories[0].extensions == {".jpg": 3}


def test_discover_root_reports_loose_files_separately(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("hello")
    result = discover_root(tmp_path)
    assert result.loose_files == ["readme.txt"]
    assert result.subdirectories == []


# ---------------------------------------------------------------------------
# pair_across_variants
# ---------------------------------------------------------------------------


def test_pair_across_variants_finds_common_stems(tmp_path: Path) -> None:
    cover = tmp_path / "Cover"
    stego = tmp_path / "JMiPOD"
    cover.mkdir()
    stego.mkdir()
    for i in range(5):
        _write_jpeg(cover / f"{i:05d}.jpg", seed=i)
    for i in range(3):  # only 3 of the 5 have a stego counterpart
        _write_jpeg(stego / f"{i:05d}.jpg", seed=100 + i)

    result = pair_across_variants({"cover": cover, "jmipod": stego})
    assert result.common_stems == 3
    assert result.total_unique_stems == 5
    assert result.per_variant_only_count["cover"] == 2
    assert result.per_variant_only_count["jmipod"] == 0


def test_pair_across_variants_no_overlap(tmp_path: Path) -> None:
    a = tmp_path / "A"
    b = tmp_path / "B"
    a.mkdir()
    b.mkdir()
    _write_jpeg(a / "1.jpg", seed=1)
    _write_jpeg(b / "2.jpg", seed=2)

    result = pair_across_variants({"a": a, "b": b})
    assert result.common_stems == 0
    assert result.per_variant_only_count == {"a": 1, "b": 1}


# ---------------------------------------------------------------------------
# JPEG quality inspection -- must never fabricate a QF label
# ---------------------------------------------------------------------------


def test_inspect_jpeg_quality_matches_exact_reference_table(tmp_path: Path) -> None:
    """A file whose luma quantization table is byte-for-byte identical to
    PIL's own standard QF=90 table should be reported as a VERIFIED match --
    this is an exact-table comparison, not a fabricated guess.
    """
    path = tmp_path / "img.jpg"
    _write_jpeg(path, quality=90)
    info = inspect_jpeg_quality(path)
    assert info.is_jpeg is True
    assert info.quantization_table_digest is not None
    assert info.matched_known_qf == 90
    assert info.certainty == "VERIFIED"


def test_inspect_jpeg_quality_never_fabricates_qf_for_nonstandard_table(tmp_path: Path) -> None:
    """A quality setting that does NOT correspond to one of the three
    candidate reference tables (75/90/95) must never be assigned a QF label.
    """
    path = tmp_path / "img.jpg"
    _write_jpeg(path, quality=60)
    info = inspect_jpeg_quality(path)
    assert info.is_jpeg is True
    assert info.matched_known_qf is None  # never fabricated for a non-matching table
    assert info.certainty == "NOT_VERIFIED"


def test_inspect_jpeg_quality_different_qualities_give_different_digests(tmp_path: Path) -> None:
    p1 = tmp_path / "q50.jpg"
    p2 = tmp_path / "q95.jpg"
    arr = np.random.default_rng(0).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p1, format="JPEG", quality=50)
    Image.fromarray(arr).save(p2, format="JPEG", quality=95)

    info1 = inspect_jpeg_quality(p1)
    info2 = inspect_jpeg_quality(p2)
    assert info1.quantization_table_digest != info2.quantization_table_digest


def test_inspect_jpeg_quality_non_jpeg_file(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")
    info = inspect_jpeg_quality(path)
    assert info.is_jpeg is False
    assert info.certainty == "NOT_AVAILABLE"


def test_inspect_jpeg_quality_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not a real jpeg")
    info = inspect_jpeg_quality(path)
    assert info.certainty == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_find_duplicate_files_detects_identical_pixel_content(tmp_path: Path) -> None:
    arr = np.random.default_rng(0).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    p3 = tmp_path / "c.jpg"
    Image.fromarray(arr).save(p1, format="JPEG", quality=100)
    Image.fromarray(arr).save(p2, format="JPEG", quality=100)
    _write_jpeg(p3, seed=99)  # distinct content

    groups = find_duplicate_files([p1, p2, p3])
    assert len(groups) == 1
    (members,) = groups.values()
    assert set(members) == {str(p1), str(p2)}


def test_find_duplicate_files_empty_when_all_unique(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        _write_jpeg(p, seed=i)
        paths.append(p)
    assert find_duplicate_files(paths) == {}


def test_content_hash_returns_none_for_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"garbage")
    assert content_hash(path) is None


# ---------------------------------------------------------------------------
# Deterministic pilot selection
# ---------------------------------------------------------------------------


def test_select_pilot_sources_deterministic() -> None:
    ids = [f"{i:05d}" for i in range(1000)]
    a = select_pilot_sources(ids, n=150, seed=42)
    b = select_pilot_sources(ids, n=150, seed=42)
    assert a == b
    assert len(a) == 150
    assert len(set(a)) == 150


def test_select_pilot_sources_not_first_n_alphabetically() -> None:
    ids = [f"{i:05d}" for i in range(1000)]
    selected = select_pilot_sources(ids, n=150, seed=42)
    first_150 = sorted(ids)[:150]
    assert selected != first_150


def test_select_pilot_sources_different_seed_changes_selection() -> None:
    ids = [f"{i:05d}" for i in range(1000)]
    a = select_pilot_sources(ids, n=150, seed=42)
    b = select_pilot_sources(ids, n=150, seed=7)
    assert a != b


def test_select_pilot_sources_order_independent() -> None:
    ids = [f"{i:05d}" for i in range(500)]
    shuffled = list(reversed(ids))
    assert select_pilot_sources(ids, n=100, seed=42) == select_pilot_sources(shuffled, n=100, seed=42)


def test_select_pilot_sources_n_exceeds_available_returns_all() -> None:
    ids = [f"{i:05d}" for i in range(10)]
    assert select_pilot_sources(ids, n=100, seed=42) == sorted(ids)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


def test_manifest_row_rejects_invalid_variant() -> None:
    with pytest.raises(ValueError):
        ManifestRow(
            sample_id="x", source_id="1", split="train", variant="not_a_real_variant",
            filepath="/x", format=None, width=None, height=None, channels=None,
            jpeg_quality_factor=None, embedding=None, dataset="ALASKA2",
        )


def test_manifest_row_accepts_all_documented_variants() -> None:
    for variant in ("cover", "lsb", "jmipod", "juniward", "uerd"):
        row = ManifestRow(
            sample_id="x", source_id="1", split="train", variant=variant,
            filepath="/x", format=None, width=None, height=None, channels=None,
            jpeg_quality_factor=None, embedding=None, dataset="ALASKA2",
        )
        assert row.variant == variant


def test_manifest_row_to_dict_matches_declared_columns() -> None:
    row = ManifestRow(
        sample_id="x", source_id="1", split="train", variant="cover",
        filepath="/x", format="JPEG", width=512, height=512, channels=3,
        jpeg_quality_factor=None, embedding=None, dataset="ALASKA2",
    )
    assert set(row.to_dict().keys()) == set(MANIFEST_COLUMNS)


def test_manifest_columns_never_collide_with_feature_names() -> None:
    """Structural guarantee that metadata columns can never be silently fed
    into the ML feature vector: no manifest column name matches a feature
    name, so a naive `df[some_columns]` selection error would be obvious
    rather than silently including e.g. a 'width' column as if it were a
    feature.
    """
    assert set(MANIFEST_COLUMNS).isdisjoint(set(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# End-to-end: pilot selection + split reuse + no cross-split/variant leakage
# ---------------------------------------------------------------------------


def test_pilot_to_split_pipeline_has_no_cross_split_leakage(tmp_path: Path) -> None:
    """Mirrors the real manifest shape: cover + 2 stego variants, some
    variants missing for some sources (as will genuinely happen with a real
    external dataset), split assigned once at source level, verified clean.
    """
    cover_dir = tmp_path / "Cover"
    variant_a_dir = tmp_path / "JMiPOD"
    variant_b_dir = tmp_path / "UERD"
    cover_dir.mkdir()
    variant_a_dir.mkdir()
    variant_b_dir.mkdir()

    source_ids = [f"{i:05d}" for i in range(60)]
    for sid in source_ids:
        _write_jpeg(cover_dir / f"{sid}.jpg", seed=int(sid))
        _write_jpeg(variant_a_dir / f"{sid}.jpg", seed=int(sid) + 1000)
        if int(sid) % 3 != 0:  # deliberately missing for 1/3 of sources
            _write_jpeg(variant_b_dir / f"{sid}.jpg", seed=int(sid) + 2000)

    selected = select_pilot_sources(source_ids, n=30, seed=42)
    split_map = assign_splits(selected, seed=42)

    rows = []
    for sid in selected:
        split = split_map[sid]
        rows.append({"source_id": sid, "split": split, "variant": "cover"})
        rows.append({"source_id": sid, "split": split, "variant": "jmipod"})
        if (variant_b_dir / f"{sid}.jpg").exists():
            rows.append({"source_id": sid, "split": split, "variant": "uerd"})

    problems = verify_split_integrity(rows)
    assert problems == []

    # Every source's rows must share exactly one split value.
    by_source: dict[str, set[str]] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], set()).add(r["split"])
    assert all(len(splits) == 1 for splits in by_source.values())
