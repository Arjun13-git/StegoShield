from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.ml.dataset import find_duplicate_source_ids, index_bossbase


def _write_pgm(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    arr = np.random.default_rng(0).integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_index_finds_all_valid_images(tmp_path: Path) -> None:
    for i in range(5):
        _write_pgm(tmp_path / f"{i}.pgm")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert result.valid_count == 5
    assert result.error_count == 0
    assert {r.source_id for r in result.records} == {"0", "1", "2", "3", "4"}


def test_index_is_ordered_by_numeric_stem_not_filesystem_order(tmp_path: Path) -> None:
    for i in [10, 2, 1, 30]:
        _write_pgm(tmp_path / f"{i}.pgm")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert [r.source_id for r in result.records] == ["1", "2", "10", "30"]


def test_index_rejects_wrong_dimensions(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "0.pgm", size=(16, 16))
    _write_pgm(tmp_path / "1.pgm", size=(8, 8))
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert result.valid_count == 1
    assert result.error_count == 1
    assert "unexpected dimensions" in result.errors[0]


def test_index_rejects_non_grayscale(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "0.pgm", size=(16, 16))
    rgb = np.random.default_rng(1).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "1.pgm")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert result.valid_count == 1
    assert result.error_count == 1
    assert "grayscale" in result.errors[0]


def test_index_reports_corrupt_files(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "0.pgm", size=(16, 16))
    (tmp_path / "1.pgm").write_bytes(b"not a real image")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert result.valid_count == 1
    assert result.error_count == 1
    assert "unreadable/corrupt" in result.errors[0]


def test_index_raises_when_directory_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        index_bossbase(tmp_path / "does_not_exist")


def test_index_raises_when_no_pgm_files(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("nothing here")
    with pytest.raises(ValueError):
        index_bossbase(tmp_path)


def test_index_raises_when_all_files_invalid(tmp_path: Path) -> None:
    (tmp_path / "0.pgm").write_bytes(b"garbage")
    with pytest.raises(ValueError):
        index_bossbase(tmp_path)


def test_index_warns_when_below_expected_count(tmp_path: Path) -> None:
    for i in range(3):
        _write_pgm(tmp_path / f"{i}.pgm")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16), expected_count=10)
    assert result.valid_count == 3
    assert any("Expected at least 10" in e for e in result.errors)


def test_index_records_content_hash(tmp_path: Path) -> None:
    _write_pgm(tmp_path / "0.pgm")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert result.records[0].content_hash
    assert len(result.records[0].content_hash) == 32  # md5 hex digest


def test_find_duplicate_source_ids_detects_identical_pixel_content(tmp_path: Path) -> None:
    identical = np.random.default_rng(0).integers(0, 256, size=(16, 16), dtype=np.uint8)
    distinct = np.random.default_rng(99).integers(0, 256, size=(16, 16), dtype=np.uint8)

    Image.fromarray(identical).save(tmp_path / "1.pgm")
    Image.fromarray(identical).save(tmp_path / "2.pgm")  # byte-identical to 1.pgm, different filename
    Image.fromarray(distinct).save(tmp_path / "3.pgm")

    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    duplicates = find_duplicate_source_ids(result.records)
    assert duplicates == {"2": "1"}


def test_find_duplicate_source_ids_empty_when_all_unique(tmp_path: Path) -> None:
    for i in range(3):
        arr = np.random.default_rng(i).integers(0, 256, size=(16, 16), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"{i}.pgm")
    result = index_bossbase(tmp_path, expected_dimensions=(16, 16))
    assert find_duplicate_source_ids(result.records) == {}
