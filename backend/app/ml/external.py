"""External-dataset (Phase 1B) discovery, pairing, and manifest utilities.

This module is deliberately generic: it does not assume ALASKA2's specific
directory names (e.g. "Cover", "JMiPOD"). It discovers whatever structure
actually exists on disk and reports what it finds, classifying every
non-trivial fact as VERIFIED, NOT_VERIFIED, NOT_AVAILABLE, or INFERRED
rather than guessing.

This module is entirely independent of the frozen Phase 1A model, feature
extractor, and BOSSBase-specific `dataset.py`/`lsb.py` modules -- none of
those are imported or modified here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

Certainty = Literal["VERIFIED", "NOT_VERIFIED", "NOT_AVAILABLE", "INFERRED"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectorySummary:
    name: str
    path: str
    file_count: int
    extensions: dict[str, int]
    sample_filenames: list[str]


@dataclass(frozen=True)
class RootDiscovery:
    root: str
    root_exists: bool
    subdirectories: list[DirectorySummary]
    loose_files: list[str]  # files directly under root, not in a subdirectory


def discover_root(root: str | Path, *, sample_size: int = 5) -> RootDiscovery:
    """One-level directory discovery under `root`. Makes no assumption about
    subdirectory naming -- reports whatever is actually there.
    """
    root = Path(root)
    if not root.exists():
        return RootDiscovery(root=str(root), root_exists=False, subdirectories=[], loose_files=[])

    subdirs: list[DirectorySummary] = []
    loose_files: list[str] = []

    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            files = [p for p in entry.iterdir() if p.is_file()]
            ext_counts = Counter(p.suffix.lower() for p in files)
            sample = sorted(p.name for p in files)[:sample_size]
            subdirs.append(
                DirectorySummary(
                    name=entry.name,
                    path=str(entry),
                    file_count=len(files),
                    extensions=dict(ext_counts),
                    sample_filenames=sample,
                )
            )
        elif entry.is_file():
            loose_files.append(entry.name)

    return RootDiscovery(root=str(root), root_exists=True, subdirectories=subdirs, loose_files=sorted(loose_files)[:sample_size])


# ---------------------------------------------------------------------------
# Cross-directory filename pairing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairingResult:
    variant_dirs: dict[str, str]  # variant label -> directory path
    common_stems: int  # filenames (by stem) present in every variant dir
    per_variant_only_count: dict[str, int]  # stems present in only that variant
    total_unique_stems: int


def pair_across_variants(variant_dirs: dict[str, Path]) -> PairingResult:
    """Given {variant_label: directory}, find filenames (by stem) common to
    every directory. Does not assume any particular naming convention.
    """
    stems_by_variant: dict[str, set[str]] = {}
    for label, d in variant_dirs.items():
        stems_by_variant[label] = {p.stem for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}

    all_stems: set[str] = set()
    for stems in stems_by_variant.values():
        all_stems |= stems

    common = set.intersection(*stems_by_variant.values()) if stems_by_variant else set()

    only_counts: dict[str, int] = {}
    for label, stems in stems_by_variant.items():
        others = set()
        for other_label, other_stems in stems_by_variant.items():
            if other_label != label:
                others |= other_stems
        only_counts[label] = len(stems - others)

    return PairingResult(
        variant_dirs={k: str(v) for k, v in variant_dirs.items()},
        common_stems=len(common),
        per_variant_only_count=only_counts,
        total_unique_stems=len(all_stems),
    )


# ---------------------------------------------------------------------------
# JPEG quality-factor inspection (honest: no fabricated QF mapping)
# ---------------------------------------------------------------------------

# ALASKA2's documented quality-factor levels (external source: competition
# data description). Used only as candidate values to test an *exact*
# quantization-table match against -- never assumed without verification.
_CANDIDATE_QFS: tuple[int, ...] = (75, 90, 95)


def _reference_luma_table(qf: int) -> tuple[int, ...]:
    """The standard IJG/libjpeg luma quantization table PIL produces for a
    given `quality`, derived empirically by round-tripping a throwaway image
    through PIL's own JPEG encoder -- not hand-transcribed from any external
    source. This is the same scaling convention the vast majority of JPEG
    encoders (including, very likely, whatever produced ALASKA2) use.
    """
    import io

    probe = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    buf = io.BytesIO()
    probe.save(buf, format="JPEG", quality=qf)
    buf.seek(0)
    with Image.open(buf) as img:
        return tuple(img.quantization[0])


_REFERENCE_TABLES: dict[int, tuple[int, ...]] = {qf: _reference_luma_table(qf) for qf in _CANDIDATE_QFS}


@dataclass(frozen=True)
class JpegQualityInfo:
    path: str
    is_jpeg: bool
    quantization_table_digest: str | None  # sha256 of the raw luma quant table, if present
    matched_known_qf: int | None  # only set if validated against a confirmed reference table
    certainty: Certainty


def inspect_jpeg_quality(path: str | Path) -> JpegQualityInfo:
    """Inspect a JPEG's raw quantization table.

    `matched_known_qf` is set ONLY when the file's actual luma quantization
    table is a byte-for-byte exact match to one of PIL's own standard
    quality=75/90/95 tables (`_REFERENCE_TABLES`, generated locally, not
    hand-transcribed or assumed). An exact match is strong, verifiable
    evidence -- not a heuristic guess -- but is still reported as VERIFIED
    only for that specific claim ("this table matches PIL's standard QF=90
    table exactly"), not as proof of ALASKA2's own internal labeling.
    """
    path = Path(path)
    try:
        with Image.open(path) as img:
            is_jpeg = img.format == "JPEG"
            quant = getattr(img, "quantization", None)
    except (UnidentifiedImageError, OSError):
        return JpegQualityInfo(path=str(path), is_jpeg=False, quantization_table_digest=None, matched_known_qf=None, certainty="NOT_AVAILABLE")

    if not is_jpeg or not quant:
        return JpegQualityInfo(path=str(path), is_jpeg=is_jpeg, quantization_table_digest=None, matched_known_qf=None, certainty="NOT_AVAILABLE")

    # quant is a dict {component_id: [64 ints]}; hash the luma (component 0) table.
    luma_table = quant.get(0)
    if luma_table is None:
        return JpegQualityInfo(path=str(path), is_jpeg=True, quantization_table_digest=None, matched_known_qf=None, certainty="NOT_AVAILABLE")

    digest = hashlib.sha256(bytes(luma_table)).hexdigest()

    matched_qf: int | None = None
    for qf, ref_table in _REFERENCE_TABLES.items():
        if tuple(luma_table) == ref_table:
            matched_qf = qf
            break

    return JpegQualityInfo(
        path=str(path),
        is_jpeg=True,
        quantization_table_digest=digest,
        matched_known_qf=matched_qf,
        certainty="VERIFIED" if matched_qf is not None else "NOT_VERIFIED",
    )


# ---------------------------------------------------------------------------
# Duplicate detection (decoded-pixel content hash, format-agnostic)
# ---------------------------------------------------------------------------


def content_hash(path: str | Path) -> str | None:
    """MD5 of decoded pixel bytes (not raw file bytes), so differently-encoded
    but pixel-identical images are still detected as duplicates. Returns None
    if the file cannot be decoded.
    """
    try:
        with Image.open(path) as img:
            img.load()
            arr = np.asarray(img)
    except (UnidentifiedImageError, OSError):
        return None
    return hashlib.md5(arr.tobytes()).hexdigest()


def find_duplicate_files(paths: list[Path]) -> dict[str, list[str]]:
    """Group file paths by decoded-pixel content hash. Returns only groups
    with more than one member (i.e. actual duplicates), keyed by hash.
    """
    groups: dict[str, list[str]] = {}
    for p in paths:
        h = content_hash(p)
        if h is None:
            continue
        groups.setdefault(h, []).append(str(p))
    return {h: members for h, members in groups.items() if len(members) > 1}


# ---------------------------------------------------------------------------
# Deterministic pilot selection (same discipline as Phase 1A)
# ---------------------------------------------------------------------------


def select_pilot_sources(source_ids: list[str], n: int, seed: int) -> list[str]:
    """Deterministic seeded selection: sort unique ids, seeded-permute, slice.
    Mirrors `scripts/generate_dataset.py`'s `_select_sources` exactly, kept
    as a separate implementation here since Phase 1A code must not be
    imported into/coupled with Phase 1B in a way that risks modifying it.
    """
    unique_ids = sorted(set(source_ids))
    if n >= len(unique_ids):
        return unique_ids
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_ids))
    return sorted(unique_ids[i] for i in order[:n])


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_id",
    "split",
    "variant",
    "filepath",
    "format",
    "width",
    "height",
    "channels",
    "jpeg_quality_factor",
    "embedding",
    "dataset",
)

VALID_VARIANTS = frozenset({"cover", "lsb", "jmipod", "juniward", "uerd"})


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    source_id: str
    split: str
    variant: str
    filepath: str
    format: str | None
    width: int | None
    height: int | None
    channels: int | None
    jpeg_quality_factor: int | None
    embedding: str | None
    dataset: str

    def __post_init__(self) -> None:
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VALID_VARIANTS)}, got {self.variant!r}")

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "split": self.split,
            "variant": self.variant,
            "filepath": self.filepath,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "jpeg_quality_factor": self.jpeg_quality_factor,
            "embedding": self.embedding,
            "dataset": self.dataset,
        }
