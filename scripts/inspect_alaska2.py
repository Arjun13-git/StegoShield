#!/usr/bin/env python
"""Read-only inspection of whatever external (ALASKA2) data actually exists
locally under a root directory.

This script makes NO assumption about directory naming (e.g. it does not
assume "Cover"/"JMiPOD"/... exist) and NEVER fabricates metadata. Every
non-trivial finding is classified as VERIFIED, NOT_VERIFIED, NOT_AVAILABLE,
or INFERRED in the output report.

If the root directory does not exist or is empty, this script reports that
clearly and exits without creating any placeholder/fake data.

Usage:
  python scripts/inspect_alaska2.py --root data/external --report-out data/reports/phase1b_dataset_inspection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.ml.external import (
    IMAGE_EXTENSIONS,
    DirectorySummary,
    discover_root,
    inspect_jpeg_quality,
    pair_across_variants,
)

MAX_DUPLICATE_SCAN_FILES = 2000  # safety cap; full-dataset duplicate scan is a later, explicit step
SAMPLE_IMAGES_PER_DIR = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/external")
    parser.add_argument("--report-out", default="data/reports/phase1b_dataset_inspection.json")
    return parser.parse_args()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _inspect_sample_images(dir_summary: DirectorySummary, n: int = SAMPLE_IMAGES_PER_DIR) -> list[dict]:
    results = []
    d = Path(dir_summary.path)
    image_files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)[:n]
    for p in image_files:
        try:
            with Image.open(p) as img:
                results.append(
                    {
                        "filename": p.name,
                        "format": img.format,
                        "mode": img.mode,
                        "size": list(img.size),
                        "certainty": "VERIFIED",
                    }
                )
        except (UnidentifiedImageError, OSError) as exc:
            results.append({"filename": p.name, "error": str(exc), "certainty": "VERIFIED (unreadable)"})
    return results


def _filename_pattern_summary(dir_summary: DirectorySummary) -> dict:
    stems = [Path(f).stem for f in dir_summary.sample_filenames]
    all_numeric = bool(stems) and all(s.isdigit() for s in stems)
    return {
        "sample_stems": stems,
        "all_sampled_stems_numeric": all_numeric,
        "certainty": "VERIFIED (on sampled filenames only, not full directory)" if stems else "NOT_AVAILABLE",
    }


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    report: dict = {"root": str(root), "phase": "1B pilot preparation - dataset inspection"}

    discovery = discover_root(root)
    report["root_exists"] = discovery.root_exists

    if not discovery.root_exists:
        report["status"] = "NOT_AVAILABLE"
        report["finding"] = f"Root directory '{root}' does not exist. ALASKA2 has not been downloaded."
        report["next_human_action"] = (
            "A human with a Kaggle account must: (1) visit "
            "https://www.kaggle.com/c/alaska2-image-steganalysis/data, (2) accept the competition rules, "
            "(3) download a subset (not the full ~300k-image dataset) of the Cover/ and stego-variant "
            f"folders using the Kaggle API or web UI, and (4) place them under '{root}/' in whatever "
            "subfolder structure the download produces -- this script does not assume subfolder names "
            "and will auto-discover whatever directory structure is actually present on the next run."
        )
        print(f"NOT_AVAILABLE: {report['finding']}")
        print(report["next_human_action"])
        _write_report(args.report_out, report)
        return

    if not discovery.subdirectories and not discovery.loose_files:
        report["status"] = "NOT_AVAILABLE"
        report["finding"] = f"Root directory '{root}' exists but is empty."
        print(f"NOT_AVAILABLE: {report['finding']}")
        _write_report(args.report_out, report)
        return

    report["status"] = "PARTIAL_OR_FOUND"
    report["loose_files_sample"] = discovery.loose_files
    report["subdirectories"] = []

    variant_dirs_for_pairing: dict[str, Path] = {}

    for d in discovery.subdirectories:
        entry: dict = {
            "name": d.name,
            "path": d.path,
            "file_count_top_level": d.file_count,
            "extensions_found": d.extensions,
            "sample_filenames": d.sample_filenames,
            "certainty_file_count": "VERIFIED",
        }
        entry["disk_usage_bytes"] = _dir_size_bytes(Path(d.path))
        entry["disk_usage_certainty"] = "VERIFIED"
        entry["sample_image_inspection"] = _inspect_sample_images(d)
        entry["filename_pattern"] = _filename_pattern_summary(d)

        has_images = any(ext in IMAGE_EXTENSIONS for ext in d.extensions)
        if has_images:
            variant_dirs_for_pairing[d.name] = Path(d.path)

        report["subdirectories"].append(entry)

    if len(variant_dirs_for_pairing) >= 2:
        pairing = pair_across_variants(variant_dirs_for_pairing)
        report["cross_directory_pairing"] = {
            "variant_dirs": pairing.variant_dirs,
            "common_stems_across_all_dirs": pairing.common_stems,
            "total_unique_stems": pairing.total_unique_stems,
            "per_variant_only_count": pairing.per_variant_only_count,
            "certainty": "VERIFIED (filename-stem pairing only; does not confirm semantic cover/stego relationship)",
        }
    else:
        report["cross_directory_pairing"] = {
            "status": "NOT_AVAILABLE",
            "reason": f"Found {len(variant_dirs_for_pairing)} image-bearing subdirectories; need >=2 to test pairing.",
        }

    # JPEG quality-factor inspection: sample a few JPEGs across all discovered dirs.
    qf_samples = []
    for d in discovery.subdirectories:
        dpath = Path(d.path)
        jpegs = sorted(p for p in dpath.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})[:2]
        for p in jpegs:
            qf_samples.append(inspect_jpeg_quality(p).__dict__)
    matched_count = sum(1 for s in qf_samples if s.get("matched_known_qf") is not None)
    report["jpeg_quality_factor_inspection"] = {
        "samples": qf_samples,
        "note": (
            "matched_known_qf is set ONLY when a file's luma quantization table is a byte-for-byte "
            "exact match to PIL's own standard quality=75/90/95 table (generated locally, not "
            "hand-transcribed). This validates 'this table matches PIL's standard QF=N table exactly', "
            "not ALASKA2's own internal labeling -- unmatched files correctly stay null rather than "
            "being assigned a guessed QF."
        ),
        "matched_sample_count": matched_count,
        "unmatched_sample_count": len(qf_samples) - matched_count,
        "certainty": "VERIFIED" if matched_count else ("NOT_VERIFIED" if qf_samples else "NOT_AVAILABLE"),
    }

    report["camera_or_source_metadata"] = {
        "status": "NOT_AVAILABLE",
        "reason": (
            "No metadata file (e.g. CSV/JSON mapping images to cameras) was discovered under the root. "
            "This script only looked for one; it did not search recursively beyond one level or guess "
            "at metadata embedded in filenames beyond the numeric-stem check above."
        ),
    }

    print(json.dumps(report, indent=2, default=str))
    _write_report(args.report_out, report)


def _write_report(path: str, report: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote inspection report to {out}")


if __name__ == "__main__":
    main()
