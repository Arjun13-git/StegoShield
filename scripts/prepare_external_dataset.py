#!/usr/bin/env python
"""Deterministic pilot selection + source-aware split + manifest generation
for a Phase 1B external dataset (e.g. ALASKA2).

This script does NOT guess which discovered directory is "cover" vs a
stego variant -- the caller must say so explicitly via --cover-dir and
--variant-dirs, once the real downloaded structure has been inspected with
scripts/inspect_alaska2.py. If those are not provided, or the given paths
don't exist, this script stops and reports rather than fabricating a
manifest.

Reuses the existing dataset-agnostic split utilities from
backend/app/ml/split.py UNCHANGED -- Phase 1A code is not modified.

Usage (once ALASKA2 is actually downloaded and inspected):
  python scripts/prepare_external_dataset.py \\
      --cover-dir data/external/ALASKA2/Cover \\
      --variant-dirs jmipod=data/external/ALASKA2/JMiPOD,juniward=data/external/ALASKA2/JUNIWARD,uerd=data/external/ALASKA2/UERD \\
      --pilot-size 150 \\
      --seed 42 \\
      --manifest-out data/external/ALASKA2/pilot_manifest.csv \\
      --report-out data/reports/phase1b_pilot_preparation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from app.ml.external import (
    MANIFEST_COLUMNS,
    ManifestRow,
    find_duplicate_files,
    inspect_jpeg_quality,
    select_pilot_sources,
)
from app.ml.split import DEFAULT_RATIOS, assign_splits, verify_split_integrity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cover-dir", default=None, help="Directory containing cover images (required)")
    parser.add_argument(
        "--variant-dirs",
        default=None,
        help="Comma-separated label=path pairs for stego variants, e.g. jmipod=path,uerd=path (optional)",
    )
    parser.add_argument("--pilot-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratios", default="0.70,0.15,0.15", help="train,val,test")
    parser.add_argument("--manifest-out", default="data/external/ALASKA2/pilot_manifest.csv")
    parser.add_argument("--report-out", default="data/reports/phase1b_pilot_preparation.json")
    parser.add_argument("--dataset-label", default="ALASKA2")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="keep only sources that have every --variant-dirs variant (never emit partial groups)",
    )
    return parser.parse_args()


def _image_meta(path: Path) -> dict:
    try:
        with Image.open(path) as img:
            img.load()
            meta = {"format": img.format, "width": img.size[0], "height": img.size[1], "channels": len(img.getbands())}
    except (UnidentifiedImageError, OSError):
        return {"format": None, "width": None, "height": None, "channels": None, "jpeg_quality_factor": None}

    # Only ever set from an exact quantization-table match (see
    # app.ml.external.inspect_jpeg_quality) -- never fabricated for a
    # non-matching table or a non-JPEG file.
    meta["jpeg_quality_factor"] = inspect_jpeg_quality(path).matched_known_qf
    return meta


def main() -> None:
    args = _parse_args()
    report: dict = {"status": None}

    if args.cover_dir is None:
        report["status"] = "STOPPED"
        report["reason"] = (
            "No --cover-dir provided. This script will not guess which directory contains cover "
            "images. Run scripts/inspect_alaska2.py first, identify the correct directory from its "
            "output, then re-run with --cover-dir explicitly set."
        )
        print(f"STOPPED: {report['reason']}")
        _write_report(args.report_out, report)
        return

    cover_dir = Path(args.cover_dir)
    if not cover_dir.exists() or not cover_dir.is_dir():
        report["status"] = "NOT_AVAILABLE"
        report["reason"] = f"--cover-dir '{cover_dir}' does not exist or is not a directory."
        print(f"NOT_AVAILABLE: {report['reason']}")
        _write_report(args.report_out, report)
        return

    variant_dirs: dict[str, Path] = {}
    if args.variant_dirs:
        for pair in args.variant_dirs.split(","):
            if "=" not in pair:
                report["status"] = "STOPPED"
                report["reason"] = f"Malformed --variant-dirs entry (expected label=path): {pair!r}"
                print(f"STOPPED: {report['reason']}")
                _write_report(args.report_out, report)
                return
            label, path_str = pair.split("=", 1)
            vpath = Path(path_str)
            if not vpath.exists():
                report["status"] = "NOT_AVAILABLE"
                report["reason"] = f"Variant directory for '{label}' does not exist: {vpath}"
                print(f"NOT_AVAILABLE: {report['reason']}")
                _write_report(args.report_out, report)
                return
            variant_dirs[label] = vpath

    cover_files = sorted(p for p in cover_dir.iterdir() if p.is_file())
    if not cover_files:
        report["status"] = "NOT_AVAILABLE"
        report["reason"] = f"--cover-dir '{cover_dir}' exists but contains no files."
        print(f"NOT_AVAILABLE: {report['reason']}")
        _write_report(args.report_out, report)
        return

    cover_source_ids = sorted({p.stem for p in cover_files})

    if args.require_complete and variant_dirs:
        def _has(vdir: Path, sid: str) -> bool:
            return any((vdir / f"{sid}{ext}").exists() for ext in (".jpg", ".jpeg", ".png"))

        complete = [sid for sid in cover_source_ids if all(_has(v, sid) for v in variant_dirs.values())]
        dropped = sorted(set(cover_source_ids) - set(complete))
        report["require_complete"] = {
            "cover_sources_found": len(cover_source_ids),
            "complete_sources_kept": len(complete),
            "incomplete_sources_dropped": dropped,
        }
        cover_source_ids = complete
        cover_files = [p for p in cover_files if p.stem in set(complete)]

    # Duplicate check BEFORE selecting the pilot, so a duplicate can never
    # silently end up representing two "independent" pilot sources. Covers
    # are checked among themselves; all variants together are also checked
    # so cross-variant / cross-source content duplicates are visible.
    duplicate_groups = find_duplicate_files(cover_files)
    report["duplicate_check"] = {
        "files_checked": len(cover_files),
        "duplicate_groups_found": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }
    selected_lookup = set(cover_source_ids)
    all_files = list(cover_files)
    for vdir in variant_dirs.values():
        all_files += [p for p in vdir.iterdir() if p.is_file() and p.stem in selected_lookup]
    all_groups = find_duplicate_files(all_files)
    report["duplicate_check_all_variants"] = {
        "files_checked": len(all_files),
        "duplicate_groups_found": len(all_groups),
        "duplicate_groups": all_groups,
    }

    selected_ids = select_pilot_sources(cover_source_ids, args.pilot_size, args.seed)
    report["pilot_selection"] = {
        "requested_size": args.pilot_size,
        "available_cover_sources": len(cover_source_ids),
        "selected_count": len(selected_ids),
        "seed": args.seed,
        "selection_method": "sorted unique ids -> seeded np.random.default_rng permutation -> slice (same discipline as Phase 1A)",
    }

    ratio_parts = [float(x) for x in args.split_ratios.split(",")]
    ratios = {"train": ratio_parts[0], "val": ratio_parts[1], "test": ratio_parts[2]} if len(ratio_parts) == 3 else DEFAULT_RATIOS
    split_map = assign_splits(selected_ids, seed=args.seed, ratios=ratios)
    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in split_map.values():
        split_counts[s] += 1
    report["split"] = {"ratios": ratios, "counts": split_counts}

    rows: list[dict] = []
    missing_variants: dict[str, list[str]] = {label: [] for label in variant_dirs}
    cover_paths_by_stem = {p.stem: p for p in cover_files}

    for source_id in selected_ids:
        split = split_map[source_id]
        cover_path = cover_paths_by_stem[source_id]
        meta = _image_meta(cover_path)
        rows.append(
            ManifestRow(
                sample_id=f"{args.dataset_label}_{source_id}_cover",
                source_id=source_id,
                split=split,
                variant="cover",
                filepath=str(cover_path),
                format=meta["format"],
                width=meta["width"],
                height=meta["height"],
                channels=meta["channels"],
                jpeg_quality_factor=meta["jpeg_quality_factor"],
                embedding=None,
                dataset=args.dataset_label,
            ).to_dict()
        )

        for label, vdir in variant_dirs.items():
            candidate = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = vdir / f"{source_id}{ext}"
                if p.exists():
                    candidate = p
                    break
            if candidate is None:
                missing_variants[label].append(source_id)
                continue
            vmeta = _image_meta(candidate)
            rows.append(
                ManifestRow(
                    sample_id=f"{args.dataset_label}_{source_id}_{label}",
                    source_id=source_id,
                    split=split,
                    variant=label,
                    filepath=str(candidate),
                    format=vmeta["format"],
                    width=vmeta["width"],
                    height=vmeta["height"],
                    channels=vmeta["channels"],
                    jpeg_quality_factor=vmeta["jpeg_quality_factor"],
                    embedding=label if label != "cover" else None,
                    dataset=args.dataset_label,
                ).to_dict()
            )

    report["missing_variants"] = {k: {"count": len(v), "example_ids": v[:10]} for k, v in missing_variants.items()}

    integrity_rows = [{"source_id": r["source_id"], "split": r["split"]} for r in rows]
    problems = verify_split_integrity(integrity_rows)
    report["split_integrity"] = {"problems_found": len(problems), "problems": problems[:20]}
    if problems:
        report["status"] = "STOPPED"
        report["reason"] = "Split integrity check failed -- see split_integrity.problems. Manifest NOT written."
        print(f"STOPPED: {report['reason']}")
        _write_report(args.report_out, report)
        return

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    df.to_csv(manifest_path, index=False)

    report["status"] = "OK"
    report["manifest_path"] = str(manifest_path)
    report["manifest_row_count"] = len(rows)
    report["variant_counts"] = df["variant"].value_counts().to_dict()

    print(json.dumps(report, indent=2, default=str))
    _write_report(args.report_out, report)
    print(f"\nWrote manifest to {manifest_path} ({len(rows)} rows)")


def _write_report(path: str, report: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote pilot-preparation report to {out}")


if __name__ == "__main__":
    main()
