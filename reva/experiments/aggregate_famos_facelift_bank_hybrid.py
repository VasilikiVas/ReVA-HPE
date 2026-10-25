#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reva.experiments.aggregate_famos_flame_bank_hybrid import (
    add_errors,
    key,
    load_bank_rows,
    matrix_from_row,
    summary_row,
    vector_from_row,
    write_csv,
    write_markdown,
)


OUT_ROOT = Path("outputs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_view00_mtcnn_visible_no091")
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "facelift_bank_hybrid",
    )
    parser.add_argument(
        "--refinement-table-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "refinement_summary_tables",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "facelift_bank_hybrid_tables",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    bank_dir = args.bank_root / args.protocol
    table_dir = args.refinement_table_root / args.protocol
    existing_summary = read_csv(table_dir / "famos_refinement_full_summary.csv")
    existing_predictions = read_csv(table_dir / "famos_refinement_full_predictions.csv")
    bank_rows = load_bank_rows(bank_dir)
    if not bank_rows:
        raise RuntimeError(f"No FaceLift-bank prediction rows under {bank_dir}")
    bank_layouts = {row.get("bank_layout", "") for row in bank_rows}
    uses_gt_transport = any(
        str(row.get("bank_uses_dataset_pose_or_depth_ground_truth", "0")) == "1"
        or str(row.get("bank_uses_famos_enrollment_gt_rotation", "0")) == "1"
        or row.get("bank_layout") == "manual_view00"
        for row in bank_rows
    )
    if uses_gt_transport:
        direct_method = "FaceLift GT-transport bank RANSAC (diagnostic)"
        direct_category = "Bank-anchor diagnostic"
        rotation_source = "FaMoS manual-view00 FaceLift bank with manifest-label/GT transport"
        hybrid_prefix = "FaceLift GT-transport bank"
        hybrid_category = "Bank-anchor diagnostic"
        hybrid_suffix = " (mixed-source)"
        direct_note = "direct manual-view00 FaceLift bank RANSAC after manifest-label/GT transport"
        metadata_note = (
            "These rows use manual-view00 FaceLift/Gaussian avatar banks. "
        )
    else:
        direct_method = "FaceLift metric bank RANSAC (BIWI-style)"
        direct_category = "Bank-anchor practical"
        rotation_source = "FaMoS FaceLift metric anchor bank RANSAC using BIWI-style anchors.json poses"
        hybrid_prefix = "FaceLift metric bank"
        hybrid_category = "Bank-anchor practical"
        hybrid_suffix = ""
        direct_note = "direct FaceLift metric bank RANSAC using rotation_h2c_opencv and translation_h2c_m from anchors.json"
        metadata_note = (
            "These rows use BIWI-style FaceLift metric anchor banks: each anchor pose is loaded "
        )

    by_method = defaultdict(dict)
    for row in existing_predictions:
        by_method[row["method"]][key(row)] = row
    bank_by_key = {key(row): row for row in bank_rows}

    base_methods = {
        f"{hybrid_prefix} R + Enrollment-PnP t{hybrid_suffix}": (
            "Enrollment-PnP + relative",
            hybrid_category,
            f"rotation from {rotation_source}, translation copied from Enrollment-PnP + relative",
        ),
        f"{hybrid_prefix} R + AvatarBox-median t{hybrid_suffix}": (
            "AvatarBox-median",
            hybrid_category,
            f"rotation from {rotation_source}, translation copied from AvatarBox-median",
        ),
        f"{hybrid_prefix} R + AvatarBox-trimmed-mean t{hybrid_suffix}": (
            "AvatarBox-trimmed-mean",
            hybrid_category,
            f"rotation from {rotation_source}, translation copied from AvatarBox-trimmed-mean",
        ),
    }

    generated = {}
    direct_bank = []
    for row in bank_rows:
        if str(row.get("direct_bank_translation_valid", "0")) != "1":
            continue
        out = add_errors(row, matrix_from_row(row, "pred"), vector_from_row(row, "pred"))
        out["method"] = direct_method
        out["category"] = direct_category
        out["bank_rotation_source"] = rotation_source
        out["translation_source_method"] = "FaceLift metric bank"
        direct_bank.append(out)
    if direct_bank:
        generated[direct_method] = direct_bank

    for method, (base_method, category, note) in base_methods.items():
        rows = []
        base_by_key = by_method[base_method]
        matched_keys = sorted(set(base_by_key) & set(bank_by_key))
        for item in matched_keys:
            base_row = base_by_key[item]
            bank_row = bank_by_key[item]
            rotation = matrix_from_row(bank_row, "pred")
            translation = vector_from_row(base_row, "pred")
            out = add_errors(base_row, rotation, translation)
            out["method"] = method
            out["category"] = category
            out["bank_rotation_source"] = rotation_source
            out["translation_source_method"] = base_method
            out["direct_bank_translation_valid"] = bank_row.get("direct_bank_translation_valid", "1")
            rows.append(out)
        generated[method] = rows

    bank_summary = []
    if direct_bank:
        bank_summary.append(
            summary_row(
                generated[direct_method],
                direct_method,
                direct_category,
                direct_note,
            )
        )
    for method, (_, category, note) in base_methods.items():
        bank_summary.append(summary_row(generated[method], method, category, note))

    out_dir = args.out_root / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_predictions = [row for rows in generated.values() for row in rows]
    combined_summary = existing_summary + bank_summary

    summary_csv = out_dir / "famos_refinement_with_facelift_bank_summary.csv"
    summary_md = out_dir / "famos_refinement_with_facelift_bank_summary.md"
    predictions_csv = out_dir / "famos_refinement_with_facelift_bank_predictions.csv"
    metadata_json = out_dir / "famos_refinement_with_facelift_bank_metadata.json"

    summary_fields = ["category", "method", "frames", "yaw", "pitch", "roll", "maer", "ge", "tx", "ty", "tz", "maet", "add", "note"]
    write_csv(summary_csv, combined_summary, summary_fields)
    write_csv(predictions_csv, combined_predictions)
    write_markdown(summary_md, f"FaMoS {args.protocol} Refinement Summary With FaceLift Label-Reanchored Bank", combined_summary)
    metadata = {
        "protocol": args.protocol,
        "bank_dir": str(bank_dir),
        "bank_layouts": sorted(bank_layouts),
        "uses_gt_transport": uses_gt_transport,
        "existing_refinement_summary": str(table_dir / "famos_refinement_full_summary.csv"),
        "existing_refinement_predictions": str(table_dir / "famos_refinement_full_predictions.csv"),
        "bank_rows": len(bank_rows),
        "generated_methods": {method: len(rows) for method, rows in generated.items()},
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "predictions_csv": str(predictions_csv),
        "note": metadata_note,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    combined_bank_path = bank_dir / "combined_predictions.csv"
    if not combined_bank_path.is_file():
        write_csv(combined_bank_path, bank_rows)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
