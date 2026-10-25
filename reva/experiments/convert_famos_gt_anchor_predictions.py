#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PROTOCOL = "famos_view00_mtcnn_visible_no091"
OUT_ROOT = Path("outputs")
DEFAULT_READY = OUT_ROOT / "refinement_inputs_cross_dataset" / DEFAULT_PROTOCOL / "ready_pose_predictions.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--ready-csv", type=Path, default=DEFAULT_READY)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--variant", default="E_h2c_correct")
    parser.add_argument("--method", default="VGGT-HPE-Rel published (GT anchor)")
    parser.add_argument("--category", default="Original VGGT-HPE")
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ready_key(row):
    source = row["source_key"].split("|")
    if len(source) != 3:
        raise ValueError(f"Unexpected source_key: {row['source_key']}")
    subject, sequence, view = source
    return subject, sequence, str(int(view)), str(row["query_original_frame"])


def raw_key(row):
    return row["subject"], row["sequence"], str(int(row["view"])), str(row["frame"])


def metric_mean(rows, key):
    vals = [float(row[key]) for row in rows if row.get(key, "") not in {"", None}]
    return float(np.mean(vals)) if vals else float("nan")


def convert_row(raw, ready, method, category):
    out = {
        "protocol": ready.get("protocol", DEFAULT_PROTOCOL),
        "dataset": "famos",
        "category": category,
        "method": method,
        "source_key": ready["source_key"],
        "source_dataset": ready.get("source_dataset", "famos"),
        "query_image_path": ready["query_image_path"],
        "query_original_frame": ready["query_original_frame"],
        "enrollment_id": ready["enrollment_id"],
        "enrollment_image_path": ready["enrollment_image_path"],
        "enrollment_rgb_cal_path": ready["enrollment_rgb_cal_path"],
        "canonical_flame_json": ready["canonical_flame_json"],
        "subject": ready["subject"],
        "anchor_frame": ready["anchor_frame"],
        "frame": ready["frame"],
        "raw_subject": raw["subject"],
        "raw_sequence": raw["sequence"],
        "raw_view": raw["view"],
        "raw_anchor_frame": raw["anchor_frame"],
        "raw_frame": raw["frame"],
        "variant": raw["variant"],
        "yaw_error_deg": raw["yaw_error_deg"],
        "pitch_error_deg": raw["pitch_error_deg"],
        "roll_error_deg": raw["roll_error_deg"],
        "rot_mean_error_deg": raw["rot_mean_error_deg"],
        "geodesic_error_deg": raw.get("geodesic_error_deg", ""),
        "pred_tx_mm": raw["pred_tx_mm"],
        "pred_ty_mm": raw["pred_ty_mm"],
        "pred_tz_mm": raw["pred_tz_mm"],
        "gt_tx_mm": raw["gt_tx_mm"],
        "gt_ty_mm": raw["gt_ty_mm"],
        "gt_tz_mm": raw["gt_tz_mm"],
        "tx_error_mm": raw["tx_error_mm"],
        "ty_error_mm": raw["ty_error_mm"],
        "tz_error_mm": raw["tz_error_mm"],
        "translation_l2_error_mm": raw["translation_l2_error_mm"],
    }
    for rr in range(3):
        for cc in range(3):
            out[f"pred_r{rr}{cc}"] = raw[f"pred_r{rr}{cc}"]
            out[f"gt_r{rr}{cc}"] = raw[f"gt_r{rr}{cc}"]
    return out


def main():
    args = parse_args()
    ready_rows = read_csv(args.ready_csv)
    raw_rows = read_csv(args.raw_predictions)
    ready_map = {ready_key(row): row for row in ready_rows}

    converted = []
    raw_primary = 0
    unmatched = []
    skipped_no_translation = 0
    for raw in raw_rows:
        if raw.get("variant") != args.variant:
            continue
        raw_primary += 1
        if raw.get("pred_tx_mm", "") in {"", None}:
            skipped_no_translation += 1
            continue
        key = raw_key(raw)
        ready = ready_map.get(key)
        if ready is None:
            unmatched.append("|".join(key))
            continue
        converted.append(convert_row(raw, ready, args.method, args.category))

    summary = [
        {
            "category": args.category,
            "method": args.method,
            "frames": str(len(converted)),
            "yaw": f"{metric_mean(converted, 'yaw_error_deg'):.9f}",
            "pitch": f"{metric_mean(converted, 'pitch_error_deg'):.9f}",
            "roll": f"{metric_mean(converted, 'roll_error_deg'):.9f}",
            "maer": f"{metric_mean(converted, 'rot_mean_error_deg'):.9f}",
            "ge": f"{metric_mean(converted, 'geodesic_error_deg'):.9f}",
            "tx": f"{metric_mean(converted, 'tx_error_mm'):.9f}",
            "ty": f"{metric_mean(converted, 'ty_error_mm'):.9f}",
            "tz": f"{metric_mean(converted, 'tz_error_mm'):.9f}",
            "maet": f"{metric_mean(converted, 'translation_l2_error_mm'):.9f}",
        }
    ]

    write_csv(args.predictions_out, converted)
    write_csv(args.summary_out, summary)
    metadata = {
        "raw_predictions": str(args.raw_predictions),
        "ready_csv": str(args.ready_csv),
        "variant": args.variant,
        "method": args.method,
        "category": args.category,
        "ready_rows": len(ready_rows),
        "raw_rows": len(raw_rows),
        "raw_primary_rows": raw_primary,
        "converted_rows": len(converted),
        "skipped_no_translation": skipped_no_translation,
        "unmatched_rows": len(unmatched),
        "unmatched_examples": unmatched[:20],
        "predictions_out": str(args.predictions_out),
        "summary_out": str(args.summary_out),
    }
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
