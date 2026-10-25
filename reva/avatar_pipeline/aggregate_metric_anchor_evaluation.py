#!/usr/bin/env python3
"""Aggregate per-subject metric-anchor BIWI evaluation CSV files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "yaw_error_deg",
    "pitch_error_deg",
    "roll_error_deg",
    "rot_mean_error_deg",
    "tx_error_mm",
    "ty_error_mm",
    "tz_error_mm",
    "translation_l2_error_mm",
    "ransac_inliers",
    "ransac_success",
    "inference_runtime_s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=range(1, 25))
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    result = {"queries": len(rows)}
    for metric in METRICS:
        result[metric] = float(np.mean([float(row[metric]) for row in rows]))
    return result


def main():
    args = parse_args()
    combined = []
    per_subject = []
    missing = []
    for subject_id in args.subjects:
        subject = f"{subject_id:02d}"
        path = args.input_dir / f"subject_{subject}_predictions.csv"
        if not path.is_file():
            missing.append(subject)
            continue
        rows = read_rows(path)
        combined.extend(rows)
        summary = {"subject": subject, **summarize(rows)}
        per_subject.append(summary)

    if not combined:
        raise RuntimeError("No subject prediction CSV files were found")
    combined_path = args.input_dir / "combined_predictions.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    per_subject_path = args.input_dir / "per_subject_summary.csv"
    with per_subject_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_subject[0]))
        writer.writeheader()
        writer.writerows(per_subject)
    overall = {
        "subjects": len(per_subject),
        "missing_subjects": " ".join(missing),
        **summarize(combined),
    }
    overall_path = args.input_dir / "overall_summary.csv"
    with overall_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall))
        writer.writeheader()
        writer.writerow(overall)
    metadata_path = args.input_dir / "aggregation.json"
    metadata_path.write_text(
        json.dumps(
            {
                "requested_subjects": len(args.subjects),
                "completed_subjects": len(per_subject),
                "missing_subjects": missing,
                "queries": len(combined),
                "overall": overall,
            },
            indent=2,
        )
    )
    print(overall_path)
    print(per_subject_path)
    print(combined_path)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
