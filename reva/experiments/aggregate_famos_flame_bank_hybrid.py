#!/usr/bin/env python3
"""Aggregate FaMoS FLAME-bank rows and build bank-hybrid refinement tables."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "eval_scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reva.experiments.eval_cross_dataset_avatarbox_median import (  # noqa: E402
    angular_metrics,
    summarize,
)


OUT_ROOT = Path("outputs")
R_KEYS = [f"pred_r{rr}{cc}" for rr in range(3) for cc in range(3)]
GT_R_KEYS = [f"gt_r{rr}{cc}" for rr in range(3) for cc in range(3)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_view00_mtcnn_visible_no091")
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "flame_bank_hybrid",
    )
    parser.add_argument(
        "--refinement-table-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "refinement_summary_tables",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "flame_bank_hybrid_tables",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return str(row["subject"]), str(row["frame"])


def matrix_from_row(row, prefix="pred"):
    return np.asarray(
        [[float(row[f"{prefix}_r{rr}{cc}"]) for cc in range(3)] for rr in range(3)],
        dtype=np.float64,
    )


def vector_from_row(row, prefix="pred"):
    return np.asarray([float(row[f"{prefix}_{axis}_mm"]) for axis in ("tx", "ty", "tz")], dtype=np.float64)


def format_cell(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def add_errors(row, rotation, translation):
    gt_rotation = matrix_from_row(row, "gt")
    gt_translation = vector_from_row(row, "gt")
    yaw, pitch, roll, geod = angular_metrics([rotation], [gt_rotation])
    err = translation - gt_translation
    out = dict(row)
    for rr in range(3):
        for cc in range(3):
            out[f"pred_r{rr}{cc}"] = f"{rotation[rr, cc]:.9f}"
            out[f"gt_r{rr}{cc}"] = f"{gt_rotation[rr, cc]:.9f}"
    out["pred_tx_mm"] = f"{translation[0]:.9f}"
    out["pred_ty_mm"] = f"{translation[1]:.9f}"
    out["pred_tz_mm"] = f"{translation[2]:.9f}"
    out["gt_tx_mm"] = f"{gt_translation[0]:.9f}"
    out["gt_ty_mm"] = f"{gt_translation[1]:.9f}"
    out["gt_tz_mm"] = f"{gt_translation[2]:.9f}"
    out["yaw_error_deg"] = f"{float(yaw[0]):.9f}"
    out["pitch_error_deg"] = f"{float(pitch[0]):.9f}"
    out["roll_error_deg"] = f"{float(roll[0]):.9f}"
    out["rot_mean_error_deg"] = f"{float((yaw[0] + pitch[0] + roll[0]) / 3.0):.9f}"
    out["geodesic_error_deg"] = f"{float(geod[0]):.9f}"
    out["tx_error_mm"] = f"{abs(err[0]):.9f}"
    out["ty_error_mm"] = f"{abs(err[1]):.9f}"
    out["tz_error_mm"] = f"{abs(err[2]):.9f}"
    out["translation_l2_error_mm"] = f"{float(np.linalg.norm(err)):.9f}"
    return out


def summary_row(rows, method, category, note=""):
    metric = summarize(rows)
    return {
        "category": category,
        "method": method,
        "frames": metric["frames"],
        "yaw": metric["yaw_mae_deg"],
        "pitch": metric["pitch_mae_deg"],
        "roll": metric["roll_mae_deg"],
        "maer": metric["maer_deg"],
        "ge": metric["geodesic_mean_deg"],
        "tx": metric["tx_mae_mm"],
        "ty": metric["ty_mae_mm"],
        "tz": metric["tz_mae_mm"],
        "maet": metric["translation_l2_mae_mm"],
        "add": "",
        "note": note,
    }


def write_markdown(path, title, rows):
    lines = [
        f"# {title}",
        "",
        "| Category | Method | Frames | Yaw | Pitch | Roll | MAEr | GE | Tx | Ty | Tz | MAEt | ADD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {method} | {frames} | {yaw} | {pitch} | {roll} | {maer} | {ge} | "
            "{tx} | {ty} | {tz} | {maet} | {add} |".format(
                category=row["category"],
                method=row["method"],
                frames=row["frames"],
                yaw=format_cell(row["yaw"]),
                pitch=format_cell(row["pitch"]),
                roll=format_cell(row["roll"]),
                maer=format_cell(row["maer"]),
                ge=format_cell(row["ge"]),
                tx=format_cell(row["tx"]),
                ty=format_cell(row["ty"]),
                tz=format_cell(row["tz"]),
                maet=format_cell(row["maet"]),
                add=format_cell(row["add"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def load_bank_rows(bank_dir):
    rows = []
    for path in sorted(bank_dir.glob("subject_*_predictions.csv")):
        rows.extend(read_csv(path))
    return rows


def main():
    args = parse_args()
    bank_dir = args.bank_root / args.protocol
    table_dir = args.refinement_table_root / args.protocol
    existing_summary = read_csv(table_dir / "famos_refinement_full_summary.csv")
    existing_predictions = read_csv(table_dir / "famos_refinement_full_predictions.csv")
    bank_rows = load_bank_rows(bank_dir)
    if not bank_rows:
        raise RuntimeError(f"No bank prediction rows under {bank_dir}")

    by_method = defaultdict(dict)
    for row in existing_predictions:
        by_method[row["method"]][key(row)] = row
    bank_by_key = {key(row): row for row in bank_rows}

    base_methods = {
        "ReVA-HPE (bank R + enrollment-PnP t)": (
            "Enrollment-PnP + relative",
            "Bank-anchor practical",
            "rotation from FLAME-bank RANSAC; translation copied from Enrollment-PnP + relative",
        ),
        "FLAME-bank R + AvatarBox-median t": (
            "AvatarBox-median",
            "Bank-anchor + AvatarBox",
            "rotation from FLAME-bank RANSAC; translation copied from AvatarBox-median",
        ),
        "FLAME-bank R + AvatarBox-trimmed-mean t": (
            "AvatarBox-trimmed-mean",
            "Bank-anchor + AvatarBox",
            "rotation from FLAME-bank RANSAC; translation copied from AvatarBox-trimmed-mean",
        ),
    }

    generated = {}
    direct_bank = []
    for row in bank_rows:
        out = add_errors(row, matrix_from_row(row, "pred"), vector_from_row(row, "pred"))
        out["method"] = "FLAME-bank RANSAC"
        out["category"] = "Bank-anchor practical"
        direct_bank.append(out)
    generated["FLAME-bank RANSAC"] = direct_bank

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
            out["bank_rotation_source"] = "FaMoS enrollment-conditioned FLAME-bank RANSAC"
            out["translation_source_method"] = base_method
            rows.append(out)
        generated[method] = rows

    bank_summary = [
        summary_row(generated["FLAME-bank RANSAC"], "FLAME-bank RANSAC", "Bank-anchor practical", "direct FLAME-bank RANSAC rotation and translation"),
    ]
    for method, (_, category, note) in base_methods.items():
        bank_summary.append(summary_row(generated[method], method, category, note))

    out_dir = args.out_root / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_predictions = [row for rows in generated.values() for row in rows]
    combined_summary = existing_summary + bank_summary

    summary_csv = out_dir / "famos_refinement_with_flame_bank_summary.csv"
    summary_md = out_dir / "famos_refinement_with_flame_bank_summary.md"
    predictions_csv = out_dir / "famos_refinement_with_flame_bank_predictions.csv"
    metadata_json = out_dir / "famos_refinement_with_flame_bank_metadata.json"

    summary_fields = ["category", "method", "frames", "yaw", "pitch", "roll", "maer", "ge", "tx", "ty", "tz", "maet", "add", "note"]
    write_csv(summary_csv, combined_summary, summary_fields)
    write_csv(predictions_csv, combined_predictions)
    write_markdown(summary_md, f"FaMoS {args.protocol} Refinement Summary With FLAME Bank", combined_summary)
    metadata = {
        "protocol": args.protocol,
        "bank_dir": str(bank_dir),
        "existing_refinement_summary": str(table_dir / "famos_refinement_full_summary.csv"),
        "existing_refinement_predictions": str(table_dir / "famos_refinement_full_predictions.csv"),
        "bank_rows": len(bank_rows),
        "generated_methods": {method: len(rows) for method, rows in generated.items()},
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "predictions_csv": str(predictions_csv),
        "note": "These are FLAME enrollment-bank rows. They do not require FaceLift/Gaussian avatar assets.",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2))

    combined_bank_path = bank_dir / "combined_predictions.csv"
    if not combined_bank_path.is_file():
        write_csv(combined_bank_path, bank_rows)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
