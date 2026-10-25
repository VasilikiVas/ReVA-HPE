#!/usr/bin/env python3
"""Build and score per-frame avatar-R + enrollment-relative-t tables."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "eval_scripts"))

import eval_biwi_crop as base 


R_KEYS = [f"pred_r{r}{c}" for r in range(3) for c in range(3)]
GT_R_KEYS = [f"gt_r{r}{c}" for r in range(3) for c in range(3)]
T_KEYS = ("pred_tx_mm", "pred_ty_mm", "pred_tz_mm")
GT_T_KEYS = ("gt_tx_mm", "gt_ty_mm", "gt_tz_mm")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--relative-pose-root",
        type=Path,
        required=True,
        help="Root containing subject_XX/pose_predictions.csv from enrollment-relative eval.",
    )
    parser.add_argument(
        "--avatar-root",
        type=Path,
        required=True,
        help="Aggregated bank evaluation root (subject_XX dirs from the bank evaluator).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--front-direction-z-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--front-min-abs-z", type=float, default=0.0)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_key(row):
    return f"{int(row['subject']):02d}", int(row.get("frame", row.get("frame_num")))


def matrix_from_row(row, prefix="pred_r"):
    return np.asarray(
        [[float(row[f"{prefix}{r}{c}"]) for c in range(3)] for r in range(3)],
        dtype=np.float64,
    )


def translation_from_pair_row(row):
    return np.asarray(
        [float(row[f"pred_{axis}_mm"]) for axis in ("tx", "ty", "tz")],
        dtype=np.float64,
    )


def vector_from_row(row, keys):
    return np.asarray([float(row[key]) for key in keys], dtype=np.float64)


def project_rotation(matrix):
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def rotation_distance_deg(first, second):
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_medoid(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    if len(rotations) == 1:
        return rotations[0]
    costs = []
    for idx, rotation in enumerate(rotations):
        costs.append(sum(rotation_distance_deg(rotation, other) for other in rotations))
    return rotations[int(np.argmin(costs))]


def avatar_ransac_rotation(original, pair_rows):
    by_anchor = {int(row["anchor_id"]): row for row in pair_rows}
    inlier_ids = [int(value) for value in original["inlier_ids"].split() if value.strip()]
    if not inlier_ids:
        raise RuntimeError(f"No RANSAC inliers for {query_key(original)}")
    rotations = [matrix_from_row(by_anchor[anchor_id]) for anchor_id in inlier_ids]
    return project_rotation(np.mean(rotations, axis=0))


def avatar_best_fusion_rotation(pair_rows, front_direction_z_sign, front_min_abs_z, trim_fraction):
    front_rows = [
        row
        for row in pair_rows
        if front_direction_z_sign * float(row["anchor_direction_z"]) >= front_min_abs_z
    ]
    if not front_rows:
        front_rows = pair_rows
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in [0, 0.5)")
    if len(front_rows) >= 3:
        translations = np.stack([translation_from_pair_row(row) for row in front_rows])
        median_t = np.median(translations, axis=0)
        trim_count = int(math.floor(len(front_rows) * trim_fraction))
        keep_count = max(2, len(front_rows) - 2 * trim_count)
        keep = np.argsort(np.linalg.norm(translations - median_t[None, :], axis=1))[:keep_count]
        kept_rows = [front_rows[int(idx)] for idx in keep]
    else:
        kept_rows = front_rows
    return rotation_medoid([matrix_from_row(row) for row in kept_rows])


def angular_errors(rotation_pred, rotation_gt):
    pred = torch.from_numpy(rotation_pred).float().unsqueeze(0)
    gt = torch.from_numpy(rotation_gt).float().unsqueeze(0)
    pred_euler = torch.stack(base.biwi_euler_deg_batch(pred), dim=-1)[0]
    gt_euler = torch.stack(base.biwi_euler_deg_batch(gt), dim=-1)[0]
    return base.angular_error(gt_euler, pred_euler).numpy()


def make_hybrid_row(relative_row, avatar_rotation, method):
    row = dict(relative_row)
    row["method"] = method
    for r in range(3):
        for c in range(3):
            row[f"pred_r{r}{c}"] = float(avatar_rotation[r, c])
    pred_t = vector_from_row(row, T_KEYS)
    gt_t = vector_from_row(row, GT_T_KEYS)
    gt_rotation = matrix_from_row(row, prefix="gt_r")
    angle = angular_errors(avatar_rotation, gt_rotation)
    abs_t = np.abs(pred_t - gt_t)
    row["yaw_error_deg"] = float(angle[0])
    row["pitch_error_deg"] = float(angle[1])
    row["roll_error_deg"] = float(angle[2])
    row["rot_mean_error_deg"] = float(angle.mean())
    row["tx_error_mm"] = float(abs_t[0])
    row["ty_error_mm"] = float(abs_t[1])
    row["tz_error_mm"] = float(abs_t[2])
    row["translation_l2_error_mm"] = float(np.linalg.norm(pred_t - gt_t))
    return row


def summarize(rows):
    metrics = (
        "yaw_error_deg",
        "pitch_error_deg",
        "roll_error_deg",
        "rot_mean_error_deg",
        "tx_error_mm",
        "ty_error_mm",
        "tz_error_mm",
        "translation_l2_error_mm",
    )
    summary = {"method": rows[0]["method"], "queries": len(rows)}
    for metric in metrics:
        summary[metric] = float(np.mean([float(row[metric]) for row in rows]))
    return summary


def write_markdown(path, rows):
    headers = (
        "method",
        "queries",
        "rot_mean_error_deg",
        "tx_error_mm",
        "ty_error_mm",
        "tz_error_mm",
        "translation_l2_error_mm",
    )
    with path.open("w") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")


def load_relative_rows(root):
    rows = []
    for path in sorted(root.glob("subject_*/pose_predictions.csv")):
        rows.extend(read_csv(path))
    return rows


def load_avatar_rows(root):
    original = {}
    pairwise = defaultdict(list)
    for path in sorted(root.glob("subject_*_predictions.csv")):
        for row in read_csv(path):
            original[query_key(row)] = row
    for path in sorted(root.glob("subject_*_pairwise.csv")):
        for row in read_csv(path):
            pairwise[query_key(row)].append(row)
    return original, pairwise


def duplicate_keys(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[query_key(row)] += 1
    return sorted(key for key, count in counts.items() if count > 1)


def max_abs_translation_diff(a, b):
    if not a:
        return math.nan
    return max(
        float(np.max(np.abs(vector_from_row(row_a, T_KEYS) - vector_from_row(row_b, T_KEYS))))
        for row_a, row_b in zip(a, b)
    )


def max_abs_rotation_diff(rows, rotations):
    if not rows:
        return math.nan
    return max(
        float(np.max(np.abs(matrix_from_row(row) - rotation)))
        for row, rotation in zip(rows, rotations)
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    relative_rows = load_relative_rows(args.relative_pose_root)
    avatar_original, avatar_pairwise = load_avatar_rows(args.avatar_root)
    if not relative_rows:
        raise RuntimeError(f"No relative pose rows under {args.relative_pose_root}")
    rel_dupes = duplicate_keys(relative_rows)
    if rel_dupes:
        raise RuntimeError(f"Duplicate relative rows: {rel_dupes[:10]}")

    relative_by_key = {query_key(row): row for row in relative_rows}
    matched_keys = sorted(set(relative_by_key) & set(avatar_original) & set(avatar_pairwise))
    missing_avatar = sorted(set(relative_by_key) - set(matched_keys))
    if missing_avatar:
        raise RuntimeError(f"Missing avatar rows for {len(missing_avatar)} queries, first={missing_avatar[:5]}")

    hybrid_a = []
    hybrid_b = []
    a_rotations = []
    b_rotations = []
    for key in matched_keys:
        rel = relative_by_key[key]
        original = avatar_original[key]
        pairs = avatar_pairwise[key]
        rotation_a = avatar_ransac_rotation(original, pairs)
        rotation_b = avatar_best_fusion_rotation(
            pairs,
            args.front_direction_z_sign,
            args.front_min_abs_z,
            args.trim_fraction,
        )
        a_rotations.append(rotation_a)
        b_rotations.append(rotation_b)
        hybrid_a.append(make_hybrid_row(rel, rotation_a, "hybrid_A_avatar_ransac_R_relative_t"))
        hybrid_b.append(make_hybrid_row(rel, rotation_b, "hybrid_B_avatar_best_fusion_R_relative_t"))

    summaries = [summarize(hybrid_a), summarize(hybrid_b)]
    write_csv(args.out_dir / "hybrid_A_predictions.csv", hybrid_a)
    write_csv(args.out_dir / "hybrid_B_predictions.csv", hybrid_b)
    write_csv(args.out_dir / "hybrid_summary.csv", summaries)
    write_markdown(args.out_dir / "hybrid_summary.md", summaries)
    sanity = {
        "relative_rows": len(relative_rows),
        "matched_rows": len(matched_keys),
        "duplicate_relative_rows": 0,
        "missing_avatar_rows": 0,
        "hybrid_A_translation_max_abs_diff_vs_relative": max_abs_translation_diff(hybrid_a, [relative_by_key[key] for key in matched_keys]),
        "hybrid_B_translation_max_abs_diff_vs_relative": max_abs_translation_diff(hybrid_b, [relative_by_key[key] for key in matched_keys]),
        "hybrid_A_rotation_max_abs_diff_vs_avatar": max_abs_rotation_diff(hybrid_a, a_rotations),
        "hybrid_B_rotation_max_abs_diff_vs_avatar": max_abs_rotation_diff(hybrid_b, b_rotations),
        "hybrid_A_rotation_source": "avatar front multiradius joint SE3 RANSAC reconstructed from inlier pairwise rotations",
        "hybrid_B_rotation_source": (
            "avatar front multiradius translation-consistent rotation medoid, "
            f"front rule {args.front_direction_z_sign}*anchor_direction_z>={args.front_min_abs_z}, "
            f"trim_fraction={args.trim_fraction}"
        ),
        "translation_source": str(args.relative_pose_root),
        "avatar_source": str(args.avatar_root),
        "ground_truth_used_to_choose_rotation": False,
    }
    (args.out_dir / "hybrid_sanity.json").write_text(json.dumps(sanity, indent=2))
    print(args.out_dir / "hybrid_summary.md")
    print(json.dumps(sanity, indent=2))
    for row in summaries:
        print(
            f"{row['method']}: queries={row['queries']} rot={row['rot_mean_error_deg']:.4f} "
            f"tx={row['tx_error_mm']:.4f} ty={row['ty_error_mm']:.4f} "
            f"tz={row['tz_error_mm']:.4f} l2={row['translation_l2_error_mm']:.4f}"
        )


if __name__ == "__main__":
    main()
