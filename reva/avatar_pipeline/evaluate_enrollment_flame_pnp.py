#!/usr/bin/env python3
"""Evaluate enrollment RGB pose from personalized FLAME landmarks and camera intrinsics."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "eval_scripts"))

import eval_biwi_crop as base


REFERENCES = ("person_expression", "person_neutral", "mean_neutral")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--biwi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=range(1, 25))
    parser.add_argument("--reprojection-threshold-px", type=float, default=5.0)
    return parser.parse_args()


def solve_pose(object_points, image_points, intrinsics, threshold_px):
    distortion = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        intrinsics.astype(np.float64),
        distortion,
        iterationsCount=300,
        reprojectionError=threshold_px,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        return None
    inlier_ids = inliers.reshape(-1)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        object_points[inlier_ids].astype(np.float64),
        image_points[inlier_ids].astype(np.float64),
        intrinsics.astype(np.float64),
        distortion,
        rotation_vec,
        translation_vec,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation, _ = cv2.Rodrigues(rotation_vec)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vec,
        translation_vec,
        intrinsics,
        distortion,
    )
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return rotation, translation_vec.reshape(3), inlier_ids, residuals


def angular_errors(rotation_pred, rotation_gt):
    import torch

    pred = torch.from_numpy(rotation_pred).float().unsqueeze(0)
    gt = torch.from_numpy(rotation_gt).float().unsqueeze(0)
    pred_euler = torch.stack(base.biwi_euler_deg_batch(pred), dim=-1)[0]
    gt_euler = torch.stack(base.biwi_euler_deg_batch(gt), dim=-1)[0]
    return base.angular_error(gt_euler, pred_euler).numpy()


def summarize(rows, reference):
    selected = [row for row in rows if row["reference"] == reference and row["success"]]
    metrics = (
        "yaw_error_deg",
        "pitch_error_deg",
        "roll_error_deg",
        "rot_mean_error_deg",
        "tx_error_mm",
        "ty_error_mm",
        "tz_error_mm",
        "translation_l2_error_mm",
        "reprojection_median_px",
        "reprojection_p95_px",
        "inliers",
    )
    summary = {"reference": reference, "subjects": len(selected)}
    for metric in metrics:
        summary[metric] = float(np.mean([float(row[metric]) for row in selected]))
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject_id in args.subjects:
        subject = f"{subject_id:02d}"
        subject_alignment = args.alignment_root / f"subject{subject}"
        similarity_path = subject_alignment / "similarity" / "facelift_to_flame_similarity.json"
        similarity = json.loads(similarity_path.read_text())
        fit_path = Path(similarity["source_flame_fit"])
        fit = json.loads(fit_path.read_text())
        image_points = np.asarray(fit["fit"]["detected_landmarks_xy"], dtype=np.float64)

        subject_dir = args.biwi_dir / subject
        intrinsics, rgb_rotation, rgb_translation = base.load_rgb_intrinsics(
            str(subject_dir / "rgb.cal")
        )
        first_frame = base.get_subject_frames(str(subject_dir))[0]
        gt_rotation, gt_translation = base.parse_pose_txt(first_frame[2])

        for reference in REFERENCES:
            object_points = np.asarray(
                fit["canonical_references"][reference]["landmarks_xyz_opencv"],
                dtype=np.float64,
            )
            result = solve_pose(
                object_points,
                image_points,
                intrinsics,
                args.reprojection_threshold_px,
            )
            if result is None:
                rows.append(
                    {
                        "subject": subject,
                        "reference": reference,
                        "success": 0,
                        "inliers": 0,
                        "yaw_error_deg": float("nan"),
                        "pitch_error_deg": float("nan"),
                        "roll_error_deg": float("nan"),
                        "rot_mean_error_deg": float("nan"),
                        "tx_error_mm": float("nan"),
                        "ty_error_mm": float("nan"),
                        "tz_error_mm": float("nan"),
                        "translation_l2_error_mm": float("nan"),
                        "signed_tx_error_mm": float("nan"),
                        "signed_ty_error_mm": float("nan"),
                        "signed_tz_error_mm": float("nan"),
                        "reprojection_median_px": float("nan"),
                        "reprojection_p95_px": float("nan"),
                    }
                )
                continue
            rotation_rgb, translation_rgb_m, inlier_ids, residuals = result
            rotation_depth = rgb_rotation.T @ rotation_rgb
            translation_depth = rgb_rotation.T @ (
                translation_rgb_m * 1000.0 - rgb_translation
            )
            angle = angular_errors(rotation_depth, gt_rotation)
            signed_translation = translation_depth - gt_translation
            absolute_translation = np.abs(signed_translation)
            rows.append(
                {
                    "subject": subject,
                    "reference": reference,
                    "success": 1,
                    "inliers": len(inlier_ids),
                    "yaw_error_deg": float(angle[0]),
                    "pitch_error_deg": float(angle[1]),
                    "roll_error_deg": float(angle[2]),
                    "rot_mean_error_deg": float(angle.mean()),
                    "tx_error_mm": float(absolute_translation[0]),
                    "ty_error_mm": float(absolute_translation[1]),
                    "tz_error_mm": float(absolute_translation[2]),
                    "translation_l2_error_mm": float(np.linalg.norm(signed_translation)),
                    "signed_tx_error_mm": float(signed_translation[0]),
                    "signed_ty_error_mm": float(signed_translation[1]),
                    "signed_tz_error_mm": float(signed_translation[2]),
                    "reprojection_median_px": float(np.median(residuals)),
                    "reprojection_p95_px": float(np.percentile(residuals, 95)),
                }
            )

    summaries = [summarize(rows, reference) for reference in REFERENCES]
    write_csv(args.output_dir / "per_subject.csv", rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    metadata = {
        "uses_biwi_pose_or_depth_as_prediction_input": False,
        "uses_rgb_intrinsics": True,
        "uses_rgb_to_depth_calibration_for_reporting_frame_only": True,
        "ground_truth_used_for_scoring_only": True,
        "subjects": len(args.subjects),
        "reprojection_threshold_px": args.reprojection_threshold_px,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
