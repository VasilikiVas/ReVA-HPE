#!/usr/bin/env python3
"""Estimate a robust metric Sim(3) from FaceLift canonical space to FLAME."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RIGID_IDS = (27, 28, 29, 30, 31, 33, 35, 36, 39, 42, 45)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triangulated-landmarks", type=Path, required=True)
    parser.add_argument("--flame-landmarks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rigid-ids", type=int, nargs="+", default=DEFAULT_RIGID_IDS)
    parser.add_argument("--inlier-threshold-mm", type=float, default=12.0)
    return parser.parse_args()


def estimate_similarity(source, target):
    """Return scale, rotation, translation for target = s R source + t."""
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Similarity estimation needs matching Nx3 arrays with N >= 3")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered ** 2, axis=1)))
    if source_variance < 1e-12:
        raise ValueError("Degenerate source points")
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transpose
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Invalid similarity scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def transform_points(points, scale, rotation, translation):
    return scale * (points @ rotation.T) + translation


def is_nondegenerate_triplet(points):
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return singular_values[0] > 1e-8 and singular_values[1] / singular_values[0] > 0.03


def robust_similarity(source, target, threshold_m):
    best = None
    for sample in combinations(range(source.shape[0]), 3):
        sample = np.asarray(sample)
        if not is_nondegenerate_triplet(source[sample]):
            continue
        try:
            scale, rotation, translation = estimate_similarity(source[sample], target[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(
            transform_points(source, scale, rotation, translation) - target, axis=1
        )
        inliers = residuals <= threshold_m
        if inliers.sum() < 3:
            continue
        objective = (
            -int(inliers.sum()),
            float(np.median(residuals[inliers])),
            float(np.mean(residuals[inliers])),
        )
        if best is None or objective < best[0]:
            best = (objective, inliers)
    if best is None:
        raise RuntimeError("No valid Sim(3) RANSAC hypothesis")

    inliers = best[1]
    scale, rotation, translation = estimate_similarity(source[inliers], target[inliers])
    residuals = np.linalg.norm(
        transform_points(source, scale, rotation, translation) - target, axis=1
    )
    refined_inliers = residuals <= threshold_m
    if refined_inliers.sum() >= 3 and not np.array_equal(refined_inliers, inliers):
        scale, rotation, translation = estimate_similarity(
            source[refined_inliers], target[refined_inliers]
        )
        residuals = np.linalg.norm(
            transform_points(source, scale, rotation, translation) - target, axis=1
        )
        inliers = residuals <= threshold_m
    else:
        inliers = refined_inliers
    return scale, rotation, translation, residuals, inliers


def inverse_similarity(scale, rotation, translation):
    inverse_scale = 1.0 / scale
    inverse_rotation = rotation.T
    inverse_translation = -inverse_scale * (inverse_rotation @ translation)
    return inverse_scale, inverse_rotation, inverse_translation


def plot_alignment(source_aligned, target, selected_ids, inlier_ids, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    views = (
        (0, 1, "X right (m)", "Y down (m)", "Front"),
        (2, 1, "Z forward (m)", "Y down (m)", "Side"),
        (0, 2, "X right (m)", "Z forward (m)", "Top"),
    )
    selected_set = set(selected_ids)
    inlier_set = set(inlier_ids)
    for axis, (first, second, xlabel, ylabel, title) in zip(axes, views):
        axis.scatter(target[:, first], target[:, second], s=15, c="tab:blue", alpha=0.7,
                     label="FLAME target")
        axis.scatter(source_aligned[:, first], source_aligned[:, second], s=13,
                     c="tab:orange", alpha=0.7, label="aligned FaceLift")
        for landmark_id in selected_ids:
            color = "green" if landmark_id in inlier_set else "red"
            axis.plot(
                [target[landmark_id, first], source_aligned[landmark_id, first]],
                [target[landmark_id, second], source_aligned[landmark_id, second]],
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )
            axis.text(target[landmark_id, first], target[landmark_id, second],
                      str(landmark_id), fontsize=6, color=color)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        if second == 1:
            axis.invert_yaxis()
    axes[0].legend(fontsize=8)
    figure.suptitle(
        f"Rigid landmarks: {len(selected_set)}; RANSAC inliers: {len(inlier_set)}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def load_facelift_points(path):
    data = json.loads(path.read_text())
    records = sorted(data["landmarks"], key=lambda record: record["landmark_id"])
    points = np.asarray([record["xyz_front_canonical"] for record in records], dtype=np.float64)
    if points.shape != (68, 3):
        raise ValueError(f"Expected 68x3 FaceLift landmarks, got {points.shape}")
    return points, data


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source, triangulation_data = load_facelift_points(args.triangulated_landmarks)
    flame_data = json.loads(args.flame_landmarks.read_text())
    selected_ids = np.asarray(args.rigid_ids, dtype=int)
    threshold_m = args.inlier_threshold_mm / 1000.0

    results = {}
    for reference_name, reference in flame_data["canonical_references"].items():
        target = np.asarray(reference["landmarks_xyz_opencv"], dtype=np.float64)
        scale, rotation, translation, selected_residuals, selected_inliers = robust_similarity(
            source[selected_ids], target[selected_ids], threshold_m
        )
        aligned = transform_points(source, scale, rotation, translation)
        all_residuals = np.linalg.norm(aligned - target, axis=1)
        inverse_scale, inverse_rotation, inverse_translation = inverse_similarity(
            scale, rotation, translation
        )
        inlier_ids = selected_ids[selected_inliers]
        results[reference_name] = {
            "transform_facelift_to_flame_opencv": {
                "formula": "x_flame_m = scale_m_per_facelift_unit * R * x_facelift + t_m",
                "scale_m_per_facelift_unit": scale,
                "rotation": rotation.tolist(),
                "translation_m": translation.tolist(),
                "rotation_determinant": float(np.linalg.det(rotation)),
            },
            "transform_flame_opencv_to_facelift": {
                "scale_facelift_units_per_m": inverse_scale,
                "rotation": inverse_rotation.tolist(),
                "translation_facelift_units": inverse_translation.tolist(),
            },
            "ransac": {
                "threshold_mm": args.inlier_threshold_mm,
                "selected_landmark_ids": selected_ids.tolist(),
                "inlier_landmark_ids": inlier_ids.tolist(),
                "outlier_landmark_ids": selected_ids[~selected_inliers].tolist(),
                "selected_residual_mm_by_landmark": {
                    str(int(landmark_id)): float(residual * 1000.0)
                    for landmark_id, residual in zip(selected_ids, selected_residuals)
                },
                "median_inlier_residual_mm": float(
                    np.median(selected_residuals[selected_inliers]) * 1000.0
                ),
                "p95_inlier_residual_mm": float(
                    np.percentile(selected_residuals[selected_inliers], 95) * 1000.0
                ),
            },
            "validation": {
                "all_68_median_residual_mm": float(np.median(all_residuals) * 1000.0),
                "all_68_p95_residual_mm": float(np.percentile(all_residuals, 95) * 1000.0),
                "all_68_residual_mm_by_landmark": (all_residuals * 1000.0).tolist(),
            },
        }
        plot_alignment(
            aligned,
            target,
            selected_ids.tolist(),
            inlier_ids.tolist(),
            args.output_dir / f"alignment_{reference_name}.png",
        )

    preferred = "person_neutral"
    output = {
        "source_triangulation": str(args.triangulated_landmarks),
        "source_flame_fit": str(args.flame_landmarks),
        "uses_dataset_pose_or_depth_ground_truth": False,
        "preferred_reference": preferred,
        "notes": [
            "Metric scale comes from the FLAME anthropometric prior, not observed depth.",
            "Eyes and nose are used for the rigid fit; contour and expression-sensitive mouth points are validation only.",
            "person_expression and mean_neutral are retained as sensitivity checks.",
        ],
        "triangulation_reprojection_summary": triangulation_data["summary"],
        "references": results,
    }
    json_path = args.output_dir / "facelift_to_flame_similarity.json"
    json_path.write_text(json.dumps(output, indent=2))
    compact = {
        name: {
            "scale_m_per_unit": result["transform_facelift_to_flame_opencv"]["scale_m_per_facelift_unit"],
            "inliers": len(result["ransac"]["inlier_landmark_ids"]),
            "median_inlier_mm": result["ransac"]["median_inlier_residual_mm"],
            "all_68_median_mm": result["validation"]["all_68_median_residual_mm"],
        }
        for name, result in results.items()
    }
    print(json_path)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
