#!/usr/bin/env python3

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inlier-threshold-px", type=float, default=3.0)
    return parser.parse_args()


def projection_matrix(camera):
    fx, fy, cx, cy = camera["intrinsics_fxfycxcy"]
    intrinsics = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    w2c = np.asarray(camera["w2c_opencv"], dtype=np.float64)
    return intrinsics @ w2c[:3, :]


def triangulate_dlt(projections, points, weights=None):
    rows = []
    if weights is None:
        weights = np.ones(len(projections), dtype=np.float64)
    for projection, (u, v), weight in zip(projections, points, weights):
        row_weight = np.sqrt(max(float(weight), 1e-8))
        rows.append(row_weight * (u * projection[2] - projection[0]))
        rows.append(row_weight * (v * projection[2] - projection[1]))
    _, _, vh = np.linalg.svd(np.stack(rows, axis=0), full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-10:
        raise ValueError("Degenerate homogeneous triangulation")
    return homogeneous[:3] / homogeneous[3]


def reprojection_errors(point, projections, observations):
    homogeneous = np.concatenate((point, [1.0]))
    errors = []
    depths = []
    for projection, observation in zip(projections, observations):
        projected = projection @ homogeneous
        depths.append(float(projected[2]))
        pixel = projected[:2] / projected[2]
        errors.append(float(np.linalg.norm(pixel - observation)))
    return np.asarray(errors), np.asarray(depths)


def robust_triangulate(projections, observations, scores, threshold):
    best = None
    for pair in combinations(range(len(projections)), 2):
        try:
            point = triangulate_dlt(
                [projections[index] for index in pair],
                [observations[index] for index in pair],
                [scores[index] for index in pair],
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors, depths = reprojection_errors(point, projections, observations)
        inliers = (errors <= threshold) & (depths > 0.0)
        if inliers.sum() < 2:
            continue
        objective = (
            -int(inliers.sum()),
            float(np.median(errors[inliers])),
            float(np.mean(errors[inliers])),
        )
        if best is None or objective < best[0]:
            best = (objective, inliers)

    if best is None:
        fallback = None
        candidate_sets = [tuple(range(len(projections)))]
        candidate_sets.extend(combinations(range(len(projections)), 2))
        for candidate in candidate_sets:
            try:
                point = triangulate_dlt(
                    [projections[index] for index in candidate],
                    [observations[index] for index in candidate],
                    [scores[index] for index in candidate],
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            errors, depths = reprojection_errors(point, projections, observations)
            if not np.all(np.isfinite(errors)):
                continue
            objective = (
                float(np.median(errors)),
                float(np.mean(errors)),
                int(np.sum(depths <= 0.0)),
            )
            if fallback is None or objective < fallback[0]:
                fallback = (objective, point, errors, depths)
        if fallback is None:
            raise RuntimeError("No valid triangulation hypothesis")
        _, point, errors, depths = fallback
        inliers = np.zeros(len(projections), dtype=bool)
        best_views = np.argsort(errors)[: min(2, len(projections))]
        inliers[best_views] = True
        return point, errors, depths, inliers
    inliers = best[1]
    point = triangulate_dlt(
        [projection for projection, keep in zip(projections, inliers) if keep],
        [observation for observation, keep in zip(observations, inliers) if keep],
        [score for score, keep in zip(scores, inliers) if keep],
    )
    errors, depths = reprojection_errors(point, projections, observations)
    final_inliers = (errors <= threshold) & (depths > 0.0)
    if final_inliers.sum() >= 2 and not np.array_equal(final_inliers, inliers):
        point = triangulate_dlt(
            [projection for projection, keep in zip(projections, final_inliers) if keep],
            [observation for observation, keep in zip(observations, final_inliers) if keep],
            [score for score, keep in zip(scores, final_inliers) if keep],
        )
        errors, depths = reprojection_errors(point, projections, observations)
        inliers = (errors <= threshold) & (depths > 0.0)
    else:
        inliers = final_inliers
    return point, errors, depths, inliers


def plot_landmarks(points, stable_ids, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = (
        (0, 2, "canonical X", "canonical Z", "Front"),
        (2, 1, "canonical Z", "canonical Y", "Side"),
        (0, 1, "canonical X", "canonical Y", "Top"),
    )
    stable_set = set(stable_ids)
    for axis, (first, second, xlabel, ylabel, title) in zip(axes, views):
        colors = ["tab:red" if index in stable_set else "tab:blue" for index in range(68)]
        axis.scatter(points[:, first], points[:, second], c=colors, s=18)
        for index in stable_ids:
            axis.text(points[index, first], points[index, second], str(index), fontsize=7)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.landmarks_json.read_text())
    names = list(data["views"])
    projections = [projection_matrix(data["views"][name]["camera"]) for name in names]
    observations_by_view = [
        np.asarray(data["views"][name]["landmarks_xy"], dtype=np.float64)
        for name in names
    ]
    scores_by_view = [
        np.asarray(data["views"][name]["landmark_scores"], dtype=np.float64)
        for name in names
    ]

    front_c2w = np.asarray(data["views"]["front"]["camera"]["c2w_opencv"])
    world_to_front_canonical = front_c2w[:3, :3].T
    records = []
    canonical_points = []
    all_inlier_errors = []
    for landmark_id in range(68):
        observations = np.stack(
            [observations[landmark_id] for observations in observations_by_view]
        )
        scores = np.asarray([scores[landmark_id] for scores in scores_by_view])
        point_world, errors, depths, inliers = robust_triangulate(
            projections,
            observations,
            scores,
            args.inlier_threshold_px,
        )
        point_canonical = world_to_front_canonical @ point_world
        canonical_points.append(point_canonical)
        all_inlier_errors.extend(errors[inliers].tolist())
        records.append(
            {
                "landmark_id": landmark_id,
                "xyz_facelift_world": point_world.tolist(),
                "xyz_front_canonical": point_canonical.tolist(),
                "reprojection_error_px_by_view": dict(zip(names, errors.tolist())),
                "camera_depth_by_view": dict(zip(names, depths.tolist())),
                "inlier_by_view": dict(zip(names, inliers.tolist())),
                "inlier_count": int(inliers.sum()),
                "median_inlier_error_px": float(np.median(errors[inliers])),
            }
        )

    canonical_points = np.stack(canonical_points, axis=0)
    stable_ids = data["stable_landmark_ids"]
    stable_errors = [
        error
        for record in records
        if record["landmark_id"] in stable_ids
        for view_name, error in record["reprojection_error_px_by_view"].items()
        if record["inlier_by_view"][view_name]
    ]
    output = {
        "source_landmarks_json": str(args.landmarks_json),
        "view_order": names,
        "inlier_threshold_px": args.inlier_threshold_px,
        "stable_landmark_ids": stable_ids,
        "summary": {
            "overall_median_inlier_error_px": float(np.median(all_inlier_errors)),
            "overall_p95_inlier_error_px": float(np.percentile(all_inlier_errors, 95)),
            "stable_median_inlier_error_px": float(np.median(stable_errors)),
            "minimum_inlier_count": min(record["inlier_count"] for record in records),
            "canonical_minimum": canonical_points.min(axis=0).tolist(),
            "canonical_maximum": canonical_points.max(axis=0).tolist(),
        },
        "landmarks": records,
    }
    json_path = args.output_dir / "triangulated_landmarks.json"
    json_path.write_text(json.dumps(output, indent=2))
    plot_path = args.output_dir / "triangulated_landmarks.png"
    plot_landmarks(canonical_points, stable_ids, plot_path)
    print(json_path)
    print(plot_path)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
