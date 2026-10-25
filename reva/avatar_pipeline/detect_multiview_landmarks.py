#!/usr/bin/env python3
"""Detect and validate 68-point landmarks on controlled FaceLift renders."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import face_alignment
import numpy as np
from PIL import Image, ImageDraw


LANDMARK_CHAINS = (
    (range(0, 17), False),
    (range(17, 22), False),
    (range(22, 27), False),
    (range(27, 31), False),
    (range(31, 36), False),
    (range(36, 42), True),
    (range(42, 48), True),
    (range(48, 60), True),
    (range(60, 68), True),
)

STABLE_LANDMARK_IDS = (8, 27, 28, 29, 30, 31, 33, 35, 36, 39, 42, 45, 48, 54)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_views(views_root):
    generalized_layout = (views_root / "yaw" / "yaw_cameras.json").is_file()
    if generalized_layout:
        yaw_root = views_root / "yaw"
        pitch_root = views_root / "pitch"
        yaw_metadata = json.loads((yaw_root / "yaw_cameras.json").read_text())
        pitch_metadata = json.loads((pitch_root / "pitch_cameras.json").read_text())
        yaw_image = lambda offset: yaw_root / f"camera_yaw_{offset:+.0f}deg.png"
        pitch_image = lambda offset: pitch_root / f"camera_pitch_{offset:+.0f}deg.png"
        yaw_name = lambda offset: f"camera_yaw_{offset:+.0f}deg"
        pitch_name = lambda offset: f"camera_pitch_{offset:+.0f}deg"
    else:
        # Legacy flat layout: yaw views at the root, pitch in a subdirectory.
        yaw_root = views_root
        pitch_root = views_root / "pitch"
        yaw_metadata = json.loads((yaw_root / "cameras.json").read_text())
        pitch_metadata = json.loads((pitch_root / "pitch_cameras.json").read_text())
        yaw_image = lambda offset: yaw_root / f"camera_azimuth_{offset:+.0f}deg.png"
        pitch_image = lambda offset: pitch_root / f"camera_pitch_{offset:+.0f}deg.png"
        yaw_name = lambda offset: f"camera_azimuth_{offset:+.0f}deg"
        pitch_name = lambda offset: f"camera_pitch_{offset:+.0f}deg"
    yaw_by_name = {view["name"]: view for view in yaw_metadata["views"]}
    pitch_by_name = {view["name"]: view for view in pitch_metadata["views"]}
    return (
        (
            "yaw_minus20",
            yaw_image(-20.0),
            yaw_by_name[yaw_name(-20.0)],
        ),
        (
            "front",
            yaw_image(0.0),
            yaw_by_name[yaw_name(0.0)],
        ),
        (
            "yaw_plus20",
            yaw_image(20.0),
            yaw_by_name[yaw_name(20.0)],
        ),
        (
            "pitch_minus20",
            pitch_image(-20.0),
            pitch_by_name[pitch_name(-20.0)],
        ),
        (
            "pitch_plus20",
            pitch_image(20.0),
            pitch_by_name[pitch_name(20.0)],
        ),
    )


def draw_overlay(image, landmarks, scores, bbox):
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for chain, closed in LANDMARK_CHAINS:
        indices = list(chain)
        points = [tuple(landmarks[index]) for index in indices]
        if closed:
            points.append(points[0])
        draw.line(points, fill=(0, 220, 255), width=2)

    for index, ((x, y), score) in enumerate(zip(landmarks, scores)):
        color = (0, 220, 80) if score >= 0.75 else (255, 150, 0)
        radius = 3 if index in STABLE_LANDMARK_IDS else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if index in STABLE_LANDMARK_IDS:
            draw.text((x + 3, y - 9), str(index), fill=(20, 20, 20))

    left, top, right, bottom = bbox[:4]
    draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=2)
    return overlay


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_matrix(view_a, view_b):
    intrinsics_a = np.asarray(view_a["intrinsics_fxfycxcy"], dtype=np.float64)
    intrinsics_b = np.asarray(view_b["intrinsics_fxfycxcy"], dtype=np.float64)
    k_a = np.array(
        [[intrinsics_a[0], 0.0, intrinsics_a[2]], [0.0, intrinsics_a[1], intrinsics_a[3]], [0.0, 0.0, 1.0]]
    )
    k_b = np.array(
        [[intrinsics_b[0], 0.0, intrinsics_b[2]], [0.0, intrinsics_b[1], intrinsics_b[3]], [0.0, 0.0, 1.0]]
    )
    w2c_a = np.asarray(view_a["w2c_opencv"], dtype=np.float64)
    w2c_b = np.asarray(view_b["w2c_opencv"], dtype=np.float64)
    relative_rotation = w2c_b[:3, :3] @ w2c_a[:3, :3].T
    relative_translation = w2c_b[:3, 3] - relative_rotation @ w2c_a[:3, 3]
    essential = skew(relative_translation) @ relative_rotation
    return np.linalg.inv(k_b).T @ essential @ np.linalg.inv(k_a)


def sampson_distances(points_a, points_b, fundamental):
    ones = np.ones((points_a.shape[0], 1), dtype=np.float64)
    homogeneous_a = np.concatenate((points_a, ones), axis=1)
    homogeneous_b = np.concatenate((points_b, ones), axis=1)
    lines_b = (fundamental @ homogeneous_a.T).T
    lines_a = (fundamental.T @ homogeneous_b.T).T
    numerator = np.sum(homogeneous_b * lines_b, axis=1)
    denominator = (
        lines_b[:, 0] ** 2
        + lines_b[:, 1] ** 2
        + lines_a[:, 0] ** 2
        + lines_a[:, 1] ** 2
    )
    return np.abs(numerator) / np.sqrt(np.maximum(denominator, 1e-12))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = load_views(args.views_root)
    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=args.device,
        flip_input=False,
        compile=False,
    )

    results = {}
    overlays = []
    for name, image_path, camera in views:
        image = Image.open(image_path).convert("RGB")
        landmark_lists, score_lists, bboxes = detector.get_landmarks(
            np.asarray(image),
            return_bboxes=True,
            return_landmark_score=True,
        )
        if not landmark_lists:
            print(f"warning: no face detected in {image_path}, skipping view {name}")
            continue
        largest_index = int(
            np.argmax(
                [
                    max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    for box in bboxes
                ]
            )
        )
        landmarks = np.asarray(landmark_lists[largest_index], dtype=np.float64)
        scores = np.asarray(score_lists[largest_index], dtype=np.float64)
        bbox = np.asarray(bboxes[largest_index], dtype=np.float64)
        overlay = draw_overlay(image, landmarks, scores, bbox)
        overlay.save(args.output_dir / f"{name}_landmarks.png")
        overlays.append((name, overlay))
        results[name] = {
            "image_path": str(image_path),
            "landmarks_xy": landmarks.tolist(),
            "landmark_scores": scores.tolist(),
            "bbox_xyxy_score": bbox.tolist(),
            "mean_landmark_score": float(scores.mean()),
            "minimum_landmark_score": float(scores.min()),
            "camera": camera,
        }

    if "front" not in results:
        raise RuntimeError("Front view landmarks are required for canonical alignment")
    if len(results) < 2:
        raise RuntimeError(f"Need at least two detected views, got {len(results)}")

    pairwise = []
    all_distances = []
    for name_a, name_b in combinations(results, 2):
        result_a = results[name_a]
        result_b = results[name_b]
        fundamental = fundamental_matrix(result_a["camera"], result_b["camera"])
        distances = sampson_distances(
            np.asarray(result_a["landmarks_xy"]),
            np.asarray(result_b["landmarks_xy"]),
            fundamental,
        )
        all_distances.append(distances)
        pairwise.append(
            {
                "view_a": name_a,
                "view_b": name_b,
                "median_sampson_px": float(np.median(distances)),
                "mean_sampson_px": float(np.mean(distances)),
                "p95_sampson_px": float(np.percentile(distances, 95)),
            }
        )

    stacked_distances = np.stack(all_distances, axis=0)
    metadata = {
        "stable_landmark_ids": list(STABLE_LANDMARK_IDS),
        "views": results,
        "epipolar_consistency": {
            "pairwise": pairwise,
            "overall_median_px": float(np.median(stacked_distances)),
            "overall_mean_px": float(np.mean(stacked_distances)),
            "overall_p95_px": float(np.percentile(stacked_distances, 95)),
            "stable_subset_median_px": float(
                np.median(stacked_distances[:, STABLE_LANDMARK_IDS])
            ),
            "per_landmark_median_px": np.median(stacked_distances, axis=0).tolist(),
        },
    }
    metadata_path = args.output_dir / "multiview_landmarks.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    label_height = 40
    width, height = overlays[0][1].size
    montage = Image.new("RGB", (len(overlays) * width, height + label_height), "white")
    draw = ImageDraw.Draw(montage)
    for index, (name, overlay) in enumerate(overlays):
        x = index * width
        draw.text((x + 10, 12), name, fill="black")
        montage.paste(overlay, (x, label_height))
    montage_path = args.output_dir / "multiview_landmarks.png"
    montage.save(montage_path)

    print(montage_path)
    print(metadata_path)
    for name, result in results.items():
        print(
            f"{name}: mean={result['mean_landmark_score']:.4f} "
            f"min={result['minimum_landmark_score']:.4f}"
        )
    consistency = metadata["epipolar_consistency"]
    print(
        "epipolar: "
        f"median={consistency['overall_median_px']:.3f}px "
        f"p95={consistency['overall_p95_px']:.3f}px "
        f"stable_median={consistency['stable_subset_median_px']:.3f}px"
    )


if __name__ == "__main__":
    main()
