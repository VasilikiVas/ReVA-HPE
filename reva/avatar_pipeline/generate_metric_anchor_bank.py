#!/usr/bin/env python3
"""Render a deterministic metric SE(3) anchor bank from a calibrated FaceLift avatar."""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facelift-root", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--similarity-json", type=Path, required=True)
    parser.add_argument("--rgb-cal", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=20)
    parser.add_argument(
        "--base-depth-m",
        type=float,
        default=0.0,
        help="Use a positive fixed depth, or <=0 for enrollment-only FLAME depth.",
    )
    parser.add_argument(
        "--base-depth-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale applied to the selected base depth.",
    )
    parser.add_argument("--minimum-base-depth-m", type=float, default=0.80)
    parser.add_argument("--dx-over-z-range", type=float, default=0.10)
    parser.add_argument("--dy-over-z-range", type=float, default=0.07)
    parser.add_argument("--log-z-range", type=float, default=0.18)
    parser.add_argument(
        "--view-hemisphere",
        choices=("full", "front"),
        default="front",
        help="Sample camera rotations on the visible face hemisphere or the full sphere.",
    )
    parser.add_argument("--preferred-reference", default="person_neutral")
    parser.add_argument(
        "--base-translation-source",
        choices=(
            "weak_centered",
            "pnp_mean_neutral",
            "pnp_mean_neutral_weak_depth",
            "pnp_person_neutral",
            "pnp_person_expression",
        ),
        default="weak_centered",
    )
    return parser.parse_args()


def load_rgb_intrinsics(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return np.asarray(
        [[float(value) for value in lines[index].split()] for index in range(3)],
        dtype=np.float64,
    )


def icosphere(subdivisions=2):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    vertices = [vertex for vertex in vertices]
    for _ in range(subdivisions):
        cache = {}

        def midpoint(first, second):
            key = tuple(sorted((first, second)))
            if key not in cache:
                value = vertices[first] + vertices[second]
                value /= np.linalg.norm(value)
                cache[key] = len(vertices)
                vertices.append(value)
            return cache[key]

        next_faces = []
        for first, second, third in faces:
            a = midpoint(first, second)
            b = midpoint(second, third)
            c = midpoint(third, first)
            next_faces.extend(
                [(first, a, c), (second, b, a), (third, c, b), (a, b, c)]
            )
        faces = next_faces
    return np.asarray(vertices, dtype=np.float64)


def farthest_point_directions(count, hemisphere="full"):
    points = icosphere(subdivisions=2)
    if hemisphere == "front":
        points = points[points[:, 2] <= 1e-9]
    elif hemisphere != "full":
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    if count > len(points):
        raise ValueError(f"Requested {count} directions from only {len(points)} candidates")
    # Start with the canonical frontal camera direction in OpenCV head axes.
    first = int(np.argmin(np.linalg.norm(points - np.asarray([0.0, 0.0, -1.0]), axis=1)))
    selected = [first]
    minimum_distance = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(1, count):
        last = points[selected[-1]]
        minimum_distance = np.minimum(
            minimum_distance, np.linalg.norm(points - last[None], axis=1)
        )
        minimum_distance[selected] = -1.0
        selected.append(int(np.argmax(minimum_distance)))
    directions = points[selected]
    directions[0] = np.asarray([0.0, 0.0, -1.0])
    return directions


def look_at_opencv(camera_center, target=None):
    if target is None:
        target = np.zeros(3, dtype=np.float64)
    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    image_up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, image_up))) > 0.97:
        image_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, image_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.stack([right, down, forward], axis=0)


def front_world_to_canonical_rotation():
    # FaceLift's frontal camera is at azimuth 270 degrees with world +Z up.
    camera_position = np.asarray([0.0, -2.7, 0.0], dtype=np.float64)
    forward = -camera_position / np.linalg.norm(camera_position)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation_c2w = np.stack((right, -up, forward), axis=1)
    return rotation_c2w.T


def pose_schedule(
    count,
    base_translation,
    dx_range,
    dy_range,
    log_z_range,
    view_hemisphere,
):
    base_translation = np.asarray(base_translation, dtype=np.float64)
    base_depth = float(base_translation[2])
    base_x_over_z = float(base_translation[0] / base_depth)
    base_y_over_z = float(base_translation[1] / base_depth)
    directions = farthest_point_directions(count, hemisphere=view_hemisphere)
    translation_directions = np.roll(
        farthest_point_directions(count, hemisphere="full"),
        max(1, count // 3),
        axis=0,
    )
    records = []
    for anchor_id, (direction, translation_direction) in enumerate(
        zip(directions, translation_directions)
    ):
        rotation = look_at_opencv(direction)
        dx_over_z = dx_range * float(translation_direction[0])
        dy_over_z = dy_range * float(translation_direction[1])
        log_z_offset = log_z_range * float(translation_direction[2])
        if anchor_id == 0:
            rotation = np.eye(3, dtype=np.float64)
            dx_over_z = 0.0
            dy_over_z = 0.0
            log_z_offset = 0.0
        z = base_depth * math.exp(log_z_offset)
        translation = np.asarray(
            [
                (base_x_over_z + dx_over_z) * z,
                (base_y_over_z + dy_over_z) * z,
                z,
            ],
            dtype=np.float64,
        )
        records.append(
            {
                "anchor_id": anchor_id,
                "icosphere_camera_direction": direction.tolist(),
                "rotation_h2c_opencv": rotation.tolist(),
                "translation_h2c_m": translation.tolist(),
                "dx_over_z": dx_over_z,
                "dy_over_z": dy_over_z,
                "log_z_offset": log_z_offset,
                "base_x_over_z": base_x_over_z,
                "base_y_over_z": base_y_over_z,
            }
        )
    return records


def estimate_pnp_translation(flame_fit, intrinsics, reference_name):
    object_points = np.asarray(
        flame_fit["canonical_references"][reference_name]["landmarks_xyz_opencv"],
        dtype=np.float64,
    )
    image_points = np.asarray(
        flame_fit["fit"]["detected_landmarks_xy"], dtype=np.float64
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        intrinsics,
        distortion,
        iterationsCount=300,
        reprojectionError=5.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        raise RuntimeError(f"PnP failed for reference {reference_name}")
    inlier_ids = inliers.reshape(-1)
    success, rotation_vec, translation_vec = cv2.solvePnP(
        object_points[inlier_ids],
        image_points[inlier_ids],
        intrinsics,
        distortion,
        rotation_vec,
        translation_vec,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError(f"PnP refinement failed for reference {reference_name}")
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vec,
        translation_vec,
        intrinsics,
        distortion,
    )
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    return translation_vec.reshape(3), {
        "reference": reference_name,
        "inliers": int(len(inlier_ids)),
        "median_reprojection_px": float(np.median(residuals)),
        "p95_reprojection_px": float(np.percentile(residuals, 95)),
    }


def metric_pose_to_facelift_c2w(rotation_h2c, translation_h2c, similarity):
    scale = float(similarity["scale_m_per_facelift_unit"])
    rotation_alignment = np.asarray(similarity["rotation"], dtype=np.float64)
    translation_alignment = np.asarray(similarity["translation_m"], dtype=np.float64)
    world_to_front = front_world_to_canonical_rotation()

    rotation_w2c = rotation_h2c @ rotation_alignment @ world_to_front
    translation_w2c = (
        rotation_h2c @ translation_alignment + translation_h2c
    ) / scale
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rotation_w2c
    w2c[:3, 3] = translation_w2c
    return np.linalg.inv(w2c), w2c


def render_with_alpha(model, renderer, c2w, intrinsics, height, width):
    device = model._xyz.device
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    with torch.no_grad():
        white = renderer(
            model, height, width, c2w_t, intrinsics_t,
            bg_color=(1.0, 1.0, 1.0),
        )["render"]
        black = renderer(
            model, height, width, c2w_t, intrinsics_t,
            bg_color=(0.0, 0.0, 0.0),
        )["render"]
    white_np = white.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
    black_np = black.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
    alpha = np.clip(1.0 - np.mean(white_np - black_np, axis=2), 0.0, 1.0)
    return (white_np * 255.0).astype(np.uint8), alpha


def mask_box(alpha, margin=0.25):
    ys, xs = np.where(alpha > 0.05)
    if len(xs) == 0:
        raise RuntimeError("Rendered anchor has no visible Gaussian foreground")
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    box_width = x2 - x1 + 1.0
    box_height = y2 - y1 + 1.0
    height, width = alpha.shape
    expanded = [
        max(0.0, x1 - margin * box_width),
        max(0.0, y1 - margin * box_height),
        min(float(width - 1), x2 + margin * box_width),
        min(float(height - 1), y2 + margin * box_height),
    ]
    touches_border = bool(x1 <= 0 or y1 <= 0 or x2 >= width - 1 or y2 >= height - 1)
    return expanded, touches_border


def projected_landmark_box(landmarks, rotation, translation, intrinsics, image_hw, margin=0.40):
    points_camera = landmarks @ rotation.T + translation
    if np.any(points_camera[:, 2] <= 1e-6):
        raise RuntimeError("Canonical landmarks project behind an anchor camera")
    pixels = np.stack(
        [
            intrinsics[0, 0] * points_camera[:, 0] / points_camera[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * points_camera[:, 1] / points_camera[:, 2] + intrinsics[1, 2],
        ],
        axis=1,
    )
    x1, y1 = pixels.min(axis=0)
    x2, y2 = pixels.max(axis=0)
    box_width = x2 - x1 + 1.0
    box_height = y2 - y1 + 1.0
    height, width = image_hw
    return [
        max(0.0, float(x1 - margin * box_width)),
        max(0.0, float(y1 - margin * box_height)),
        min(float(width - 1), float(x2 + margin * box_width)),
        min(float(height - 1), float(y2 + margin * box_height)),
    ]


def save_montage(images, records, output_path):
    thumb_width, thumb_height = 256, 192
    label_height = 42
    columns = 5
    rows = math.ceil(len(images) / columns)
    montage = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(montage)
    for index, (image, record) in enumerate(zip(images, records)):
        row, column = divmod(index, columns)
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        panel = Image.fromarray(image).resize((thumb_width, thumb_height))
        montage.paste(panel, (x, y))
        translation = record["translation_h2c_m"]
        draw.text(
            (x + 4, y + thumb_height + 3),
            f"a{index:02d} t=[{translation[0]:+.2f},{translation[1]:+.2f},{translation[2]:.2f}]m",
            fill="black",
        )
    montage.save(output_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("FaceLift Gaussian rendering requires CUDA")
    reference_image = Image.open(args.reference_image).convert("RGB")
    width, height = reference_image.size
    intrinsics = load_rgb_intrinsics(args.rgb_cal)
    intrinsics_vector = np.asarray(
        [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
        dtype=np.float64,
    )
    similarity_data = json.loads(args.similarity_json.read_text())
    similarity_reference_name = args.preferred_reference
    similarity = similarity_data["references"][similarity_reference_name][
        "transform_facelift_to_flame_opencv"
    ]
    flame_fit_path = Path(similarity_data["source_flame_fit"])
    flame_fit = json.loads(flame_fit_path.read_text())
    canonical_landmarks = np.asarray(
        flame_fit["canonical_references"][similarity_reference_name]["landmarks_xyz_opencv"],
        dtype=np.float64,
    )
    weak_scale = float(flame_fit["fit"]["weak_perspective_scale"])
    focal = 0.5 * float(intrinsics[0, 0] + intrinsics[1, 1])
    weak_depth_estimate_m = focal / weak_scale
    pnp_quality = None
    if args.base_translation_source == "weak_centered":
        enrollment_depth_estimate_m = weak_depth_estimate_m
        estimated_translation = np.asarray(
            [0.0, 0.0, enrollment_depth_estimate_m], dtype=np.float64
        )
        base_depth_source = "enrollment_rgb_flame_weak_perspective_centered"
    else:
        use_weak_depth = args.base_translation_source.endswith("_weak_depth")
        pnp_reference_name = args.base_translation_source.removeprefix("pnp_")
        if use_weak_depth:
            pnp_reference_name = pnp_reference_name.removesuffix("_weak_depth")
        estimated_translation, pnp_quality = estimate_pnp_translation(
            flame_fit, intrinsics, pnp_reference_name
        )
        if use_weak_depth:
            estimated_translation *= weak_depth_estimate_m / float(
                estimated_translation[2]
            )
            enrollment_depth_estimate_m = weak_depth_estimate_m
        else:
            enrollment_depth_estimate_m = float(estimated_translation[2])
        base_depth_source = f"enrollment_rgb_flame_{args.base_translation_source}"
    if estimated_translation[2] <= 0.0:
        raise RuntimeError(f"Estimated non-positive enrollment depth: {estimated_translation[2]}")
    base_depth_m = (
        args.base_depth_m
        if args.base_depth_m > 0.0
        else max(enrollment_depth_estimate_m, args.minimum_base_depth_m)
    )
    if args.base_depth_scale <= 0.0:
        raise RuntimeError(f"Expected positive base-depth scale, got {args.base_depth_scale}")
    base_depth_m *= args.base_depth_scale
    base_translation_m = estimated_translation * (
        base_depth_m / float(estimated_translation[2])
    )
    if args.base_depth_m > 0.0:
        base_depth_source += "_fixed_depth_override"
    if args.base_depth_scale != 1.0:
        base_depth_source += f"_scale_{args.base_depth_scale:.4g}"

    sys.path.insert(0, str(args.facelift_root))
    from gslrm.model.gaussians_renderer import GaussianModel, render_opencv_cam

    model = GaussianModel(sh_degree=3)
    model.load_ply(str(args.ply))
    model.to("cuda")

    records = pose_schedule(
        args.anchor_count,
        base_translation_m,
        args.dx_over_z_range,
        args.dy_over_z_range,
        args.log_z_range,
        args.view_hemisphere,
    )
    images = []
    clipped_count = 0
    for record in records:
        rotation_h2c = np.asarray(record["rotation_h2c_opencv"], dtype=np.float64)
        translation_h2c = np.asarray(record["translation_h2c_m"], dtype=np.float64)
        c2w, w2c = metric_pose_to_facelift_c2w(
            rotation_h2c, translation_h2c, similarity
        )
        image, alpha = render_with_alpha(
            model, render_opencv_cam, c2w, intrinsics_vector, height, width
        )
        foreground_box, clipped = mask_box(alpha)
        head_box = projected_landmark_box(
            canonical_landmarks,
            rotation_h2c,
            translation_h2c,
            intrinsics,
            (height, width),
        )
        clipped_count += int(clipped)
        image_name = f"anchor_{record['anchor_id']:02d}.png"
        mask_name = f"anchor_{record['anchor_id']:02d}_alpha.png"
        Image.fromarray(image).save(args.output_dir / image_name)
        Image.fromarray((alpha * 255.0).astype(np.uint8)).save(args.output_dir / mask_name)
        record.update(
            {
                "image": image_name,
                "alpha_mask": mask_name,
                "head_box_xyxy": head_box,
                "foreground_box_xyxy": foreground_box,
                "foreground_fraction": float(np.mean(alpha > 0.05)),
                "touches_image_border": clipped,
                "render_c2w_facelift_world": c2w.tolist(),
                "render_w2c_facelift_world": w2c.tolist(),
            }
        )
        images.append(image)

    output = {
        "source_ply": str(args.ply),
        "source_similarity": str(args.similarity_json),
        "source_rgb_calibration": str(args.rgb_cal),
        "source_reference_image": str(args.reference_image),
        "uses_dataset_pose_or_depth_ground_truth": False,
        "preferred_similarity_reference": similarity_reference_name,
        "image_size_hw": [height, width],
        "intrinsics": intrinsics.tolist(),
        "schedule": {
            "anchor_count": args.anchor_count,
            "base_depth_m": base_depth_m,
            "base_depth_scale": args.base_depth_scale,
            "base_translation_m": base_translation_m.tolist(),
            "base_depth_source": base_depth_source,
            "base_translation_source": args.base_translation_source,
            "enrollment_depth_estimate_m": enrollment_depth_estimate_m,
            "weak_depth_estimate_m": weak_depth_estimate_m,
            "minimum_base_depth_m": args.minimum_base_depth_m,
            "dx_over_z_range": args.dx_over_z_range,
            "dy_over_z_range": args.dy_over_z_range,
            "log_z_range": args.log_z_range,
            "view_hemisphere": args.view_hemisphere,
            "canonical_anchor_id": 0,
            "pnp_quality": pnp_quality,
        },
        "quality": {
            "clipped_anchor_count": clipped_count,
            "minimum_foreground_fraction": min(
                record["foreground_fraction"] for record in records
            ),
            "maximum_foreground_fraction": max(
                record["foreground_fraction"] for record in records
            ),
        },
        "anchors": records,
    }
    json_path = args.output_dir / "anchors.json"
    json_path.write_text(json.dumps(output, indent=2))
    montage_path = args.output_dir / "anchors_montage.png"
    save_montage(images, records, montage_path)
    print(json_path)
    print(montage_path)
    print(json.dumps(output["quality"], indent=2))


if __name__ == "__main__":
    main()
