#!/usr/bin/env python
"""Evaluate query-conditioned FLAME icosphere anchors with SE(3) consensus on BIWI."""

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
EVAL_ROOT = REPO_ROOT / "evaluation" / "eval_scripts"
RECON_ROOT = Path(os.environ.get("BLENDED_IN_FLAMES_ROOT", str(REPO_ROOT.parent / "blended_in_flames")))
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ROOT))
if not (ROOT / "trial_photo_to_5views.py").is_file() and RECON_ROOT.is_dir():
    sys.path.insert(0, str(RECON_ROOT))

import eval_biwi_crop as base  # noqa: E402
import eval_biwi_crop_mtcnn_square_translation as translation  # noqa: E402
from eval_biwi_crop_mtcnn_square_compare import (  # noqa: E402
    _square_box_rgb,
    _union_mtcnn_boxes,
)
from eval_flame_fitting_biwi import (  # noqa: E402
    axis_angle_to_matrix,
    build_landmark_detector,
    detect_landmarks,
)
_TRIAL_MODULE = None


def _trial():
    global _TRIAL_MODULE
    if _TRIAL_MODULE is None:
        try:
            import trial_photo_to_5views as module
        except ImportError as exc:
            raise ImportError(
                "trial_photo_to_5views.py was not found "
            ) from exc
        _TRIAL_MODULE = module
    return _TRIAL_MODULE


LOG = logging.getLogger("query_flame_icosphere")
FLAME_TO_RGB = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _icosphere(subdivisions=2):
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

    vertices = [v for v in vertices]
    for _ in range(subdivisions):
        midpoint_cache = {}

        def midpoint(i, j):
            key = tuple(sorted((i, j)))
            if key not in midpoint_cache:
                value = vertices[i] + vertices[j]
                value /= np.linalg.norm(value)
                midpoint_cache[key] = len(vertices)
                vertices.append(value)
            return midpoint_cache[key]

        next_faces = []
        for i, j, k in faces:
            a, b, c = midpoint(i, j), midpoint(j, k), midpoint(k, i)
            next_faces.extend([(i, a, c), (j, b, a), (k, c, b), (a, b, c)])
        faces = next_faces
    return np.asarray(vertices, dtype=np.float64)


def _farthest_point_directions(count):
    points = _icosphere(subdivisions=2)
    first = int(np.argmax(points[:, 2]))
    selected = [first]
    min_dist = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(1, count):
        last = points[selected[-1]]
        min_dist = np.minimum(min_dist, np.linalg.norm(points - last[None], axis=1))
        min_dist[selected] = -1.0
        selected.append(int(np.argmax(min_dist)))
    return points[selected]


def _look_at_opencv(camera_center, target):
    forward = target - camera_center
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.97:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.stack([right, down, forward], axis=0)
    translation_vec = -rotation @ camera_center
    return rotation, translation_vec


def _expanded_mask_box(mask, margin=0.4):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Rendered anchor has an empty foreground mask")
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    width, height = x2 - x1 + 1.0, y2 - y1 + 1.0
    h, w = mask.shape
    return [
        max(0.0, x1 - margin * width),
        min(float(w - 1), x2 + margin * width),
        max(0.0, y1 - margin * height),
        min(float(h - 1), y2 + margin * height),
    ]


def _fitted_rgb_pose(fit, intrinsics):
    flame_rotation = axis_angle_to_matrix(fit["global_pose"].detach().cpu().numpy())
    rotation = FLAME_TO_RGB.astype(np.float64) @ flame_rotation @ FLAME_TO_RGB.astype(np.float64).T
    scale = float(fit["scale"].reshape(-1)[0].item())
    image_translation = fit["trans"].detach().cpu().numpy().reshape(2).astype(np.float64)
    focal = 0.5 * float(intrinsics[0, 0] + intrinsics[1, 1])
    translation_vec = np.asarray(
        [
            (image_translation[0] - intrinsics[0, 2]) / scale,
            (image_translation[1] - intrinsics[1, 2]) / scale,
            focal / scale,
        ],
        dtype=np.float64,
    )
    return rotation, translation_vec


def _render_anchors(
    flame,
    fit,
    colors,
    intrinsics,
    image_hw,
    count,
    device,
    pose_mode="centered_icosphere",
    dx_over_z_range=0.10,
    dy_over_z_range=0.05,
    log_z_range=0.10,
):
    from pytorch3d.renderer import (
        AmbientLights,
        BlendParams,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    canonical_pose = torch.cat([torch.zeros_like(fit["global_pose"]), fit["jaw_pose"]], dim=1)
    with torch.no_grad():
        vertices, _, _, _ = flame(
            shape_params=fit["shape"],
            expression_params=fit["expr"],
            pose_params=canonical_pose,
        )
    conversion = torch.as_tensor(FLAME_TO_RGB, dtype=vertices.dtype, device=device)
    vertices = vertices[0] @ conversion.T
    center = vertices.mean(dim=0)
    radius = float(torch.linalg.norm(vertices - center, dim=1).max().item())

    scale = float(fit["scale"].reshape(-1)[0].item())
    focal = 0.5 * float(intrinsics[0, 0] + intrinsics[1, 1])
    distance = max(focal / max(scale, 1e-6), 2.5 * radius)
    directions = _farthest_point_directions(count)
    center_np = center.detach().cpu().numpy().astype(np.float64)

    rotations, translations = [], []
    translation_offsets = np.zeros((count, 3), dtype=np.float64)
    if pose_mode == "centered_icosphere":
        for direction in directions:
            camera_center = center_np + distance * direction
            rotation, translation_vec = _look_at_opencv(camera_center, center_np)
            rotations.append(rotation)
            translations.append(translation_vec)
    elif pose_mode == "fit_conditioned_se3":
        fit_rotation, fit_translation = _fitted_rgb_pose(fit, intrinsics)
        distance = float(fit_translation[2])
        translation_directions = np.roll(directions, max(1, count // 3), axis=0)
        translation_offsets[:, 0] = dx_over_z_range * translation_directions[:, 0]
        translation_offsets[:, 1] = dy_over_z_range * translation_directions[:, 1]
        translation_offsets[:, 2] = log_z_range * translation_directions[:, 2]

        # Preserve one exact FLAME hypothesis while all other anchors perturb R and T.
        fit_anchor_idx = int(np.argmin(np.linalg.norm(directions - np.asarray([0.0, 0.0, -1.0]), axis=1)))
        translation_offsets[fit_anchor_idx] = 0.0
        for direction, (dx, dy, dlogz) in zip(directions, translation_offsets):
            view_rotation, _ = _look_at_opencv(direction, np.zeros(3, dtype=np.float64))
            rotation = view_rotation @ fit_rotation
            z = fit_translation[2] * math.exp(float(dlogz))
            translation_vec = np.asarray(
                [
                    fit_translation[0] + dx * fit_translation[2],
                    fit_translation[1] + dy * fit_translation[2],
                    z,
                ],
                dtype=np.float64,
            )
            rotations.append(rotation)
            translations.append(translation_vec)
    else:
        raise ValueError(f"Unknown anchor pose mode: {pose_mode}")

    rotations_np = np.asarray(rotations, dtype=np.float32)
    translations_np = np.asarray(translations, dtype=np.float32)
    height, width = image_hw
    rotations_t = torch.from_numpy(rotations_np).to(device)
    translations_t = torch.from_numpy(translations_np).to(device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    intrinsics_t = intrinsics_t.unsqueeze(0).expand(count, -1, -1)
    image_size_t = torch.tensor([[height, width]], dtype=torch.float32, device=device).expand(count, -1)
    cameras = cameras_from_opencv_projection(rotations_t, translations_t, intrinsics_t, image_size_t)

    mesh = Meshes(
        verts=[vertices],
        faces=[flame.faces_tensor.to(device)],
        textures=TexturesVertex(verts_features=colors.unsqueeze(0)),
    ).extend(count)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1,
            ),
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=AmbientLights(device=device),
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        ),
    )
    with torch.no_grad():
        rendered = renderer(mesh).clamp(0.0, 1.0).cpu().numpy()
    images_rgb = (rendered[..., :3] * 255.0).astype(np.uint8)
    masks = rendered[..., 3] > 0.5
    boxes = [_expanded_mask_box(mask) for mask in masks]
    return (
        images_rgb,
        boxes,
        rotations_np,
        translations_np,
        directions,
        translation_offsets,
        distance,
        radius,
    )


def _rotation_distance_deg(a, b):
    relative = a @ b.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _project_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _ransac_fuse(rotations, translations, rot_threshold_deg, trans_threshold_mm, min_inliers):
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    best = None
    for seed in range(len(rotations)):
        rot_residual = np.asarray([_rotation_distance_deg(r, rotations[seed]) for r in rotations])
        trans_residual = np.linalg.norm(translations - translations[seed], axis=1)
        inliers = (rot_residual <= rot_threshold_deg) & (trans_residual <= trans_threshold_mm)
        ids = np.flatnonzero(inliers)
        normalized_residual = (
            rot_residual[ids] / max(rot_threshold_deg, 1e-6)
            + trans_residual[ids] / max(trans_threshold_mm, 1e-6)
        )
        score = (len(ids), -float(normalized_residual.mean()) if len(ids) else -np.inf)
        if best is None or score > best[0]:
            best = (score, ids, seed)

    inlier_ids = best[1]
    if len(inlier_ids) < min_inliers:
        # A deterministic medoid fallback preserves an output while exposing failure in the CSV.
        inlier_ids = np.asarray([best[2]], dtype=np.int64)
    fused_rotation = _project_rotation(rotations[inlier_ids].mean(axis=0))
    fused_translation = np.median(translations[inlier_ids], axis=0)
    return fused_rotation, fused_translation, inlier_ids, len(best[1]) >= min_inliers


def _biwi_errors(rotation, translation_vec, gt_rotation, gt_translation):
    pred = torch.from_numpy(rotation).float().unsqueeze(0)
    gt = torch.from_numpy(gt_rotation).float().unsqueeze(0)
    pred_euler = torch.stack(base.biwi_euler_deg_batch(pred), dim=-1)[0]
    gt_euler = torch.stack(base.biwi_euler_deg_batch(gt), dim=-1)[0]
    angle = base.angular_error(gt_euler, pred_euler).numpy()
    trans_abs = np.abs(translation_vec - gt_translation)
    return angle, trans_abs, float(np.linalg.norm(translation_vec - gt_translation))


def _save_montage(path, query_rgb, anchors_rgb, inlier_ids, directions, translation_offsets):
    thumb_w, thumb_h = 240, 180
    panels = [np.asarray(Image.fromarray(query_rgb).resize((thumb_w, thumb_h)))]
    labels = ["query"]
    inliers = set(int(i) for i in inlier_ids)
    for idx, image in enumerate(anchors_rgb):
        panels.append(np.asarray(Image.fromarray(image).resize((thumb_w, thumb_h))))
        d = directions[idx]
        dt = translation_offsets[idx]
        labels.append(
            f"a{idx:02d} {'IN' if idx in inliers else 'OUT'} "
            f"r[{d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}] "
            f"t[{dt[0]:+.2f},{dt[1]:+.2f},{dt[2]:+.2f}]"
        )
    cols = 5
    rows = math.ceil(len(panels) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        row, col = divmod(idx, cols)
        x, y = col * thumb_w, row * (thumb_h + 24)
        canvas.paste(Image.fromarray(panel), (x, y))
        draw.text((x + 4, y + thumb_h + 4), label, fill="black")
    canvas.save(path)


def _summary(rows, expected_queries, failed_queries):
    if not rows:
        return {}
    metric_names = [
        "yaw_error_deg", "pitch_error_deg", "roll_error_deg", "rot_mean_error_deg",
        "tx_error_mm", "ty_error_mm", "tz_error_mm", "translation_l2_error_mm",
        "ransac_inliers", "fit_runtime_s", "inference_runtime_s",
    ]
    result = {
        "expected_queries": expected_queries,
        "queries": len(rows),
        "failed_queries": failed_queries,
        "coverage": len(rows) / max(expected_queries, 1),
    }
    for name in metric_names:
        result[name] = float(np.mean([float(row[name]) for row in rows]))
    result["ransac_success_rate"] = float(np.mean([int(row["ransac_success"]) for row in rows]))
    return result


def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    torch.manual_seed(0)
    np.random.seed(0)
    flame, flame_cfg = _trial().build_flame(device, args.flame_assets)
    landmark_detector = build_landmark_detector(str(device))
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True, device=device)

    model = base.load_vggt_model(
        args.base_checkpoint,
        args.lora_checkpoint,
        device,
        pose_encoding_type=args.pose_encoding_type,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    subject_dir = Path(args.biwi_dir) / f"{args.subject:02d}"
    if not subject_dir.is_dir():
        subject_dir = Path(args.biwi_dir) / str(args.subject)
    frames = base.get_subject_frames(str(subject_dir))[1:]
    if args.frame_stride > 1:
        frames = frames[::args.frame_stride]
    if args.max_queries > 0:
        frames = frames[:args.max_queries]
    intrinsics, rgb_rotation, rgb_translation = base.load_rgb_intrinsics(subject_dir / "rgb.cal")
    rgb_rotation_t = torch.from_numpy(rgb_rotation).float()
    rgb_translation_t = torch.from_numpy(rgb_translation).float()
    LOG.info("Subject %02d: %d queries, %d anchors each", args.subject, len(frames), args.anchor_count)

    rows = []
    pair_rows = []
    failure_rows = []
    previous_query_box = None
    for query_index, (frame_num, rgb_path, pose_path) in enumerate(frames):
        query_start = time.perf_counter()
        image_bgr = cv2.imread(rgb_path)
        if image_bgr is None:
            LOG.warning("Could not read %s", rgb_path)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_rotation, gt_translation = base.parse_pose_txt(pose_path)

        fit_start = time.perf_counter()
        try:
            landmarks = detect_landmarks(landmark_detector, image_rgb)
        except RuntimeError as exc:
            if "no face" not in str(exc).lower():
                raise
            LOG.warning("[%02d/%05d] FLAME fit skipped: %s", args.subject, frame_num, exc)
            failure_rows.append(
                {
                    "subject": f"{args.subject:02d}",
                    "frame_num": frame_num,
                    "rgb_path": rgb_path,
                    "error": str(exc),
                }
            )
            continue
        fit = _trial().fit_flame(flame, flame_cfg, landmarks, device, iters=(args.fit_rigid_iters, args.fit_full_iters))
        colors = _trial().sample_vertex_colors(flame, fit["verts_fit"], fit["scale"], fit["trans"], image_rgb, device)
        fit_runtime = time.perf_counter() - fit_start

        (
            anchors_rgb,
            anchor_boxes,
            anchor_rot_rgb,
            anchor_trans_rgb_m,
            directions,
            translation_offsets,
            distance,
            radius,
        ) = _render_anchors(
            flame,
            fit,
            colors,
            intrinsics,
            image_rgb.shape[:2],
            args.anchor_count,
            device,
            pose_mode=args.anchor_pose_mode,
            dx_over_z_range=args.anchor_dx_over_z_range,
            dy_over_z_range=args.anchor_dy_over_z_range,
            log_z_range=args.anchor_log_z_range,
        )
        _, previous_query_box = base.crop_face_mtcnn(
            image_bgr,
            mtcnn,
            previous_query_box,
            crop_size=args.crop_size,
            ad=args.mtcnn_ad,
        )

        predicted_rotations = []
        predicted_translations = []
        inference_start = time.perf_counter()
        for start in range(0, args.anchor_count, args.pair_batch_size):
            end = min(start + args.pair_batch_size, args.anchor_count)
            anchor_tensors, query_tensors = [], []
            anchor_depth_rotations, anchor_depth_translations = [], []
            for anchor_id in range(start, end):
                pair_box = _union_mtcnn_boxes(anchor_boxes[anchor_id], previous_query_box)
                anchor_bgr = cv2.cvtColor(anchors_rgb[anchor_id], cv2.COLOR_RGB2BGR)
                anchor_crop = _square_box_rgb(anchor_bgr, pair_box, args.crop_size)
                query_crop = _square_box_rgb(image_bgr, pair_box, args.crop_size)
                anchor_tensors.append(base.vggt_transform(anchor_crop))
                query_tensors.append(base.vggt_transform(query_crop))

                rotation_rgb = anchor_rot_rgb[anchor_id].astype(np.float64)
                translation_rgb_mm = anchor_trans_rgb_m[anchor_id].astype(np.float64) * 1000.0
                rotation_depth = rgb_rotation.T @ rotation_rgb
                translation_depth = rgb_rotation.T @ (translation_rgb_mm - rgb_translation)
                anchor_depth_rotations.append(torch.from_numpy(rotation_depth).float())
                anchor_depth_translations.append(torch.from_numpy(translation_depth).float())

            images = torch.stack([torch.stack(anchor_tensors), torch.stack(query_tensors)], dim=1).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                output = model(images=images)
            variants, translations = translation.decode_vggt_predictions(
                output["pose_enc"],
                torch.stack(anchor_depth_rotations),
                torch.stack(anchor_depth_translations),
                rgb_rotation_t,
                rgb_translation_t,
                pose_mode="relative_h2c",
                translation_scale=1000.0,
                pose_encoding_type=args.pose_encoding_type,
                translation_rotation_mode="predicted",
                translation_target_type="depth_norm_delta_t",
            )
            predicted_rotations.extend(variants["E_h2c_correct"].cpu().numpy())
            predicted_translations.extend(translations.cpu().numpy())

        inference_runtime = time.perf_counter() - inference_start
        fused_rotation, fused_translation, inlier_ids, ransac_success = _ransac_fuse(
            predicted_rotations,
            predicted_translations,
            args.ransac_rot_threshold_deg,
            args.ransac_trans_threshold_mm,
            args.ransac_min_inliers,
        )
        angle_error, translation_error, l2_error = _biwi_errors(
            fused_rotation, fused_translation, gt_rotation, gt_translation
        )
        row = {
            "subject": f"{args.subject:02d}",
            "frame_num": frame_num,
            "anchor_pose_mode": args.anchor_pose_mode,
            "anchors": args.anchor_count,
            "ransac_inliers": len(inlier_ids),
            "ransac_success": int(ransac_success),
            "inlier_ids": " ".join(str(int(i)) for i in inlier_ids),
            "yaw_error_deg": float(angle_error[0]),
            "pitch_error_deg": float(angle_error[1]),
            "roll_error_deg": float(angle_error[2]),
            "rot_mean_error_deg": float(angle_error.mean()),
            "tx_error_mm": float(translation_error[0]),
            "ty_error_mm": float(translation_error[1]),
            "tz_error_mm": float(translation_error[2]),
            "translation_l2_error_mm": l2_error,
            "pred_tx_mm": float(fused_translation[0]),
            "pred_ty_mm": float(fused_translation[1]),
            "pred_tz_mm": float(fused_translation[2]),
            "gt_tx_mm": float(gt_translation[0]),
            "gt_ty_mm": float(gt_translation[1]),
            "gt_tz_mm": float(gt_translation[2]),
            "fit_scale": float(fit["scale"].reshape(-1)[0]),
            "anchor_distance_model_units": distance,
            "head_radius_model_units": radius,
            "fit_runtime_s": fit_runtime,
            "inference_runtime_s": inference_runtime,
            "total_runtime_s": time.perf_counter() - query_start,
        }
        rows.append(row)

        inlier_set = set(int(i) for i in inlier_ids)
        for anchor_id, (rotation_pred, translation_pred) in enumerate(zip(predicted_rotations, predicted_translations)):
            anchor_angle, anchor_translation, anchor_l2 = _biwi_errors(
                rotation_pred, translation_pred, gt_rotation, gt_translation
            )
            pair_rows.append(
                {
                    "subject": f"{args.subject:02d}",
                    "frame_num": frame_num,
                    "anchor_id": anchor_id,
                    "anchor_pose_mode": args.anchor_pose_mode,
                    "anchor_dx_over_z_offset": float(translation_offsets[anchor_id, 0]),
                    "anchor_dy_over_z_offset": float(translation_offsets[anchor_id, 1]),
                    "anchor_log_z_offset": float(translation_offsets[anchor_id, 2]),
                    "anchor_tx_mm": float(anchor_trans_rgb_m[anchor_id, 0] * 1000.0),
                    "anchor_ty_mm": float(anchor_trans_rgb_m[anchor_id, 1] * 1000.0),
                    "anchor_tz_mm": float(anchor_trans_rgb_m[anchor_id, 2] * 1000.0),
                    "inlier": int(anchor_id in inlier_set),
                    "yaw_error_deg": float(anchor_angle[0]),
                    "pitch_error_deg": float(anchor_angle[1]),
                    "roll_error_deg": float(anchor_angle[2]),
                    "rot_mean_error_deg": float(anchor_angle.mean()),
                    "tx_error_mm": float(anchor_translation[0]),
                    "ty_error_mm": float(anchor_translation[1]),
                    "tz_error_mm": float(anchor_translation[2]),
                    "translation_l2_error_mm": anchor_l2,
                    "pred_tx_mm": float(translation_pred[0]),
                    "pred_ty_mm": float(translation_pred[1]),
                    "pred_tz_mm": float(translation_pred[2]),
                }
            )

        if query_index < args.max_visualizations:
            _save_montage(
                out_dir / f"subject_{args.subject:02d}_frame_{frame_num:05d}_anchors.png",
                image_rgb,
                anchors_rgb,
                inlier_ids,
                directions,
                translation_offsets,
            )
        LOG.info(
            "[%02d/%05d] inliers=%d/%d rot=%.3f deg trans=%.3f mm runtime=%.1fs",
            args.subject,
            frame_num,
            len(inlier_ids),
            args.anchor_count,
            angle_error.mean(),
            l2_error,
            time.perf_counter() - query_start,
        )

    prediction_path = out_dir / f"subject_{args.subject:02d}_predictions.csv"
    pair_path = out_dir / f"subject_{args.subject:02d}_pairwise.csv"
    if rows:
        with open(prediction_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(pair_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
    if failure_rows:
        failure_path = out_dir / f"subject_{args.subject:02d}_failures.csv"
        with open(failure_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failure_rows[0]))
            writer.writeheader()
            writer.writerows(failure_rows)

    summary = _summary(rows, len(frames), len(failure_rows))
    summary_path = out_dir / f"subject_{args.subject:02d}_summary.csv"
    if summary:
        with open(summary_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
    LOG.info("Summary: %s", summary)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biwi_dir", required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--lora_checkpoint", required=True)
    parser.add_argument("--flame_assets", default=str(RECON_ROOT / "assets"))
    parser.add_argument("--pose_encoding_type", default="absT_quaR_FoV")
    parser.add_argument("--anchor_count", type=int, default=20)
    parser.add_argument(
        "--anchor_pose_mode",
        choices=["centered_icosphere", "fit_conditioned_se3"],
        default="centered_icosphere",
    )
    parser.add_argument("--anchor_dx_over_z_range", type=float, default=0.10)
    parser.add_argument("--anchor_dy_over_z_range", type=float, default=0.05)
    parser.add_argument("--anchor_log_z_range", type=float, default=0.10)
    parser.add_argument("--pair_batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--mtcnn_ad", type=float, default=0.4)
    parser.add_argument("--ransac_rot_threshold_deg", type=float, default=10.0)
    parser.add_argument("--ransac_trans_threshold_mm", type=float, default=50.0)
    parser.add_argument("--ransac_min_inliers", type=int, default=3)
    parser.add_argument("--fit_rigid_iters", type=int, default=250)
    parser.add_argument("--fit_full_iters", type=int, default=400)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_visualizations", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
