#!/usr/bin/env python
"""Evaluate weak-perspective photo-to-FLAME fitting on the BIWI query protocol."""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


PREDICTION_FIELDS = [
    "subject",
    "frame_num",
    "intrinsics_mode",
    "rgb_path",
    "status",
    "error",
    "runtime_s",
    "landmark_mse_px2",
    "scale_px_per_m",
    "image_tx_px",
    "image_ty_px",
    "focal_x_px",
    "focal_y_px",
    "principal_x_px",
    "principal_y_px",
    "gt_yaw_deg",
    "gt_pitch_deg",
    "gt_roll_deg",
    "pred_yaw_deg",
    "pred_pitch_deg",
    "pred_roll_deg",
    "yaw_error_deg",
    "pitch_error_deg",
    "roll_error_deg",
    "rot_mean_error_deg",
    "rot_geodesic_error_deg",
    "gt_tx_mm",
    "gt_ty_mm",
    "gt_tz_mm",
    "pred_tx_mm",
    "pred_ty_mm",
    "pred_tz_mm",
    "tx_error_mm",
    "ty_error_mm",
    "tz_error_mm",
    "translation_l2_error_mm",
    "gt_tx_over_z",
    "gt_ty_over_z",
    "pred_tx_over_z",
    "pred_ty_over_z",
    "tx_over_z_error",
    "ty_over_z_error",
    "log_z_error",
]

METRIC_FIELDS = [
    "yaw_error_deg",
    "pitch_error_deg",
    "roll_error_deg",
    "rot_mean_error_deg",
    "rot_geodesic_error_deg",
    "tx_error_mm",
    "ty_error_mm",
    "tz_error_mm",
    "translation_l2_error_mm",
    "tx_over_z_error",
    "ty_over_z_error",
    "log_z_error",
    "landmark_mse_px2",
    "runtime_s",
]

SUMMARY_FIELDS = [
    "subject",
    "model",
    "intrinsics_mode",
    "expected_queries",
    "completed_queries",
    "successful_queries",
    "coverage",
] + [f"mean_{field}" for field in METRIC_FIELDS]


def get_subject_frames(subject_dir):
    frames = []
    for rgb_path in sorted(Path(subject_dir).glob("frame_*_rgb.png")):
        stem = rgb_path.name
        try:
            frame_num = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        pose_path = Path(str(rgb_path).replace("_rgb.png", "_pose.txt"))
        if pose_path.is_file():
            frames.append((frame_num, rgb_path, pose_path))
    return frames


def parse_pose_txt(path):
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    rotation = np.array(
        [[float(value) for value in lines[i].split()] for i in range(3)],
        dtype=np.float64,
    )
    translation = np.array([float(value) for value in lines[3].split()], dtype=np.float64)
    return rotation, translation


def load_rgb_calibration(path):
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    intrinsics = np.array(
        [[float(value) for value in lines[i].split()] for i in range(3)],
        dtype=np.float64,
    )
    rotation = np.array(
        [[float(value) for value in lines[i].split()] for i in range(4, 7)],
        dtype=np.float64,
    )
    translation = np.array([float(value) for value in lines[7].split()], dtype=np.float64)
    return intrinsics, rotation, translation


def fixed_fov_intrinsics(image_shape, fov_deg):
    height, width = image_shape[:2]
    focal = 0.5 * float(height) / math.tan(0.5 * math.radians(fov_deg))
    return np.array(
        [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis_angle):
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def biwi_euler_deg(rotation):
    transposed = rotation.T
    yaw = -math.degrees(
        math.atan2(
            -transposed[2, 0],
            math.sqrt(transposed[2, 1] ** 2 + transposed[2, 2] ** 2),
        )
    )
    pitch = math.degrees(math.atan2(transposed[2, 1], transposed[2, 2]))
    roll = -math.degrees(math.atan2(transposed[1, 0], transposed[0, 0]))
    return np.array([yaw, pitch, roll], dtype=np.float64)


def angular_error_deg(prediction, target):
    return np.abs((prediction - target + 180.0) % 360.0 - 180.0)


def rotation_geodesic_deg(prediction, target):
    relative = prediction @ target.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def build_landmark_detector(device):
    import face_alignment

    try:
        landmark_type = face_alignment.LandmarksType.TWO_D
    except AttributeError:
        landmark_type = face_alignment.LandmarksType._2D
    return face_alignment.FaceAlignment(landmark_type, flip_input=False, device=device)


def detect_landmarks(detector, image_rgb):
    predictions = detector.get_landmarks(image_rgb)
    if not predictions:
        raise RuntimeError("face_alignment found no face")
    best = max(predictions, key=lambda points: np.ptp(points[:, 0]) * np.ptp(points[:, 1]))
    return np.asarray(best, dtype=np.float32)


def fitted_pose_in_biwi_depth_frame(fit, intrinsics, rgb_rotation, rgb_translation):
    # FLAME uses x-right/y-up/z-back while BIWI RGB uses x-right/y-down/z-forward.
    flame_to_rgb = np.diag([1.0, -1.0, -1.0])
    flame_rotation = axis_angle_to_matrix(fit["global_pose"].detach().cpu().numpy())
    pred_rotation_rgb = flame_to_rgb @ flame_rotation @ flame_to_rgb.T

    scale = float(fit["scale"].detach().cpu().reshape(-1)[0])
    image_translation = fit["trans"].detach().cpu().numpy().reshape(2).astype(np.float64)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid fitted weak-perspective scale: {scale}")

    focal = 0.5 * (intrinsics[0, 0] + intrinsics[1, 1])
    pred_translation_rgb = np.array(
        [
            (image_translation[0] - intrinsics[0, 2]) / scale,
            (image_translation[1] - intrinsics[1, 2]) / scale,
            focal / scale,
        ],
        dtype=np.float64,
    ) * 1000.0

    pred_rotation_depth = rgb_rotation.T @ pred_rotation_rgb
    pred_translation_depth = rgb_rotation.T @ (pred_translation_rgb - rgb_translation)
    return pred_rotation_depth, pred_translation_depth, scale, image_translation


def score_fit(fit, gt_rotation, gt_translation, intrinsics, rgb_rotation, rgb_translation):
    pred_rotation, pred_translation, scale, image_translation = fitted_pose_in_biwi_depth_frame(
        fit, intrinsics, rgb_rotation, rgb_translation
    )
    gt_euler = biwi_euler_deg(gt_rotation)
    pred_euler = biwi_euler_deg(pred_rotation)
    euler_error = angular_error_deg(pred_euler, gt_euler)
    translation_error = np.abs(pred_translation - gt_translation)
    translation_l2 = float(np.linalg.norm(pred_translation - gt_translation))

    gt_z = max(float(gt_translation[2]), 1e-6)
    pred_z = max(float(pred_translation[2]), 1e-6)
    gt_norm = gt_translation[:2] / gt_z
    pred_norm = pred_translation[:2] / pred_z

    landmark_error = (
        (fit["lmk_proj"] - fit["lmk_gt"]).square().sum(dim=-1).mean().detach().cpu().item()
    )
    return {
        "landmark_mse_px2": landmark_error,
        "scale_px_per_m": scale,
        "image_tx_px": float(image_translation[0]),
        "image_ty_px": float(image_translation[1]),
        "focal_x_px": float(intrinsics[0, 0]),
        "focal_y_px": float(intrinsics[1, 1]),
        "principal_x_px": float(intrinsics[0, 2]),
        "principal_y_px": float(intrinsics[1, 2]),
        "gt_yaw_deg": float(gt_euler[0]),
        "gt_pitch_deg": float(gt_euler[1]),
        "gt_roll_deg": float(gt_euler[2]),
        "pred_yaw_deg": float(pred_euler[0]),
        "pred_pitch_deg": float(pred_euler[1]),
        "pred_roll_deg": float(pred_euler[2]),
        "yaw_error_deg": float(euler_error[0]),
        "pitch_error_deg": float(euler_error[1]),
        "roll_error_deg": float(euler_error[2]),
        "rot_mean_error_deg": float(euler_error.mean()),
        "rot_geodesic_error_deg": rotation_geodesic_deg(pred_rotation, gt_rotation),
        "gt_tx_mm": float(gt_translation[0]),
        "gt_ty_mm": float(gt_translation[1]),
        "gt_tz_mm": float(gt_translation[2]),
        "pred_tx_mm": float(pred_translation[0]),
        "pred_ty_mm": float(pred_translation[1]),
        "pred_tz_mm": float(pred_translation[2]),
        "tx_error_mm": float(translation_error[0]),
        "ty_error_mm": float(translation_error[1]),
        "tz_error_mm": float(translation_error[2]),
        "translation_l2_error_mm": translation_l2,
        "gt_tx_over_z": float(gt_norm[0]),
        "gt_ty_over_z": float(gt_norm[1]),
        "pred_tx_over_z": float(pred_norm[0]),
        "pred_ty_over_z": float(pred_norm[1]),
        "tx_over_z_error": float(abs(pred_norm[0] - gt_norm[0])),
        "ty_over_z_error": float(abs(pred_norm[1] - gt_norm[1])),
        "log_z_error": float(abs(math.log(pred_z / gt_z))),
    }


def read_csv_rows(path):
    if not Path(path).is_file():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def mean_metric(rows, field):
    values = []
    for row in rows:
        value = row.get(field, "")
        if value not in (None, ""):
            values.append(float(value))
    return float(np.mean(values)) if values else math.nan


def summarize_rows(subject, rows, expected_queries, intrinsics_mode):
    mode_rows = [
        row
        for row in rows
        if row.get("intrinsics_mode", "calibrated") == intrinsics_mode
    ]
    successful = [row for row in mode_rows if row.get("status") == "ok"]
    summary = {
        "subject": subject,
        "model": f"flame_weakpersp_landmark_fit_{intrinsics_mode}",
        "intrinsics_mode": intrinsics_mode,
        "expected_queries": expected_queries,
        "completed_queries": len(mode_rows),
        "successful_queries": len(successful),
        "coverage": len(successful) / expected_queries if expected_queries else 0.0,
    }
    for field in METRIC_FIELDS:
        summary[f"mean_{field}"] = mean_metric(successful, field)
    return summary


def write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evaluate_subject(args):
    subject = f"{args.subject:02d}"
    subject_dir = Path(args.biwi_dir) / subject
    frames = get_subject_frames(subject_dir)
    if len(frames) < 2:
        raise RuntimeError(f"Subject {subject} has fewer than two valid BIWI frames")

    # Match VGGT-HPE: frame 0 is reserved as the anchor and is not scored.
    query_frames = frames[1:]
    if args.max_frames is not None:
        query_frames = query_frames[: args.max_frames]

    output_dir = Path(args.out_dir) / f"subject_{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    existing_rows = read_csv_rows(predictions_path)
    completed = {
        (int(row["frame_num"]), row.get("intrinsics_mode", "calibrated"))
        for row in existing_rows
    }
    intrinsics_modes = ("calibrated", "fixed_fov30")

    sys.path.insert(0, str(Path(args.blended_root).resolve()))
    import trial_photo_to_5views as trial

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    detector = build_landmark_detector(args.landmark_device)
    flame, config = trial.build_flame(device, Path(args.blended_root) / "assets")
    calibrated_intrinsics, rgb_rotation, rgb_translation = load_rgb_calibration(
        subject_dir / "rgb.cal"
    )

    append_header = not predictions_path.is_file() or predictions_path.stat().st_size == 0
    with open(predictions_path, "a", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        if append_header:
            writer.writeheader()

        for index, (frame_num, rgb_path, pose_path) in enumerate(query_frames, start=1):
            pending_modes = [
                mode for mode in intrinsics_modes if (frame_num, mode) not in completed
            ]
            if not pending_modes:
                continue
            started = time.time()
            rows_to_write = []
            try:
                image_rgb = trial.load_image(rgb_path)
                landmarks = detect_landmarks(detector, image_rgb)
                fit = trial.fit_flame(
                    flame,
                    config,
                    landmarks,
                    device,
                    iters=(args.rigid_iters, args.full_iters),
                    n_shape_fit=args.n_shape_fit,
                )
                gt_rotation, gt_translation = parse_pose_txt(pose_path)
                mode_intrinsics = {
                    "calibrated": calibrated_intrinsics,
                    "fixed_fov30": fixed_fov_intrinsics(image_rgb.shape, args.fixed_fov_deg),
                }
                for mode in pending_modes:
                    row = {field: "" for field in PREDICTION_FIELDS}
                    row.update(
                        {
                            "subject": subject,
                            "frame_num": frame_num,
                            "intrinsics_mode": mode,
                            "rgb_path": str(rgb_path),
                            "status": "ok",
                            "error": "",
                        }
                    )
                    row.update(
                        score_fit(
                            fit,
                            gt_rotation,
                            gt_translation,
                            mode_intrinsics[mode],
                            rgb_rotation,
                            rgb_translation,
                        )
                    )
                    rows_to_write.append(row)
                del fit
            except Exception as exc:  # Keep failed detections/fits in the coverage denominator.
                error = f"{type(exc).__name__}: {exc}"
                for mode in pending_modes:
                    row = {field: "" for field in PREDICTION_FIELDS}
                    row.update(
                        {
                            "subject": subject,
                            "frame_num": frame_num,
                            "intrinsics_mode": mode,
                            "rgb_path": str(rgb_path),
                            "status": "error",
                            "error": error,
                        }
                    )
                    rows_to_write.append(row)
                print(f"[subject {subject} frame {frame_num}] ERROR: {error}", flush=True)

            runtime = time.time() - started
            for row in rows_to_write:
                row["runtime_s"] = runtime
                writer.writerow(row)
            handle.flush()

            if index % 10 == 0 or index == len(query_frames):
                print(
                    f"[subject {subject}] {index}/{len(query_frames)} frame={frame_num} "
                    f"status={rows_to_write[0]['status']} runtime={runtime:.2f}s",
                    flush=True,
                )
            if index % 50 == 0:
                torch.cuda.empty_cache()

    final_rows = read_csv_rows(predictions_path)
    summaries = [
        summarize_rows(subject, final_rows, len(query_frames), mode)
        for mode in intrinsics_modes
    ]
    write_csv_atomic(output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    print(f"[subject {subject}] summaries: {summaries}", flush=True)


def aggregate_results(args):
    all_predictions = []
    summaries = []
    intrinsics_modes = ("calibrated", "fixed_fov30")
    for subject_id in args.subjects:
        subject = f"{subject_id:02d}"
        subject_dir = Path(args.biwi_dir) / subject
        expected = max(0, len(get_subject_frames(subject_dir)) - 1)
        rows = read_csv_rows(Path(args.out_dir) / f"subject_{subject}" / "predictions.csv")
        all_predictions.extend(rows)
        for mode in intrinsics_modes:
            summaries.append(summarize_rows(subject, rows, expected, mode))

    expected_total = sum(
        max(0, len(get_subject_frames(Path(args.biwi_dir) / f"{subject_id:02d}")) - 1)
        for subject_id in args.subjects
    )
    for mode in intrinsics_modes:
        summaries.append(summarize_rows("overall", all_predictions, expected_total, mode))
    write_csv_atomic(Path(args.out_dir) / "biwi_flame_fit_predictions.csv", PREDICTION_FIELDS, all_predictions)
    write_csv_atomic(Path(args.out_dir) / "biwi_flame_fit_results.csv", SUMMARY_FIELDS, summaries)
    print(f"[aggregate] {summaries[-2:]}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--biwi_dir",
        default="data/biwi/faces_0",
    )
    parser.add_argument(
        "--blended_root",
        default="blended_in_flames",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/eval_flame_fitting_biwi",
    )
    parser.add_argument("--subject", type=int)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 25)))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--landmark_device", default="cuda")
    parser.add_argument("--rigid_iters", type=int, default=250)
    parser.add_argument("--full_iters", type=int, default=400)
    parser.add_argument("--n_shape_fit", type=int, default=150)
    parser.add_argument("--fixed_fov_deg", type=float, default=30.0)
    args = parser.parse_args()
    if not args.aggregate and args.subject is None:
        parser.error("--subject is required unless --aggregate is used")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.aggregate:
        aggregate_results(arguments)
    else:
        evaluate_subject(arguments)
