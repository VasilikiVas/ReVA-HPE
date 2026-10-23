import argparse
import csv
import gc
import glob
import json
import logging
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
for _path in (_HERE / "eval_scripts", _HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_biwi_crop as base
import eval_biwi_crop_mtcnn_square as square_crop
import eval_biwi_crop_mtcnn_square_translation as vggt_translation
from prepare_biwi_for_sixdof_face import prepare_biwi_for_sixdof_face
from reva.avatar_pipeline.evaluate_enrollment_flame_pnp import solve_pose


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_flame_pnp_anchor_pose(args, subj, anchor_rgb_path, intrinsics, rgb_rotation, rgb_translation):
    alignment_root = Path(args.anchor_pnp_alignment_root)
    similarity_path = (
        alignment_root / f"subject{int(subj):02d}" / "similarity" /
        "facelift_to_flame_similarity.json"
    )
    with similarity_path.open() as handle:
        similarity = json.load(handle)
    fit_path = Path(similarity["source_flame_fit"])
    with fit_path.open() as handle:
        fit = json.load(handle)
    source_image = Path(fit["source_image"])
    if source_image.name != Path(anchor_rgb_path).name:
        raise ValueError(
            f"Subject {subj}: PnP fit image {source_image.name} does not match "
            f"anchor {Path(anchor_rgb_path).name}"
        )
    object_points = np.asarray(
        fit["canonical_references"][args.anchor_pnp_reference][
            "landmarks_xyz_opencv"
        ],
        dtype=np.float64,
    )
    image_points = np.asarray(
        fit["fit"]["detected_landmarks_xy"], dtype=np.float64
    )
    result = solve_pose(
        object_points,
        image_points,
        intrinsics,
        args.anchor_pnp_reprojection_threshold_px,
    )
    if result is None:
        raise RuntimeError(f"Subject {subj}: FLAME-PnP anchor pose failed")
    rotation_rgb, translation_rgb_m, inlier_ids, residuals = result
    rotation_depth = rgb_rotation.T @ rotation_rgb
    translation_depth = rgb_rotation.T @ (
        translation_rgb_m * 1000.0 - rgb_translation
    )
    logging.info(
        "Subject %s PnP anchor: ref=%s inliers=%d median=%.3fpx "
        "t=[%.2f, %.2f, %.2f]mm",
        subj,
        args.anchor_pnp_reference,
        len(inlier_ids),
        float(np.median(residuals)),
        *translation_depth,
    )
    return rotation_depth, translation_depth


def _empty_sixdof_acc():
    return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "l2": 0.0, "count": 0}


TRANSLATION_PREDICTION_FIELDS = [
    "subject",
    "anchor_frame",
    "frame",
    "pred_tx",
    "pred_ty",
    "pred_tz",
    "gt_tx",
    "gt_ty",
    "gt_tz",
    "err_tx",
    "err_ty",
    "err_tz",
    "err_l2",
    "pred_x_over_z",
    "pred_y_over_z",
    "pred_log_z",
    "gt_x_over_z",
    "gt_y_over_z",
    "gt_log_z",
    "err_x_over_z",
    "err_y_over_z",
    "err_log_z",
    "norm_l2",
    "anchor_tx",
    "anchor_ty",
    "anchor_tz",
    "pred_delta_x_over_z",
    "pred_delta_y_over_z",
    "pred_log_z_ratio",
    "gt_delta_x_over_z",
    "gt_delta_y_over_z",
    "gt_log_z_ratio",
    "err_delta_x_over_z",
    "err_delta_y_over_z",
    "err_log_z_ratio",
    "relproj_l2",
    "anchor_crop_w_norm",
    "anchor_crop_h_norm",
    "target_crop_w_norm",
    "target_crop_h_norm",
    "anchor_fx_eff",
    "anchor_fy_eff",
    "anchor_f_eff",
    "target_fx_eff",
    "target_fy_eff",
    "target_f_eff",
]

TRANSLATION_SUMMARY_METRICS = [
    ("tx_mm", "pred_tx", "gt_tx"),
    ("ty_mm", "pred_ty", "gt_ty"),
    ("tz_mm", "pred_tz", "gt_tz"),
    ("x_over_z", "pred_x_over_z", "gt_x_over_z"),
    ("y_over_z", "pred_y_over_z", "gt_y_over_z"),
    ("log_z", "pred_log_z", "gt_log_z"),
    ("delta_x_over_z", "pred_delta_x_over_z", "gt_delta_x_over_z"),
    ("delta_y_over_z", "pred_delta_y_over_z", "gt_delta_y_over_z"),
    ("log_z_ratio", "pred_log_z_ratio", "gt_log_z_ratio"),
]

POSE_PREDICTION_FIELDS = [
    "subject",
    "anchor_frame",
    "frame",
    "pred_tx_mm",
    "pred_ty_mm",
    "pred_tz_mm",
    "gt_tx_mm",
    "gt_ty_mm",
    "gt_tz_mm",
    "tx_error_mm",
    "ty_error_mm",
    "tz_error_mm",
    "translation_l2_error_mm",
    "yaw_error_deg",
    "pitch_error_deg",
    "roll_error_deg",
    "rot_mean_error_deg",
    "pred_r00",
    "pred_r01",
    "pred_r02",
    "pred_r10",
    "pred_r11",
    "pred_r12",
    "pred_r20",
    "pred_r21",
    "pred_r22",
    "gt_r00",
    "gt_r01",
    "gt_r02",
    "gt_r10",
    "gt_r11",
    "gt_r12",
    "gt_r20",
    "gt_r21",
    "gt_r22",
]


def _effective_focal_from_crop_params(K_rgb, image_shape_hw, crop_params, crop_schema, model_input_size=518):
    if K_rgb is None or image_shape_hw is None or crop_params is None:
        n = 0 if crop_params is None else crop_params.shape[0]
        nan = torch.full((n,), float("nan"), dtype=torch.float32)
        return nan, nan, nan

    crop = crop_params.detach().cpu().float()
    img_h, img_w = float(image_shape_hw[0]), float(image_shape_hw[1])

    if crop_schema == "full_frame_square":
        side_x = torch.full((crop.shape[0],), max(img_w, img_h), dtype=torch.float32)
        side_y = side_x.clone()
    else:
        side_x = (crop[:, 6] * img_w).clamp(min=1.0)
        side_y = (crop[:, 7] * img_h).clamp(min=1.0)

    fx = float(K_rgb[0, 0]) * float(model_input_size) / side_x
    fy = float(K_rgb[1, 1]) * float(model_input_size) / side_y
    f = 0.5 * (fx + fy)
    return fx, fy, f


def _append_translation_prediction_rows(
    rows,
    subj,
    anchor_frame,
    frame_nums,
    pred_t,
    gt_t,
    anchor_t=None,
    K_rgb=None,
    image_shape_hw=None,
    anchor_crop_params=None,
    target_crop_params=None,
    crop_schema="mtcnn_square_per_view",
):
    pred = pred_t.detach().cpu().float()
    gt = gt_t.detach().cpu().float()
    eps = 1e-6
    pred_z = pred[:, 2].clamp(min=eps)
    gt_z = gt[:, 2].clamp(min=eps)

    if anchor_t is None:
        anchor = torch.full_like(gt, float("nan"))
        anchor_z = torch.full_like(gt_z, float("nan"))
    else:
        anchor = anchor_t.detach().cpu().float().unsqueeze(0).expand_as(gt)
        anchor_z = anchor[:, 2].clamp(min=eps)

    pred_norm = torch.stack([pred[:, 0] / pred_z, pred[:, 1] / pred_z, torch.log(pred_z)], dim=-1)
    gt_norm = torch.stack([gt[:, 0] / gt_z, gt[:, 1] / gt_z, torch.log(gt_z)], dim=-1)
    norm_err = pred_norm - gt_norm

    pred_rel = torch.stack(
        [
            pred[:, 0] / pred_z - anchor[:, 0] / anchor_z,
            pred[:, 1] / pred_z - anchor[:, 1] / anchor_z,
            torch.log(pred_z / anchor_z),
        ],
        dim=-1,
    )
    gt_rel = torch.stack(
        [
            gt[:, 0] / gt_z - anchor[:, 0] / anchor_z,
            gt[:, 1] / gt_z - anchor[:, 1] / anchor_z,
            torch.log(gt_z / anchor_z),
        ],
        dim=-1,
    )
    rel_err = pred_rel - gt_rel

    err = pred - gt
    n = pred.shape[0]
    nan = torch.full((n,), float("nan"), dtype=torch.float32)
    if anchor_crop_params is None:
        anchor_crop_w = anchor_crop_h = nan
        anchor_fx_eff = anchor_fy_eff = anchor_f_eff = nan
    else:
        anchor_crop_cpu = anchor_crop_params.detach().cpu().float()
        anchor_crop_w = anchor_crop_cpu[:, 6]
        anchor_crop_h = anchor_crop_cpu[:, 7]
        anchor_fx_eff, anchor_fy_eff, anchor_f_eff = _effective_focal_from_crop_params(
            K_rgb,
            image_shape_hw,
            anchor_crop_cpu,
            crop_schema,
        )

    if target_crop_params is None:
        target_crop_w = target_crop_h = nan
        target_fx_eff = target_fy_eff = target_f_eff = nan
    else:
        target_crop_cpu = target_crop_params.detach().cpu().float()
        target_crop_w = target_crop_cpu[:, 6]
        target_crop_h = target_crop_cpu[:, 7]
        target_fx_eff, target_fy_eff, target_f_eff = _effective_focal_from_crop_params(
            K_rgb,
            image_shape_hw,
            target_crop_cpu,
            crop_schema,
        )

    for idx, frame_num in enumerate(frame_nums):
        rows.append(
            {
                "subject": subj,
                "anchor_frame": anchor_frame,
                "frame": frame_num,
                "pred_tx": pred[idx, 0].item(),
                "pred_ty": pred[idx, 1].item(),
                "pred_tz": pred[idx, 2].item(),
                "gt_tx": gt[idx, 0].item(),
                "gt_ty": gt[idx, 1].item(),
                "gt_tz": gt[idx, 2].item(),
                "err_tx": err[idx, 0].item(),
                "err_ty": err[idx, 1].item(),
                "err_tz": err[idx, 2].item(),
                "err_l2": err[idx].norm().item(),
                "pred_x_over_z": pred_norm[idx, 0].item(),
                "pred_y_over_z": pred_norm[idx, 1].item(),
                "pred_log_z": pred_norm[idx, 2].item(),
                "gt_x_over_z": gt_norm[idx, 0].item(),
                "gt_y_over_z": gt_norm[idx, 1].item(),
                "gt_log_z": gt_norm[idx, 2].item(),
                "err_x_over_z": norm_err[idx, 0].item(),
                "err_y_over_z": norm_err[idx, 1].item(),
                "err_log_z": norm_err[idx, 2].item(),
                "norm_l2": norm_err[idx].norm().item(),
                "anchor_tx": anchor[idx, 0].item(),
                "anchor_ty": anchor[idx, 1].item(),
                "anchor_tz": anchor[idx, 2].item(),
                "pred_delta_x_over_z": pred_rel[idx, 0].item(),
                "pred_delta_y_over_z": pred_rel[idx, 1].item(),
                "pred_log_z_ratio": pred_rel[idx, 2].item(),
                "gt_delta_x_over_z": gt_rel[idx, 0].item(),
                "gt_delta_y_over_z": gt_rel[idx, 1].item(),
                "gt_log_z_ratio": gt_rel[idx, 2].item(),
                "err_delta_x_over_z": rel_err[idx, 0].item(),
                "err_delta_y_over_z": rel_err[idx, 1].item(),
                "err_log_z_ratio": rel_err[idx, 2].item(),
                "relproj_l2": rel_err[idx].norm().item(),
                "anchor_crop_w_norm": anchor_crop_w[idx].item(),
                "anchor_crop_h_norm": anchor_crop_h[idx].item(),
                "target_crop_w_norm": target_crop_w[idx].item(),
                "target_crop_h_norm": target_crop_h[idx].item(),
                "anchor_fx_eff": anchor_fx_eff[idx].item(),
                "anchor_fy_eff": anchor_fy_eff[idx].item(),
                "anchor_f_eff": anchor_f_eff[idx].item(),
                "target_fx_eff": target_fx_eff[idx].item(),
                "target_fy_eff": target_fy_eff[idx].item(),
                "target_f_eff": target_f_eff[idx].item(),
            }
        )


def _append_pose_prediction_rows(
    rows,
    subj,
    anchor_frame,
    frame_nums,
    pred_rot,
    gt_rot,
    pred_t,
    gt_t,
    yaw_error,
    pitch_error,
    roll_error,
):
    pred_r = pred_rot.detach().cpu().float()
    gt_r = gt_rot.detach().cpu().float()
    pred = pred_t.detach().cpu().float()
    gt = gt_t.detach().cpu().float()
    err = pred - gt
    yaw = yaw_error.detach().cpu().float()
    pitch = pitch_error.detach().cpu().float()
    roll = roll_error.detach().cpu().float()
    for idx, frame_num in enumerate(frame_nums):
        row = {
            "subject": subj,
            "anchor_frame": anchor_frame,
            "frame": frame_num,
            "pred_tx_mm": pred[idx, 0].item(),
            "pred_ty_mm": pred[idx, 1].item(),
            "pred_tz_mm": pred[idx, 2].item(),
            "gt_tx_mm": gt[idx, 0].item(),
            "gt_ty_mm": gt[idx, 1].item(),
            "gt_tz_mm": gt[idx, 2].item(),
            "tx_error_mm": abs(err[idx, 0].item()),
            "ty_error_mm": abs(err[idx, 1].item()),
            "tz_error_mm": abs(err[idx, 2].item()),
            "translation_l2_error_mm": err[idx].norm().item(),
            "yaw_error_deg": yaw[idx].item(),
            "pitch_error_deg": pitch[idx].item(),
            "roll_error_deg": roll[idx].item(),
            "rot_mean_error_deg": ((yaw[idx] + pitch[idx] + roll[idx]) / 3.0).item(),
        }
        for r in range(3):
            for c in range(3):
                row[f"pred_r{r}{c}"] = pred_r[idx, r, c].item()
                row[f"gt_r{r}{c}"] = gt_r[idx, r, c].item()
        rows.append(row)


def _finite_pair_arrays(rows, pred_key, gt_key):
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    gt = np.asarray([float(row[gt_key]) for row in rows], dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt)
    return pred[valid], gt[valid]


def _prediction_summary_row(scope, subject, metric, pred, gt):
    count = int(pred.size)
    if count == 0:
        return {
            "scope": scope,
            "subject": subject,
            "metric": metric,
            "count": 0,
            "mae": "",
            "rmse": "",
            "bias_pred_minus_gt": "",
            "pred_mean": "",
            "gt_mean": "",
            "pearson": "",
            "slope_pred_vs_gt": "",
            "intercept_pred_vs_gt": "",
            "gt_std": "",
            "pred_std": "",
        }

    diff = pred - gt
    gt_std = float(gt.std())
    pred_std = float(pred.std())
    pearson = float(np.corrcoef(pred, gt)[0, 1]) if count > 1 and pred_std > 1e-12 and gt_std > 1e-12 else float("nan")
    if count > 1 and gt_std > 1e-12:
        slope, intercept = np.polyfit(gt, pred, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = float("nan")
        intercept = float("nan")

    return {
        "scope": scope,
        "subject": subject,
        "metric": metric,
        "count": count,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias_pred_minus_gt": float(diff.mean()),
        "pred_mean": float(pred.mean()),
        "gt_mean": float(gt.mean()),
        "pearson": pearson,
        "slope_pred_vs_gt": slope,
        "intercept_pred_vs_gt": intercept,
        "gt_std": gt_std,
        "pred_std": pred_std,
    }


def _summarize_translation_predictions(rows):
    summary = []
    subjects = sorted({str(row["subject"]) for row in rows})
    groups = [("overall", "all", rows)]
    groups.extend(("subject", subj, [row for row in rows if str(row["subject"]) == subj]) for subj in subjects)

    for scope, subject, group_rows in groups:
        for metric, pred_key, gt_key in TRANSLATION_SUMMARY_METRICS:
            pred, gt = _finite_pair_arrays(group_rows, pred_key, gt_key)
            summary.append(_prediction_summary_row(scope, subject, metric, pred, gt))
    return summary


def _parse_sixdof_subject(img_path):
    path = Path(img_path)
    if len(path.parts) < 3:
        return "unknown"
    return path.parts[-3]


def _resolve_sixdof_face_data_root(args):
    if args.sixdof_face_data_root:
        return args.sixdof_face_data_root
    if args.sixdof_face_cache_root:
        return args.sixdof_face_cache_root
    return os.path.join(args.sixdof_face_repo, "dataset", "ARKitFace")


def _sixdof_face_root_is_ready(path):
    path = Path(path)
    return (
        (path / "csv" / "metadata_biwi.csv").is_file()
        and (path / "image").is_dir()
        and (path / "info").is_dir()
    )


def _prepare_sixdof_face_data_root_if_needed(args):
    data_root = _resolve_sixdof_face_data_root(args)
    if _sixdof_face_root_is_ready(data_root):
        return data_root

    if args.no_sixdof_face:
        return data_root

    if not args.prepare_sixdof_face_if_missing:
        return data_root

    logging.info(
        "6DoF Face BIWI evaluation root is missing, preparing it from raw BIWI data at %s",
        args.biwi_dir,
    )
    subjects = [str(s).zfill(2) for s in args.test_subjects] if args.test_subjects else None
    prepare_biwi_for_sixdof_face(
        biwi_root=args.biwi_dir,
        output_root=data_root,
        subjects=subjects,
        overwrite=False,
    )
    return data_root


def _validate_sixdof_face_paths(args):
    if args.no_sixdof_face:
        return

    if not os.path.isdir(args.sixdof_face_repo):
        raise FileNotFoundError(f"6DoF Face repo not found: {args.sixdof_face_repo}")

    checkpoints_dir = args.sixdof_face_checkpoints_dir or os.path.join(args.sixdof_face_repo, "checkpoint")
    ckpt_path = os.path.join(checkpoints_dir, args.sixdof_face_name, f"{args.sixdof_face_epoch}_net_R.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"6DoF Face checkpoint not found: {ckpt_path}")

    data_root = _prepare_sixdof_face_data_root_if_needed(args)
    if not _sixdof_face_root_is_ready(data_root):
        raise FileNotFoundError(
            f"6DoF Face BIWI evaluation root not found: {data_root}\n"
            f"Pass --sixdof_face_data_root to the preprocessed BIWI eval root used by 6DoF Face "
            f"(it should contain csv/, image/, and info/), or enable --prepare_sixdof_face_if_missing."
        )


def _load_pair_subset_csv(pair_csv):
    pair_subset = defaultdict(list)
    with open(pair_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "subject",
            "anchor_frame_num",
            "anchor_pose_file",
            "query_frame_num",
            "query_pose_file",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Pair CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            subj = str(int(row["subject"])).zfill(2)
            pair_subset[subj].append(
                {
                    "anchor_frame_num": int(row["anchor_frame_num"]),
                    "anchor_pose_file": row["anchor_pose_file"],
                    "query_frame_num": int(row["query_frame_num"]),
                    "query_pose_file": row["query_pose_file"],
                }
            )
    return pair_subset


def _pair_csv_uses_first_frame_anchors(pair_csv, biwi_dir, test_subjects):
    selected = {str(subject).zfill(2) for subject in test_subjects} if test_subjects else None
    first_anchor_by_subject = {}
    seen = set()
    with open(pair_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj = str(int(row["subject"])).zfill(2)
            if selected is not None and subj not in selected:
                continue
            seen.add(subj)
            if subj not in first_anchor_by_subject:
                frames = base.get_subject_frames(os.path.join(biwi_dir, subj))
                if not frames:
                    raise RuntimeError(f"Subject {subj}: no BIWI frames found")
                first_frame_num, _, first_pose_path = frames[0]
                first_anchor_by_subject[subj] = (
                    int(first_frame_num),
                    os.path.basename(first_pose_path),
                )
            expected_frame_num, expected_pose_file = first_anchor_by_subject[subj]
            if int(row["anchor_frame_num"]) != expected_frame_num:
                return False
            if row["anchor_pose_file"] != expected_pose_file:
                return False
    return bool(seen)


def _select_evenly_spaced_frames(frames, count):
    if count <= 1 or count >= len(frames):
        return list(frames[: max(1, min(count, len(frames)))])
    idxs = np.linspace(0, len(frames) - 1, num=count)
    selected = []
    seen = set()
    for idx in idxs:
        frame = frames[int(round(idx))]
        if frame[0] in seen:
            continue
        selected.append(frame)
        seen.add(frame[0])
    for frame in frames:
        if len(selected) >= count:
            break
        if frame[0] not in seen:
            selected.append(frame)
            seen.add(frame[0])
    return selected


def _resolve_subject_frame(frame_by_num, frame_by_pose, frame_num, pose_file, role, subj):
    frame = frame_by_num.get(frame_num)
    if frame is not None and os.path.basename(frame[2]) == pose_file:
        return frame

    frame = frame_by_pose.get(pose_file)
    if frame is not None and frame[0] == frame_num:
        return frame

    raise FileNotFoundError(
        f"Subject {subj}: could not resolve {role} frame_num={frame_num}, pose_file={pose_file}"
    )


def _union_mtcnn_boxes(box_a, box_b):
    """Return the union box in the local [x1, x2, y1, y2] convention."""
    return [
        min(float(box_a[0]), float(box_b[0])),
        max(float(box_a[1]), float(box_b[1])),
        min(float(box_a[2]), float(box_b[2])),
        max(float(box_a[3]), float(box_b[3])),
    ]


def _full_frame_square_rgb(img_bgr, crop_size):
    """Pad the full frame to a square before VGGT resizing, this avoids a face detector crop."""
    img_h, img_w = img_bgr.shape[:2]
    side = max(img_h, img_w)
    canvas = np.zeros((side, side, 3), dtype=img_bgr.dtype)
    y0 = (side - img_h) // 2
    x0 = (side - img_w) // 2
    canvas[y0 : y0 + img_h, x0 : x0 + img_w] = img_bgr
    crop_bgr = cv2.resize(canvas, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _square_box_rgb(img_bgr, box, crop_size):
    crop_bgr = square_crop.square_crop_from_box(img_bgr, box, crop_size=crop_size)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def _identity_crop_params(img_bgr):
    return torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0], dtype=torch.float32)


def _square_box_crop_params(img_bgr, box_xyxy):
    img_h, img_w = img_bgr.shape[:2]
    x1, x2, y1, y2 = [int(v) for v in box_xyxy]

    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    side = max(box_w, box_h)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    sq_x1 = int(round(cx - side / 2))
    sq_y1 = int(round(cy - side / 2))
    sq_x2 = sq_x1 + side
    sq_y2 = sq_y1 + side

    x0 = float(sq_x1) / max(float(img_w), 1.0)
    y0 = float(sq_y1) / max(float(img_h), 1.0)
    x1n = float(sq_x2) / max(float(img_w), 1.0)
    y1n = float(sq_y2) / max(float(img_h), 1.0)
    w = x1n - x0
    h = y1n - y0
    return torch.tensor([x0, y0, x1n, y1n, x0 + 0.5 * w, y0 + 0.5 * h, w, h], dtype=torch.float32)


def _build_subject_eval_groups(subj, frames, args, pair_subset):
    use_anchor_pair = args.vggt_pose_mode in {"relative_h2c", "anchor_abs_relative_h2c", "absolute_second"}
    if not pair_subset:
        if use_anchor_pair:
            multi_anchor_count = max(1, int(getattr(args, "multi_anchor_count", 1)))
            if multi_anchor_count > 1:
                anchors = _select_evenly_spaced_frames(frames, multi_anchor_count)
                groups = []
                for anchor in anchors:
                    groups.append(
                        {
                            "anchor": anchor,
                            "targets": [frame for frame in frames if frame[0] != anchor[0]],
                        }
                    )
                return groups
            return [{"anchor": frames[0], "targets": list(frames[1:])}]
        return [{"anchor": None, "targets": list(frames)}]

    frame_by_num = {frame_num: (frame_num, rgb_path, pose_path) for frame_num, rgb_path, pose_path in frames}
    frame_by_pose = {
        os.path.basename(pose_path): (frame_num, rgb_path, pose_path)
        for frame_num, rgb_path, pose_path in frames
    }

    if use_anchor_pair:
        grouped = defaultdict(list)
        for row in pair_subset:
            anchor_frame = _resolve_subject_frame(
                frame_by_num,
                frame_by_pose,
                row["anchor_frame_num"],
                row["anchor_pose_file"],
                role="anchor",
                subj=subj,
            )
            query_frame = _resolve_subject_frame(
                frame_by_num,
                frame_by_pose,
                row["query_frame_num"],
                row["query_pose_file"],
                role="query",
                subj=subj,
            )
            grouped[anchor_frame].append(query_frame)

        groups = []
        for anchor_frame, target_frames in sorted(grouped.items(), key=lambda item: item[0][0]):
            groups.append(
                {
                    "anchor": anchor_frame,
                    "targets": sorted(target_frames, key=lambda item: item[0]),
                }
            )
        return groups

    target_frames = [
        _resolve_subject_frame(
            frame_by_num,
            frame_by_pose,
            row["query_frame_num"],
            row["query_pose_file"],
            role="query",
            subj=subj,
        )
        for row in pair_subset
    ]
    return [{"anchor": None, "targets": sorted(target_frames, key=lambda item: item[0])}]


def _rotation_geodesic_deg_np(R_a, R_b):
    rel = R_a @ np.swapaxes(R_b, -1, -2)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _rotation_medoid_index(Rs):
    n = Rs.shape[0]
    if n <= 1:
        return 0
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dists[i] = _rotation_geodesic_deg_np(Rs[i : i + 1], Rs)
    return int(np.argmin(np.median(dists, axis=1)))


def _fuse_multi_anchor_predictions(records, fusion):
    Rs = np.stack([row["R_pred"] for row in records]).astype(np.float64)
    ts = np.stack([row["t_pred"] for row in records]).astype(np.float64)

    if fusion == "trimmed_mean" and len(records) >= 3:
        median_t = np.median(ts, axis=0)
        keep_count = max(2, len(records) - 1)
        keep = np.argsort(np.linalg.norm(ts - median_t[None, :], axis=1))[:keep_count]
        Rs_keep = Rs[keep]
        ts_keep = ts[keep]
        medoid_local = _rotation_medoid_index(Rs_keep)
        medoid_idx = int(keep[medoid_local])
        return {
            "R_pred": Rs_keep[medoid_local].astype(np.float32),
            "t_pred": ts_keep.mean(axis=0).astype(np.float32),
            "selected_anchor": records[medoid_idx]["anchor_frame"],
            "kept_anchors": [records[int(i)]["anchor_frame"] for i in keep],
        }

    medoid_idx = _rotation_medoid_index(Rs)
    return {
        "R_pred": Rs[medoid_idx].astype(np.float32),
        "t_pred": ts[medoid_idx].astype(np.float32),
        "selected_anchor": records[medoid_idx]["anchor_frame"],
        "kept_anchors": [row["anchor_frame"] for row in records],
    }


def _write_multi_anchor_results(args, multi_anchor_records):
    fused_rows = []
    for (subj, frame_num), records in sorted(multi_anchor_records.items()):
        if len(records) < args.multi_anchor_min_predictions:
            continue
        fused = _fuse_multi_anchor_predictions(records, args.multi_anchor_fusion)
        first = records[0]
        fused_rows.append(
            {
                "subject": subj,
                "frame": frame_num,
                "n_predictions": len(records),
                "anchor_frames": ",".join(str(row["anchor_frame"]) for row in records),
                "kept_anchor_frames": ",".join(str(v) for v in fused["kept_anchors"]),
                "selected_anchor_frame": fused["selected_anchor"],
                "R_pred": fused["R_pred"],
                "t_pred": fused["t_pred"],
                "R_gt": first["R_gt"],
                "t_gt": first["t_gt"],
            }
        )

    summary_path = os.path.join(args.vis_dir, "biwi_multi_anchor_results.csv")
    pred_path = os.path.join(args.vis_dir, "biwi_multi_anchor_predictions.csv")

    if not fused_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["message"])
            w.writerow(["No query had enough repeated anchor predictions for fusion."])
        return

    pred_R = torch.from_numpy(np.stack([row["R_pred"] for row in fused_rows])).float()
    gt_R = torch.from_numpy(np.stack([row["R_gt"] for row in fused_rows])).float()
    pred_t = torch.from_numpy(np.stack([row["t_pred"] for row in fused_rows])).float()
    gt_t = torch.from_numpy(np.stack([row["t_gt"] for row in fused_rows])).float()

    y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(gt_R)
    y_pred, p_pred, r_pred = base.biwi_euler_deg_batch(pred_R)
    err_y = base.angular_error(y_gt, y_pred)
    err_p = base.angular_error(p_gt, p_pred)
    err_r = base.angular_error(r_gt, r_pred)
    t_err = pred_t - gt_t
    l2_err = torch.linalg.norm(t_err, dim=-1)

    for idx, row in enumerate(fused_rows):
        row["yaw_err"] = float(err_y[idx].item())
        row["pitch_err"] = float(err_p[idx].item())
        row["roll_err"] = float(err_r[idx].item())
        row["mean_err"] = float((err_y[idx] + err_p[idx] + err_r[idx]).item() / 3.0)
        row["tx_err"] = float(abs(t_err[idx, 0].item()))
        row["ty_err"] = float(abs(t_err[idx, 1].item()))
        row["tz_err"] = float(abs(t_err[idx, 2].item()))
        row["l2_err"] = float(l2_err[idx].item())

    with open(pred_path, "w", newline="") as f:
        fieldnames = [
            "subject",
            "frame",
            "n_predictions",
            "anchor_frames",
            "kept_anchor_frames",
            "selected_anchor_frame",
            "yaw_err",
            "pitch_err",
            "roll_err",
            "mean_err",
            "tx_err",
            "ty_err",
            "tz_err",
            "l2_err",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in fused_rows:
            w.writerow({field: row[field] for field in fieldnames})

    header = [
        "subject",
        "model",
        "variant",
        "fusion",
        "queries",
        "mean_predictions_per_query",
        "yaw_mae",
        "pitch_mae",
        "roll_mae",
        "mean_mae",
        "tx_mae_mm",
        "ty_mae_mm",
        "tz_mae_mm",
        "translation_l2_mae_mm",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        subjects = sorted({row["subject"] for row in fused_rows})
        groups = [("overall", fused_rows)]
        groups.extend((subj, [row for row in fused_rows if row["subject"] == subj]) for subj in subjects)
        for subj, rows in groups:
            n = len(rows)
            w.writerow(
                [
                    subj,
                    f"vggt_multi_anchor_{args.multi_anchor_variant}",
                    args.multi_anchor_variant,
                    args.multi_anchor_fusion,
                    n,
                    f"{np.mean([row['n_predictions'] for row in rows]):.4f}",
                    f"{np.mean([row['yaw_err'] for row in rows]):.4f}",
                    f"{np.mean([row['pitch_err'] for row in rows]):.4f}",
                    f"{np.mean([row['roll_err'] for row in rows]):.4f}",
                    f"{np.mean([row['mean_err'] for row in rows]):.4f}",
                    f"{np.mean([row['tx_err'] for row in rows]):.4f}",
                    f"{np.mean([row['ty_err'] for row in rows]):.4f}",
                    f"{np.mean([row['tz_err'] for row in rows]):.4f}",
                    f"{np.mean([row['l2_err'] for row in rows]):.4f}",
                ]
            )


def evaluate_sixdof_face(args):
    """
    Run the native 6DoF Face BIWI evaluation pipeline and return overall/per-subject metrics.

    The 6DoF code reports translation errors in meters internally, so we convert tx/ty/tz/L2 to mm.
    """
    _validate_sixdof_face_paths(args)
    if args.no_sixdof_face:
        return None, {}

    repo_root = os.path.abspath(args.sixdof_face_repo)
    data_root = os.path.abspath(_prepare_sixdof_face_data_root_if_needed(args))
    checkpoints_dir = os.path.abspath(
        args.sixdof_face_checkpoints_dir or os.path.join(args.sixdof_face_repo, "checkpoint")
    )

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    try:
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from options.test_options import TestOptions
        from models import create_model
        if "data.augmentation" not in sys.modules:
            augmentation_stub = types.ModuleType("data.augmentation")

            class _UnusedAugmentor:
                def __init__(self, *args, **kwargs):
                    pass

            augmentation_stub.EulerAugmentor = _UnusedAugmentor
            augmentation_stub.HorizontalFlipAugmentor = _UnusedAugmentor
            sys.modules["data.augmentation"] = augmentation_stub
        import models.model_repository as model_repository_module

        orig_resnet18 = model_repository_module.resnet18

        def _offline_resnet18(*args, **kwargs):
            kwargs["pretrained"] = False
            return orig_resnet18(*args, **kwargs)

        model_repository_module.resnet18 = _offline_resnet18
        import data.biwi_dataset as biwi_dataset_module

        biwi_dataset_module.data_root = Path(data_root)
        BIWIDataset = biwi_dataset_module.BIWIDataset

        gpu_ids = str(args.gpu if torch.cuda.is_available() else -1)
        cmd_line = " ".join(
            [
                "--dataset_mode", "biwi",
                "--model", args.sixdof_face_model,
                "--img_size", str(args.sixdof_face_img_size),
                "--batch_size", str(args.sixdof_face_batch_size),
                "--num_threads", str(args.sixdof_face_num_workers),
                "--checkpoints_dir", checkpoints_dir,
                "--name", args.sixdof_face_name,
                "--epoch", args.sixdof_face_epoch,
                "--gpu_ids", gpu_ids,
                "--use_gt_bbox",
            ]
        )

        opt = TestOptions(cmd_line=cmd_line).parse()
        dataset = BIWIDataset(opt)
        if args.test_subjects:
            subject_ids = set(int(s) for s in args.test_subjects)
            dataset.df = dataset.df[dataset.df["subject_id"].astype(int).isin(subject_ids)].reset_index(drop=True)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
            drop_last=False,
        )

        model = create_model(opt)
        per_subject_acc = defaultdict(_empty_sixdof_acc)
        did_init = False

        for batch_idx, data in enumerate(dataloader):
            if batch_idx % 20 == 0:
                logging.info(f"[6DoF Face] batch {batch_idx + 1}/{len(dataloader)}")

            if not did_init:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                model.eval()
                model.init_evaluation()
                did_init = True

            model.set_input(data)
            start_idx = len(model.inference_data["yaw_mae"])
            try:
                model.inference_curr_batch()
            except cv2.error as e:
                logging.warning(f"[6DoF Face] batch {batch_idx + 1}: OpenCV error, skipping batch — {e}")
                continue
            end_idx = len(model.inference_data["yaw_mae"])
            new_count = end_idx - start_idx

            batch_paths = list(model.img_path_list)[:new_count]
            for local_idx in range(new_count):
                subj = _parse_sixdof_subject(batch_paths[local_idx])
                yaw = float(model.inference_data["yaw_mae"][start_idx + local_idx])
                pitch = float(model.inference_data["pitch_mae"][start_idx + local_idx])
                roll = float(model.inference_data["roll_mae"][start_idx + local_idx])
                tx_m = float(model.inference_data["tx_mae"][start_idx + local_idx])
                ty_m = float(model.inference_data["ty_mae"][start_idx + local_idx])
                tz_m = float(model.inference_data["tz_mae"][start_idx + local_idx])

                acc = per_subject_acc[subj]
                acc["yaw"] += yaw
                acc["pitch"] += pitch
                acc["roll"] += roll
                acc["tx"] += tx_m * 1000.0
                acc["ty"] += ty_m * 1000.0
                acc["tz"] += tz_m * 1000.0
                acc["l2"] += float(np.sqrt(tx_m ** 2 + ty_m ** 2 + tz_m ** 2) * 1000.0)
                acc["count"] += 1

        yaw_arr = np.asarray(model.inference_data["yaw_mae"], dtype=np.float64)
        pitch_arr = np.asarray(model.inference_data["pitch_mae"], dtype=np.float64)
        roll_arr = np.asarray(model.inference_data["roll_mae"], dtype=np.float64)
        tx_arr_m = np.asarray(model.inference_data["tx_mae"], dtype=np.float64)
        ty_arr_m = np.asarray(model.inference_data["ty_mae"], dtype=np.float64)
        tz_arr_m = np.asarray(model.inference_data["tz_mae"], dtype=np.float64)

        overall = {
            "yaw": float(yaw_arr.mean()) if yaw_arr.size else float("nan"),
            "pitch": float(pitch_arr.mean()) if pitch_arr.size else float("nan"),
            "roll": float(roll_arr.mean()) if roll_arr.size else float("nan"),
            "mean": float((yaw_arr.mean() + pitch_arr.mean() + roll_arr.mean()) / 3.0) if yaw_arr.size else float("nan"),
            "tx": float(tx_arr_m.mean() * 1000.0) if tx_arr_m.size else float("nan"),
            "ty": float(ty_arr_m.mean() * 1000.0) if ty_arr_m.size else float("nan"),
            "tz": float(tz_arr_m.mean() * 1000.0) if tz_arr_m.size else float("nan"),
            "l2": float(np.sqrt(tx_arr_m ** 2 + ty_arr_m ** 2 + tz_arr_m ** 2).mean() * 1000.0) if tx_arr_m.size else float("nan"),
            "count": int(yaw_arr.size),
        }

        per_subject = {}
        for subj, acc in per_subject_acc.items():
            count = acc["count"]
            if count == 0:
                continue
            per_subject[subj] = {
                "yaw": acc["yaw"] / count,
                "pitch": acc["pitch"] / count,
                "roll": acc["roll"] / count,
                "mean": (acc["yaw"] + acc["pitch"] + acc["roll"]) / (3 * count),
                "tx": acc["tx"] / count,
                "ty": acc["ty"] / count,
                "tz": acc["tz"] / count,
                "l2": acc["l2"] / count,
            }

        logging.info(
            "[6DoF Face] overall — "
            f"Yaw: {overall['yaw']:.4f}  Pitch: {overall['pitch']:.4f}  Roll: {overall['roll']:.4f}  "
            f"MAE: {overall['mean']:.4f}  Tx: {overall['tx']:.4f}  Ty: {overall['ty']:.4f}  "
            f"Tz: {overall['tz']:.4f}  L2: {overall['l2']:.4f}"
        )

        return overall, per_subject

    finally:
        os.chdir(old_cwd)
        sys.path = old_sys_path
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args):
    _validate_sixdof_face_paths(args)
    if args.anchor_pose_source == "flame_pnp":
        if not args.anchor_pnp_alignment_root:
            raise ValueError("--anchor_pnp_alignment_root is required for flame_pnp")
        if args.multi_anchor_count > 1:
            raise ValueError(
                "flame_pnp currently supports the fixed first-frame enrollment anchor only"
            )
        if args.pair_csv and not _pair_csv_uses_first_frame_anchors(
            args.pair_csv, args.biwi_dir, args.test_subjects
        ):
            raise ValueError(
                "flame_pnp pair_csv mode requires every anchor to be the fixed first-frame enrollment anchor"
            )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    variant_names = base.get_variant_names(args.vggt_pose_mode)
    primary_variant = base.get_primary_variant_name(args.vggt_pose_mode)
    if (
        args.vggt_crop_schema == "mtcnn_square_pair_union"
        and args.vggt_pose_mode not in {"relative_h2c", "anchor_abs_relative_h2c", "absolute_second"}
    ):
        raise ValueError("mtcnn_square_pair_union crop schema requires a paired VGGT pose mode.")

    test_subjects = set(args.test_subjects)
    logging.info(f"Device: {device}  |  AMP: {amp_dtype}")
    logging.info(f"Test subjects: {sorted(test_subjects)}")
    logging.info(f"VGGT pose mode: {args.vggt_pose_mode}")
    logging.info(f"VGGT pose encoding: {args.vggt_pose_encoding_type}")
    logging.info(f"VGGT crop schema: {args.vggt_crop_schema}")
    logging.info(f"VGGT crop positional encoding dim: {args.vggt_crop_posenc_dim}")
    logging.info("6DRepNet crop mode: rectangular MTCNN crop (matches training distortion)")
    logging.info(f"VGGT translation scale: x{args.vggt_translation_scale:.4f}")
    logging.info(f"VGGT translation target type: {args.vggt_translation_target_type}")
    logging.info(f"VGGT translation rotation mode: {args.vggt_translation_rotation_mode}")
    logging.info(f"VGGT translation diagnostics: {args.translation_diagnostics}")
    logging.info(f"VGGT translation prediction diagnostics: {args.translation_prediction_diagnostics}")
    logging.info(f"VGGT pose prediction diagnostics: {args.pose_prediction_diagnostics}")
    logging.info(f"VGGT multi-anchor eval: {args.multi_anchor_eval}")
    if args.multi_anchor_eval:
        logging.info(
            "VGGT multi-anchor settings: count=%d, variant=%s, fusion=%s, min_predictions=%d",
            args.multi_anchor_count,
            args.multi_anchor_variant,
            args.multi_anchor_fusion,
            args.multi_anchor_min_predictions,
        )
    logging.info("VGGT translation is decoded in BIWI pose.txt space (depth-camera frame, mm).")
    os.makedirs(args.vis_dir, exist_ok=True)
    with open(os.path.join(args.vis_dir, "eval_metadata.txt"), "w") as f:
        f.write(f"vggt_crop_schema: {args.vggt_crop_schema}\n")
        f.write(f"vggt_crop_posenc_dim: {args.vggt_crop_posenc_dim}\n")
        f.write(f"vggt_pose_mode: {args.vggt_pose_mode}\n")
        f.write(f"vggt_pose_encoding_type: {args.vggt_pose_encoding_type}\n")
        f.write(f"vggt_translation_target_type: {args.vggt_translation_target_type}\n")
        f.write(f"vggt_translation_rotation_mode: {args.vggt_translation_rotation_mode}\n")
        f.write(f"translation_diagnostics: {args.translation_diagnostics}\n")
        f.write(f"translation_prediction_diagnostics: {args.translation_prediction_diagnostics}\n")
        f.write(f"pose_prediction_diagnostics: {args.pose_prediction_diagnostics}\n")
        f.write(f"multi_anchor_eval: {args.multi_anchor_eval}\n")
        f.write(f"multi_anchor_count: {args.multi_anchor_count}\n")
        f.write(f"multi_anchor_variant: {args.multi_anchor_variant}\n")
        f.write(f"multi_anchor_fusion: {args.multi_anchor_fusion}\n")
        f.write(f"multi_anchor_min_predictions: {args.multi_anchor_min_predictions}\n")
        f.write(f"lora_checkpoint: {args.lora_checkpoint}\n")
        f.write(f"pair_csv: {args.pair_csv}\n")

    lora_path = None if args.no_lora else args.lora_checkpoint
    vggt_model = base.load_vggt_model(
        args.base_checkpoint,
        lora_path,
        device,
        enable_track=args.enable_track,
        pose_encoding_type=args.vggt_pose_encoding_type,
        crop_posenc_dim=args.vggt_crop_posenc_dim,
        enable_absolute_camera=args.vggt_pose_mode == "anchor_abs_relative_h2c",
    )

    sixd_model = None
    if not args.no_sixdrepnet:
        sixd_model = base.load_sixdrepnet_model(
            args.sixdrepnet_checkpoint,
            args.sixdrepnet_dir,
            device,
        )

    from facenet_pytorch import MTCNN as FacenetMTCNN

    mtcnn_detector = FacenetMTCNN(keep_all=True, device=device)
    logging.info("MTCNN detector initialized for shared VGGT/6DRepNet crops")

    mesh_cache, cal_cache = {}, {}

    def get_mesh(subj):
        if subj not in mesh_cache:
            mesh_cache[subj] = base.load_obj(os.path.join(args.biwi_dir, f"{subj}.obj"))
        return mesh_cache[subj]

    def get_cal(subj):
        if subj not in cal_cache:
            cal_cache[subj] = base.load_rgb_intrinsics(os.path.join(args.biwi_dir, subj, "rgb.cal"))
        return cal_cache[subj]

    all_dirs = sorted(glob.glob(os.path.join(args.biwi_dir, "*")))
    test_dirs = [
        d
        for d in all_dirs
        if os.path.isdir(d)
        and os.path.basename(d).isdigit()
        and int(os.path.basename(d)) in test_subjects
    ]
    pair_subset_by_subject = _load_pair_subset_csv(args.pair_csv) if args.pair_csv else None
    logging.info(f"Evaluating {len(test_dirs)} VGGT test subjects\n")

    vggt_tot_y = vggt_tot_p = vggt_tot_r = 0.0
    sixd_tot_y = sixd_tot_p = sixd_tot_r = 0.0
    vggt_tot_t = vggt_translation._empty_translation_acc()
    vggt_tot_t_norm = vggt_translation._empty_normalized_translation_acc()
    vggt_tot_t_relproj = vggt_translation._empty_normalized_translation_acc()
    translation_diagnostic_totals = {}
    translation_prediction_rows = []
    pose_prediction_rows = []
    multi_anchor_records = defaultdict(list)
    multi_anchor_missing_variant_warned = False
    total_count = 0
    vis_count = 0

    per_subject = {}
    global_variant_errors = base._empty_variant_acc(variant_names)
    global_variant_translation_errors = {
        vname: vggt_translation._empty_translation_acc()
        for vname in variant_names
    }

    for subj_dir in test_dirs:
        subj = os.path.basename(subj_dir)

        if not os.path.exists(os.path.join(args.biwi_dir, f"{subj}.obj")):
            logging.warning(f"Subject {subj}: no .obj mesh, skipping.")
            continue
        if not os.path.exists(os.path.join(subj_dir, "rgb.cal")):
            logging.warning(f"Subject {subj}: no rgb.cal, skipping.")
            continue

        frames = base.get_subject_frames(subj_dir)
        use_anchor_pair = args.vggt_pose_mode in {"relative_h2c", "anchor_abs_relative_h2c", "absolute_second"}
        min_required_frames = 2 if use_anchor_pair else 1
        if len(frames) < min_required_frames:
            logging.warning(f"Subject {subj}: <{min_required_frames} frames, skipping.")
            continue

        subj_pair_subset = None
        if pair_subset_by_subject is not None:
            subj_pair_subset = pair_subset_by_subject.get(subj, [])
            if len(subj_pair_subset) == 0:
                logging.info(f"Subject {subj}: no rows in pair CSV, skipping.")
                continue

        vertices, faces = get_mesh(subj)
        K_rgb, R_cam, t_cam = get_cal(subj)

        R_cam_t = torch.from_numpy(R_cam).float()
        t_cam_t = torch.from_numpy(t_cam).float()
        eval_groups = _build_subject_eval_groups(subj, frames, args, subj_pair_subset)
        if len(eval_groups) == 0:
            logging.warning(f"Subject {subj}: no valid evaluation groups, skipping.")
            continue

        total_targets = sum(len(group["targets"]) for group in eval_groups)
        if args.pair_csv:
            logging.info(f"[{subj}] pair-CSV mode: {len(eval_groups)} anchor groups, {total_targets} target pairs")
        elif use_anchor_pair:
            logging.info(
                f"[{subj}] anchor=frame {eval_groups[0]['anchor'][0]}, "
                f"{total_targets} targets, {(total_targets + args.batch_size - 1) // args.batch_size} batches"
            )
        else:
            logging.info(
                f"[{subj}] single-view evaluation, {total_targets} frames, "
                f"{(total_targets + args.batch_size - 1) // args.batch_size} batches"
            )

        s_variant_errors = base._empty_variant_acc(variant_names)
        s_sixd_y = s_sixd_p = s_sixd_r = 0.0
        s_vggt_t = vggt_translation._empty_translation_acc()
        s_vggt_t_norm = vggt_translation._empty_normalized_translation_acc()
        s_vggt_t_relproj = vggt_translation._empty_normalized_translation_acc()
        s_translation_diagnostics = {}
        s_variant_translation_errors = {
            vname: vggt_translation._empty_translation_acc()
            for vname in variant_names
        }
        s_count = 0

        for group_idx, group in enumerate(eval_groups):
            anchor_tensor = None
            anchor_box = None
            anchor_crop_params = None
            anchor_bgr = None
            R_anchor_t = None
            t_anchor_t = None
            mtcnn_prev_box = None

            if use_anchor_pair:
                _, anchor_rgb_path, anchor_pose_path = group["anchor"]
                if args.anchor_pose_source == "flame_pnp":
                    R_anchor, t_anchor = _load_flame_pnp_anchor_pose(
                        args,
                        subj,
                        anchor_rgb_path,
                        K_rgb,
                        R_cam,
                        t_cam,
                    )
                else:
                    R_anchor, t_anchor = base.parse_pose_txt(anchor_pose_path)
                anchor_bgr = cv2.imread(anchor_rgb_path)
                if anchor_bgr is None:
                    logging.warning(f"Subject {subj}: failed to read anchor frame {anchor_rgb_path}, skipping group.")
                    continue
                if args.vggt_crop_schema == "full_frame_square":
                    anchor_tensor = base.vggt_transform(_full_frame_square_rgb(anchor_bgr, args.crop_size))
                    anchor_crop_params = _identity_crop_params(anchor_bgr)
                else:
                    _, anchor_box = base.crop_face_mtcnn(
                        anchor_bgr,
                        mtcnn_detector,
                        None,
                        crop_size=args.crop_size,
                        ad=args.mtcnn_ad,
                    )
                    anchor_crop_params = _square_box_crop_params(anchor_bgr, anchor_box)
                    if args.vggt_crop_schema == "mtcnn_square_per_view":
                        anchor_tensor = base.vggt_transform(
                            _square_box_rgb(anchor_bgr, anchor_box, args.crop_size)
                        )
                R_anchor_t = torch.from_numpy(R_anchor).float()
                t_anchor_t = torch.from_numpy(t_anchor).float()

            target_frames = group["targets"]
            n_targets = len(target_frames)
            n_batches = (n_targets + args.batch_size - 1) // args.batch_size

            if use_anchor_pair:
                logging.info(
                    f"  group {group_idx + 1}/{len(eval_groups)}: anchor=frame {group['anchor'][0]}, "
                    f"{n_targets} targets, {n_batches} batches"
                )
            else:
                logging.info(f"  group {group_idx + 1}/{len(eval_groups)}: {n_targets} targets, {n_batches} batches")

            for b_idx, b_start in enumerate(range(0, n_targets, args.batch_size)):
                batch = target_frames[b_start : b_start + args.batch_size]

                if b_idx % 20 == 0:
                    logging.info(f"    batch {b_idx + 1}/{n_batches}")

                vggt_tensors, anchor_tensors, sixd_tensors = [], [], []
                vggt_crop_params, anchor_crop_params_batch = [], []
                frame_nums, R_targets, t_targets = [], [], []
                imgs_rgb = []

                for frame_num, rgb_path, pose_path in batch:
                    R_tgt, t_tgt = base.parse_pose_txt(pose_path)
                    img_bgr = cv2.imread(rgb_path)
                    if img_bgr is None:
                        logging.warning(f"Failed to read target frame {rgb_path}, skipping frame.")
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    rect_crop_bgr = None
                    target_box = None
                    if args.vggt_crop_schema == "full_frame_square":
                        vggt_tensors.append(
                            base.vggt_transform(_full_frame_square_rgb(img_bgr, args.crop_size))
                        )
                        vggt_crop_params.append(_identity_crop_params(img_bgr))
                        if sixd_model is not None:
                            rect_crop_bgr, mtcnn_prev_box = base.crop_face_mtcnn(
                                img_bgr,
                                mtcnn_detector,
                                mtcnn_prev_box,
                                crop_size=args.crop_size,
                                ad=args.mtcnn_ad,
                            )
                    else:
                        rect_crop_bgr, mtcnn_prev_box = base.crop_face_mtcnn(
                            img_bgr,
                            mtcnn_detector,
                            mtcnn_prev_box,
                            crop_size=args.crop_size,
                            ad=args.mtcnn_ad,
                        )
                        target_box = mtcnn_prev_box

                        if args.vggt_crop_schema == "mtcnn_square_pair_union":
                            pair_box = _union_mtcnn_boxes(anchor_box, target_box)
                            anchor_tensors.append(
                                base.vggt_transform(_square_box_rgb(anchor_bgr, pair_box, args.crop_size))
                            )
                            anchor_crop_params_batch.append(_square_box_crop_params(anchor_bgr, pair_box))
                            vggt_tensors.append(
                                base.vggt_transform(_square_box_rgb(img_bgr, pair_box, args.crop_size))
                            )
                            vggt_crop_params.append(_square_box_crop_params(img_bgr, pair_box))
                        else:
                            vggt_tensors.append(
                                base.vggt_transform(_square_box_rgb(img_bgr, target_box, args.crop_size))
                            )
                            vggt_crop_params.append(_square_box_crop_params(img_bgr, target_box))

                    if sixd_model is not None:
                        sixd_tensors.append(base.SIXD_TRANSFORM(Image.fromarray(rect_crop_bgr)))

                    frame_nums.append(frame_num)
                    R_targets.append(torch.from_numpy(R_tgt).float())
                    t_targets.append(torch.from_numpy(t_tgt).float())
                    imgs_rgb.append(img_rgb)

                if len(vggt_tensors) == 0:
                    continue

                bs = len(vggt_tensors)
                R_gt_batch = torch.stack(R_targets)
                t_gt_batch = torch.stack(t_targets)
                y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(R_gt_batch)

                tgt_stack = torch.stack(vggt_tensors)
                tgt_crop_param_stack = torch.stack(vggt_crop_params)
                if use_anchor_pair:
                    if args.vggt_crop_schema == "mtcnn_square_pair_union":
                        anchor_stack = torch.stack(anchor_tensors)
                        anchor_crop_param_stack = torch.stack(anchor_crop_params_batch)
                    else:
                        anchor_stack = anchor_tensor.unsqueeze(0).expand(bs, -1, -1, -1)
                        anchor_crop_param_stack = anchor_crop_params.unsqueeze(0).expand(bs, -1)
                    vggt_images = torch.stack([anchor_stack, tgt_stack], dim=1).to(device)
                    vggt_crop_param_tensor = torch.stack(
                        [anchor_crop_param_stack, tgt_crop_param_stack],
                        dim=1,
                    ).to(device)
                else:
                    vggt_images = tgt_stack.unsqueeze(1).to(device)
                    vggt_crop_param_tensor = tgt_crop_param_stack.unsqueeze(1).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                    model_kwargs = {"images": vggt_images}
                    if args.vggt_crop_posenc_dim > 0:
                        model_kwargs["crop_params"] = vggt_crop_param_tensor
                    vggt_out = vggt_model(**model_kwargs)

                decoded_vggt = vggt_translation.decode_vggt_predictions(
                    vggt_out["pose_enc"],
                    R_anchor_t,
                    t_anchor_t,
                    R_cam_t,
                    t_cam_t,
                    R_target_t=R_gt_batch,
                    pose_enc_abs=vggt_out.get("pose_enc_abs"),
                    pose_mode=args.vggt_pose_mode,
                    translation_scale=args.vggt_translation_scale,
                    pose_encoding_type=args.vggt_pose_encoding_type,
                    translation_rotation_mode=args.vggt_translation_rotation_mode,
                    translation_target_type=args.vggt_translation_target_type,
                    return_translation_diagnostics=args.translation_diagnostics,
                    return_translation_variants=args.vggt_pose_mode == "anchor_abs_relative_h2c",
                )
                translation_variants = None
                if args.translation_diagnostics and args.vggt_pose_mode == "anchor_abs_relative_h2c":
                    variants, t_vggt_primary, translation_diagnostics, translation_variants = decoded_vggt
                elif args.translation_diagnostics:
                    variants, t_vggt_primary, translation_diagnostics = decoded_vggt
                elif args.vggt_pose_mode == "anchor_abs_relative_h2c":
                    variants, t_vggt_primary, translation_variants = decoded_vggt
                    translation_diagnostics = None
                else:
                    variants, t_vggt_primary = decoded_vggt
                    translation_diagnostics = None
                if translation_variants is None:
                    translation_variants = {primary_variant: t_vggt_primary}

                if args.multi_anchor_eval and args.vggt_pose_mode == "anchor_abs_relative_h2c":
                    multi_variant = args.multi_anchor_variant
                    if multi_variant in variants and multi_variant in translation_variants:
                        R_multi = variants[multi_variant].detach().cpu().float().numpy()
                        t_multi = translation_variants[multi_variant].detach().cpu().float().numpy()
                        R_gt_np = R_gt_batch.detach().cpu().float().numpy()
                        t_gt_np = t_gt_batch.detach().cpu().float().numpy()
                        anchor_frame_num = group["anchor"][0]
                        for fi, frame_num in enumerate(frame_nums):
                            multi_anchor_records[(subj, frame_num)].append(
                                {
                                    "anchor_frame": anchor_frame_num,
                                    "R_pred": R_multi[fi],
                                    "t_pred": t_multi[fi],
                                    "R_gt": R_gt_np[fi],
                                    "t_gt": t_gt_np[fi],
                                }
                            )
                    elif not multi_anchor_missing_variant_warned:
                        logging.warning(
                            "Multi-anchor variant %s not available. Variants=%s, translation_variants=%s",
                            multi_variant,
                            sorted(variants.keys()),
                            sorted(translation_variants.keys()),
                        )
                        multi_anchor_missing_variant_warned = True

                for vname, R_v in variants.items():
                    yv, pv, rv = base.biwi_euler_deg_batch(R_v)
                    err_y = base.angular_error(y_gt, yv).sum().item()
                    err_p = base.angular_error(p_gt, pv).sum().item()
                    err_r = base.angular_error(r_gt, rv).sum().item()
                    global_variant_errors[vname]["y"] += err_y
                    global_variant_errors[vname]["p"] += err_p
                    global_variant_errors[vname]["r"] += err_r
                    s_variant_errors[vname]["y"] += err_y
                    s_variant_errors[vname]["p"] += err_p
                    s_variant_errors[vname]["r"] += err_r

                R_vggt_primary = variants[primary_variant]
                y_vp, p_vp, r_vp = base.biwi_euler_deg_batch(R_vggt_primary)
                vggt_tot_y += base.angular_error(y_gt, y_vp).sum().item()
                vggt_tot_p += base.angular_error(p_gt, p_vp).sum().item()
                vggt_tot_r += base.angular_error(r_gt, r_vp).sum().item()
                vggt_translation.accumulate_translation_errors(vggt_tot_t, t_vggt_primary, t_gt_batch)
                vggt_translation.accumulate_translation_errors(s_vggt_t, t_vggt_primary, t_gt_batch)
                for tv_name, tv_pred in translation_variants.items():
                    if tv_name in global_variant_translation_errors:
                        vggt_translation.accumulate_translation_errors(
                            global_variant_translation_errors[tv_name],
                            tv_pred,
                            t_gt_batch,
                        )
                    if tv_name in s_variant_translation_errors:
                        vggt_translation.accumulate_translation_errors(
                            s_variant_translation_errors[tv_name],
                            tv_pred,
                            t_gt_batch,
                        )
                vggt_translation.accumulate_normalized_translation_errors(
                    vggt_tot_t_norm,
                    t_vggt_primary,
                    t_gt_batch,
                )
                vggt_translation.accumulate_normalized_translation_errors(
                    s_vggt_t_norm,
                    t_vggt_primary,
                    t_gt_batch,
                )
                if use_anchor_pair and t_anchor_t is not None:
                    anchor_t_batch = t_anchor_t.unsqueeze(0).expand_as(t_gt_batch)
                    vggt_translation.accumulate_relative_projective_translation_errors(
                        vggt_tot_t_relproj,
                        t_vggt_primary,
                        t_gt_batch,
                        anchor_t_batch,
                    )
                    vggt_translation.accumulate_relative_projective_translation_errors(
                        s_vggt_t_relproj,
                        t_vggt_primary,
                        t_gt_batch,
                        anchor_t_batch,
                    )
                if args.translation_prediction_diagnostics:
                    anchor_frame_num = group["anchor"][0] if use_anchor_pair else ""
                    _append_translation_prediction_rows(
                        translation_prediction_rows,
                        subj,
                        anchor_frame_num,
                        frame_nums,
                        t_vggt_primary,
                        t_gt_batch,
                        t_anchor_t if use_anchor_pair else None,
                        K_rgb=K_rgb,
                        image_shape_hw=imgs_rgb[0].shape[:2] if imgs_rgb else None,
                        anchor_crop_params=anchor_crop_param_stack if use_anchor_pair else None,
                        target_crop_params=tgt_crop_param_stack,
                        crop_schema=args.vggt_crop_schema,
                    )
                if args.pose_prediction_diagnostics:
                    anchor_frame_num = group["anchor"][0] if use_anchor_pair else ""
                    _append_pose_prediction_rows(
                        pose_prediction_rows,
                        subj,
                        anchor_frame_num,
                        frame_nums,
                        R_vggt_primary,
                        R_gt_batch,
                        t_vggt_primary,
                        t_gt_batch,
                        base.angular_error(y_gt, y_vp),
                        base.angular_error(p_gt, p_vp),
                        base.angular_error(r_gt, r_vp),
                    )
                if translation_diagnostics is not None:
                    vggt_translation.accumulate_translation_diagnostics(
                        translation_diagnostic_totals,
                        translation_diagnostics,
                        t_gt_batch,
                    )
                    vggt_translation.accumulate_translation_diagnostics(
                        s_translation_diagnostics,
                        translation_diagnostics,
                        t_gt_batch,
                    )

                y_sp = p_sp = r_sp = None
                R_sixd_biwi = None
                if sixd_model is not None:
                    sixd_imgs = torch.stack(sixd_tensors).to(device)
                    with torch.no_grad():
                        R_sixd_batch = sixd_model(sixd_imgs).cpu()
                    y_sp, p_sp, r_sp, R_sixd_biwi = base.decode_sixdrepnet_pose(R_sixd_batch)

                    s_sixd_y += base.angular_error(y_gt, y_sp).sum().item()
                    s_sixd_p += base.angular_error(p_gt, p_sp).sum().item()
                    s_sixd_r += base.angular_error(r_gt, r_sp).sum().item()
                    sixd_tot_y += base.angular_error(y_gt, y_sp).sum().item()
                    sixd_tot_p += base.angular_error(p_gt, p_sp).sum().item()
                    sixd_tot_r += base.angular_error(r_gt, r_sp).sum().item()

                s_count += bs

                if args.visualize:
                    for fi in range(bs):
                        if vis_count % args.vis_every_n == 0:
                            frame_num = frame_nums[fi]
                            save_path = os.path.join(args.vis_dir, f"subj{subj}_frame{frame_num:05d}.png")

                            euler_gt_i = (y_gt[fi].item(), p_gt[fi].item(), r_gt[fi].item())
                            vggt_vis = {}
                            for vname, R_v in variants.items():
                                yv, pv, rv = base.biwi_euler_deg_batch(R_v)
                                vggt_vis[vname] = (
                                    R_v[fi].numpy(),
                                    (yv[fi].item(), pv[fi].item(), rv[fi].item()),
                                )

                            euler_sixd_i = (
                                (y_sp[fi].item(), p_sp[fi].item(), r_sp[fi].item())
                                if y_sp is not None
                                else None
                            )

                            base.save_vis_frame(
                                img_rgb=imgs_rgb[fi],
                                vertices=vertices,
                                faces=faces,
                                K_rgb=K_rgb,
                                R_cam=R_cam,
                                t_cam=t_cam,
                                R_gt=R_gt_batch[fi].numpy(),
                                t_gt=t_gt_batch[fi].numpy(),
                                euler_gt=euler_gt_i,
                                vggt_variants=vggt_vis,
                                R_sixd=R_sixd_biwi[fi] if R_sixd_biwi is not None else None,
                                euler_sixd=euler_sixd_i,
                                save_path=save_path,
                            )
                        vis_count += 1

        if s_count > 0:
            per_subject[subj] = {}
            for vname in variant_names:
                vy = s_variant_errors[vname]["y"] / s_count
                vp = s_variant_errors[vname]["p"] / s_count
                vr = s_variant_errors[vname]["r"] / s_count
                vm = (vy + vp + vr) / 3
                per_subject[subj][f"vggt_{vname}"] = {
                    "yaw": vy,
                    "pitch": vp,
                    "roll": vr,
                    "mean": vm,
                }

            t_metrics = vggt_translation.translation_metrics_from_acc(s_vggt_t, s_count)
            per_subject[subj][f"vggt_{primary_variant}"].update(t_metrics)
            if args.vggt_pose_mode == "anchor_abs_relative_h2c":
                for vname in variant_names:
                    per_subject[subj][f"vggt_{vname}"].update(
                        vggt_translation.translation_metrics_from_acc(
                            s_variant_translation_errors[vname],
                            s_count,
                        )
                    )
            if args.translation_diagnostics:
                for diag_name, diag_acc in sorted(s_translation_diagnostics.items()):
                    diag_metrics = vggt_translation.translation_metrics_from_acc(diag_acc, s_count)
                    per_subject[subj][f"vggt_translation_diag_{diag_name}"] = diag_metrics

            vy = per_subject[subj][f"vggt_{primary_variant}"]["yaw"]
            vp = per_subject[subj][f"vggt_{primary_variant}"]["pitch"]
            vr = per_subject[subj][f"vggt_{primary_variant}"]["roll"]
            vm = per_subject[subj][f"vggt_{primary_variant}"]["mean"]
            line = (
                f"Subject {subj} ({s_count} frames):\n"
                f"  VGGT ({primary_variant}) — Yaw: {vy:.2f}  Pitch: {vp:.2f}  Roll: {vr:.2f}  MAE: {vm:.2f}\n"
                f"  VGGT Translation    — Tx: {t_metrics['tx']:.2f}  Ty: {t_metrics['ty']:.2f}"
                f"  Tz: {t_metrics['tz']:.2f}  L2: {t_metrics['l2']:.2f}"
            )

            if "vggt_B_no_flip" in per_subject[subj]:
                by = per_subject[subj]["vggt_B_no_flip"]["yaw"]
                bp = per_subject[subj]["vggt_B_no_flip"]["pitch"]
                br = per_subject[subj]["vggt_B_no_flip"]["roll"]
                bm = per_subject[subj]["vggt_B_no_flip"]["mean"]
                line += (
                    f"\n  VGGT (B_no_flip) — Yaw: {by:.2f}  Pitch: {bp:.2f}"
                    f"  Roll: {br:.2f}  MAE: {bm:.2f}"
                )

            if sixd_model is not None:
                sy = s_sixd_y / s_count
                sp = s_sixd_p / s_count
                sr = s_sixd_r / s_count
                sm = (sy + sp + sr) / 3
                per_subject[subj]["sixdrepnet"] = {
                    "yaw": sy,
                    "pitch": sp,
                    "roll": sr,
                    "mean": sm,
                }
                line += f"\n  6DRepNet          — Yaw: {sy:.2f}  Pitch: {sp:.2f}  Roll: {sr:.2f}  MAE: {sm:.2f}"
            logging.info(line)

        total_count += s_count
        gc.collect()
        torch.cuda.empty_cache()

    sixdof_overall, sixdof_per_subject = None, {}
    try:
        sixdof_overall, sixdof_per_subject = evaluate_sixdof_face(args)
    except Exception as e:
        logging.error(f"6DoF Face evaluation failed: {e}")
        logging.error("Continuing with VGGT/6DRepNet results only.")
    for subj, metrics in sixdof_per_subject.items():
        per_subject.setdefault(subj, {})
        per_subject[subj]["sixdof_face"] = metrics

    print("\n" + "=" * 60)
    print("BIWI EVALUATION RESULTS")
    print("=" * 60)
    if total_count > 0:
        vy = vggt_tot_y / total_count
        vp = vggt_tot_p / total_count
        vr = vggt_tot_r / total_count
        vm = (vy + vp + vr) / 3
        t_metrics = vggt_translation.translation_metrics_from_acc(vggt_tot_t, total_count)
        t_norm_metrics = vggt_translation.normalized_translation_metrics_from_acc(vggt_tot_t_norm, total_count)
        t_relproj_metrics = vggt_translation.normalized_translation_metrics_from_acc(vggt_tot_t_relproj, total_count)

        print(f"Frames evaluated: {total_count}")
        print(f"\nVGGT ({primary_variant}) — Yaw: {vy:.4f}  Pitch: {vp:.4f}  Roll: {vr:.4f}  MAE: {vm:.4f}")
        print(
            f"VGGT Translation    — Tx: {t_metrics['tx']:.4f}  Ty: {t_metrics['ty']:.4f}"
            f"  Tz: {t_metrics['tz']:.4f}  L2: {t_metrics['l2']:.4f}"
        )
        print(
            f"VGGT Norm Camera    — X/Z: {t_norm_metrics['x_over_z']:.6f}"
            f"  Y/Z: {t_norm_metrics['y_over_z']:.6f}"
            f"  logZ: {t_norm_metrics['log_z']:.6f}"
            f"  L2: {t_norm_metrics['l2']:.6f}"
        )
        print(
            f"VGGT Rel Projective — dX/Z: {t_relproj_metrics['x_over_z']:.6f}"
            f"  dY/Z: {t_relproj_metrics['y_over_z']:.6f}"
            f"  logZ ratio: {t_relproj_metrics['log_z']:.6f}"
            f"  L2: {t_relproj_metrics['l2']:.6f}"
        )
        if sixd_model is not None:
            sy = sixd_tot_y / total_count
            sp = sixd_tot_p / total_count
            sr = sixd_tot_r / total_count
            sm = (sy + sp + sr) / 3
            print(f"6DRepNet           — Yaw: {sy:.4f}  Pitch: {sp:.4f}  Roll: {sr:.4f}  MAE: {sm:.4f}")
        if sixdof_overall is not None:
            print(
                f"6DoF Face          — Yaw: {sixdof_overall['yaw']:.4f}  Pitch: {sixdof_overall['pitch']:.4f}"
                f"  Roll: {sixdof_overall['roll']:.4f}  MAE: {sixdof_overall['mean']:.4f}"
            )
            print(
                f"6DoF Face Trans    — Tx: {sixdof_overall['tx']:.4f}  Ty: {sixdof_overall['ty']:.4f}"
                f"  Tz: {sixdof_overall['tz']:.4f}  L2: {sixdof_overall['l2']:.4f}"
            )

        print(f"\n{'─' * 60}")
        print("VGGT variant comparison:")
        print(f"{'─' * 60}")
        best_name, best_mae = None, float("inf")
        for vname in variant_names:
            errs = global_variant_errors[vname]
            vy_ = errs["y"] / total_count
            vp_ = errs["p"] / total_count
            vr_ = errs["r"] / total_count
            vm_ = (vy_ + vp_ + vr_) / 3
            if vm_ < best_mae:
                best_mae = vm_
                best_name = vname
            marker = " <- best" if vname == best_name else ""
            line = f"  {vname:15s} — Yaw: {vy_:.4f}  Pitch: {vp_:.4f}  Roll: {vr_:.4f}  MAE: {vm_:.4f}"
            if args.vggt_pose_mode == "anchor_abs_relative_h2c":
                tv_metrics = vggt_translation.translation_metrics_from_acc(
                    global_variant_translation_errors[vname],
                    total_count,
                )
                line += (
                    f"  Tx: {tv_metrics['tx']:.4f}  Ty: {tv_metrics['ty']:.4f}"
                    f"  Tz: {tv_metrics['tz']:.4f}  L2: {tv_metrics['l2']:.4f}"
                )
            print(f"{line}{marker}")
        print(f"\n  Best variant: {best_name}  (MAE: {best_mae:.4f} deg)")
        if args.translation_diagnostics and translation_diagnostic_totals:
            print(f"\n{'─' * 60}")
            print("VGGT translation decode diagnostics:")
            print(f"{'─' * 60}")
            ranked = []
            for diag_name, diag_acc in translation_diagnostic_totals.items():
                diag_metrics = vggt_translation.translation_metrics_from_acc(diag_acc, total_count)
                ranked.append((diag_metrics["l2"], diag_name, diag_metrics))
            for _, diag_name, diag_metrics in sorted(ranked):
                print(
                    f"  {diag_name:22s} — Tx: {diag_metrics['tx']:.4f}  Ty: {diag_metrics['ty']:.4f}"
                    f"  Tz: {diag_metrics['tz']:.4f}  L2: {diag_metrics['l2']:.4f}"
                )
    else:
        print("No VGGT frames evaluated.")
    print("=" * 60)

    csv_path = os.path.join(args.vis_dir, "biwi_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "subject",
            "model",
            "yaw_mae",
            "pitch_mae",
            "roll_mae",
            "mean_mae",
            "tx_mae_mm",
            "ty_mae_mm",
            "tz_mae_mm",
            "translation_l2_mae_mm",
        ]
        w.writerow(header)
        for subj, res in sorted(per_subject.items(), key=lambda item: item[0]):
            for model_name, vals in res.items():
                w.writerow(
                    [
                        subj,
                        model_name,
                        f"{vals.get('yaw', float('nan')):.4f}" if "yaw" in vals else "",
                        f"{vals.get('pitch', float('nan')):.4f}" if "pitch" in vals else "",
                        f"{vals.get('roll', float('nan')):.4f}" if "roll" in vals else "",
                        f"{vals.get('mean', float('nan')):.4f}" if "mean" in vals else "",
                        f"{vals.get('tx', float('nan')):.4f}" if "tx" in vals else "",
                        f"{vals.get('ty', float('nan')):.4f}" if "ty" in vals else "",
                        f"{vals.get('tz', float('nan')):.4f}" if "tz" in vals else "",
                        f"{vals.get('l2', float('nan')):.4f}" if "l2" in vals else "",
                    ]
                )

        if total_count > 0:
            t_metrics = vggt_translation.translation_metrics_from_acc(vggt_tot_t, total_count)
            t_norm_metrics = vggt_translation.normalized_translation_metrics_from_acc(vggt_tot_t_norm, total_count)
            t_relproj_metrics = vggt_translation.normalized_translation_metrics_from_acc(vggt_tot_t_relproj, total_count)
            w.writerow([])
            w.writerow(["# Overall results"])
            w.writerow(header)
            w.writerow(
                [
                    "overall",
                    f"vggt_{primary_variant}",
                    f"{vggt_tot_y / total_count:.4f}",
                    f"{vggt_tot_p / total_count:.4f}",
                    f"{vggt_tot_r / total_count:.4f}",
                    f"{(vggt_tot_y + vggt_tot_p + vggt_tot_r) / (total_count * 3):.4f}",
                    f"{t_metrics['tx']:.4f}",
                    f"{t_metrics['ty']:.4f}",
                    f"{t_metrics['tz']:.4f}",
                    f"{t_metrics['l2']:.4f}",
                ]
            )
            if args.vggt_pose_mode == "anchor_abs_relative_h2c":
                for vname in variant_names:
                    if vname == primary_variant:
                        continue
                    errs = global_variant_errors[vname]
                    tv_metrics = vggt_translation.translation_metrics_from_acc(
                        global_variant_translation_errors[vname],
                        total_count,
                    )
                    w.writerow(
                        [
                            "overall",
                            f"vggt_{vname}",
                            f"{errs['y'] / total_count:.4f}",
                            f"{errs['p'] / total_count:.4f}",
                            f"{errs['r'] / total_count:.4f}",
                            f"{(errs['y'] + errs['p'] + errs['r']) / (total_count * 3):.4f}",
                            f"{tv_metrics['tx']:.4f}",
                            f"{tv_metrics['ty']:.4f}",
                            f"{tv_metrics['tz']:.4f}",
                            f"{tv_metrics['l2']:.4f}",
                        ]
                    )
            w.writerow([])
            w.writerow(["# Normalized camera translation"])
            w.writerow(["model", "x_over_z_mae", "y_over_z_mae", "log_z_mae", "normalized_l2_mae"])
            w.writerow(
                [
                    f"vggt_{primary_variant}",
                    f"{t_norm_metrics['x_over_z']:.8f}",
                    f"{t_norm_metrics['y_over_z']:.8f}",
                    f"{t_norm_metrics['log_z']:.8f}",
                    f"{t_norm_metrics['l2']:.8f}",
                ]
            )
            w.writerow([])
            w.writerow(["# Relative projective translation"])
            w.writerow(["model", "delta_x_over_z_mae", "delta_y_over_z_mae", "log_z_ratio_mae", "relative_projective_l2_mae"])
            w.writerow(
                [
                    f"vggt_{primary_variant}",
                    f"{t_relproj_metrics['x_over_z']:.8f}",
                    f"{t_relproj_metrics['y_over_z']:.8f}",
                    f"{t_relproj_metrics['log_z']:.8f}",
                    f"{t_relproj_metrics['l2']:.8f}",
                ]
            )
            if sixd_model is not None:
                w.writerow(
                    [
                        "overall",
                        "sixdrepnet",
                        f"{sixd_tot_y / total_count:.4f}",
                        f"{sixd_tot_p / total_count:.4f}",
                        f"{sixd_tot_r / total_count:.4f}",
                        f"{(sixd_tot_y + sixd_tot_p + sixd_tot_r) / (total_count * 3):.4f}",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
        if sixdof_overall is not None:
            w.writerow(
                [
                    "overall",
                    "sixdof_face",
                    f"{sixdof_overall['yaw']:.4f}",
                    f"{sixdof_overall['pitch']:.4f}",
                    f"{sixdof_overall['roll']:.4f}",
                    f"{sixdof_overall['mean']:.4f}",
                    f"{sixdof_overall['tx']:.4f}",
                    f"{sixdof_overall['ty']:.4f}",
                    f"{sixdof_overall['tz']:.4f}",
                    f"{sixdof_overall['l2']:.4f}",
                ]
            )

        if total_count > 0:
            w.writerow([])
            w.writerow(["# VGGT variants (rotation only)"])
            w.writerow(["variant", "model", "yaw_mae", "pitch_mae", "roll_mae", "mean_mae"])
            best_name, best_mae = None, float("inf")
            for vname in variant_names:
                errs = global_variant_errors[vname]
                vy_ = errs["y"] / total_count
                vp_ = errs["p"] / total_count
                vr_ = errs["r"] / total_count
                vm_ = (vy_ + vp_ + vr_) / 3
                w.writerow([vname, "vggt", f"{vy_:.4f}", f"{vp_:.4f}", f"{vr_:.4f}", f"{vm_:.4f}"])
                if vm_ < best_mae:
                    best_mae = vm_
                    best_name = vname
            w.writerow(["best_variant", best_name, "", "", "", f"{best_mae:.4f}"])

            if args.translation_diagnostics and translation_diagnostic_totals:
                w.writerow([])
                w.writerow(["# Translation decode diagnostics"])
                w.writerow(["diagnostic", "tx_mae_mm", "ty_mae_mm", "tz_mae_mm", "translation_l2_mae_mm"])
                ranked = []
                for diag_name, diag_acc in translation_diagnostic_totals.items():
                    diag_metrics = vggt_translation.translation_metrics_from_acc(diag_acc, total_count)
                    ranked.append((diag_metrics["l2"], diag_name, diag_metrics))
                for _, diag_name, diag_metrics in sorted(ranked):
                    w.writerow(
                        [
                            diag_name,
                            f"{diag_metrics['tx']:.4f}",
                            f"{diag_metrics['ty']:.4f}",
                            f"{diag_metrics['tz']:.4f}",
                            f"{diag_metrics['l2']:.4f}",
                        ]
                    )

    if args.translation_prediction_diagnostics:
        pred_csv_path = os.path.join(args.vis_dir, "translation_predictions.csv")
        with open(pred_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRANSLATION_PREDICTION_FIELDS)
            w.writeheader()
            for row in translation_prediction_rows:
                w.writerow(row)

        summary_csv_path = os.path.join(args.vis_dir, "translation_prediction_summary.csv")
        summary_rows = _summarize_translation_predictions(translation_prediction_rows)
        summary_fields = [
            "scope",
            "subject",
            "metric",
            "count",
            "mae",
            "rmse",
            "bias_pred_minus_gt",
            "pred_mean",
            "gt_mean",
            "pearson",
            "slope_pred_vs_gt",
            "intercept_pred_vs_gt",
            "gt_std",
            "pred_std",
        ]
        with open(summary_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
        logging.info(f"Translation per-frame predictions saved -> {pred_csv_path}")
        logging.info(f"Translation prediction summary saved -> {summary_csv_path}")

    if args.pose_prediction_diagnostics:
        pose_csv_path = os.path.join(args.vis_dir, "pose_predictions.csv")
        with open(pose_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=POSE_PREDICTION_FIELDS)
            w.writeheader()
            for row in pose_prediction_rows:
                w.writerow(row)
        logging.info(f"Pose per-frame predictions saved -> {pose_csv_path}")

    if args.multi_anchor_eval:
        _write_multi_anchor_results(args, multi_anchor_records)
        logging.info(
            "Multi-anchor fused results saved -> %s",
            os.path.join(args.vis_dir, "biwi_multi_anchor_results.csv"),
        )

    logging.info(f"Results saved -> {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="BIWI evaluation with schema-matched VGGT crops: VGGT + 6DRepNet + 6DoF Face"
    )

    p.add_argument("--biwi_dir", default="data/biwi/faces_0")
    p.add_argument(
        "--test_subjects",
        type=int,
        nargs="+",
        default=list(range(1, 25)),
        help="Subject IDs to evaluate (default: all 24 BIWI subjects).",
    )
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument(
        "--vggt_crop_schema",
        choices=["mtcnn_square_per_view", "mtcnn_square_pair_union", "full_frame_square"],
        default="mtcnn_square_per_view",
        help=(
            "VGGT input crop schema. Use mtcnn_square_per_view for per-view crop checkpoints, "
            "mtcnn_square_pair_union for unified-crop checkpoints, and full_frame_square for no-crop checkpoints."
        ),
    )
    p.add_argument(
        "--vggt_crop_posenc_dim",
        type=int,
        default=0,
        help="Optional crop positional encoding dimension expected by the VGGT camera head.",
    )
    p.add_argument(
        "--pair_csv",
        default="",
        help="Optional CSV of explicit anchor/query pairs to evaluate instead of the default per-subject protocol.",
    )
    p.add_argument(
        "--anchor_pose_source",
        choices=("gt", "flame_pnp"),
        default="gt",
        help="Pose supplied for the real enrollment anchor image.",
    )
    p.add_argument(
        "--anchor_pnp_alignment_root",
        default="",
        help="Root containing subjectXX FLAME enrollment alignments for flame_pnp anchors.",
    )
    p.add_argument(
        "--anchor_pnp_reference",
        choices=("person_expression", "person_neutral", "mean_neutral"),
        default="mean_neutral",
    )
    p.add_argument("--anchor_pnp_reprojection_threshold_px", type=float, default=5.0)

    p.add_argument("--base_checkpoint", default="checkpoints/VGGT-1B/model.pt")
    p.add_argument(
        "--lora_checkpoint",
        default="checkpoints/reva_hpe/checkpoint_410.pt",
    )
    p.add_argument("--no_lora", action="store_true")
    p.add_argument(
        "--vggt_pose_mode",
        choices=["relative_h2c", "anchor_abs_relative_h2c", "absolute_second", "absolute_single"],
        default="relative_h2c",
        help="How to decode the target-frame VGGT prediction during BIWI evaluation.",
    )
    p.add_argument(
        "--vggt_pose_encoding_type",
        default="absT_quaR_FoV",
        help="VGGT pose encoding used by the checkpoint (for example absT_quaR_FoV or absT_quaR).",
    )
    p.add_argument(
        "--vggt_translation_scale",
        type=float,
        default=1000.0,
        help=(
            "Multiplier applied to decoded VGGT translation before BIWI comparison "
            "(default 1000.0 assumes model outputs meters and BIWI is in mm)."
        ),
    )
    p.add_argument(
        "--vggt_translation_rotation_mode",
        choices=["predicted", "gt"],
        default="predicted",
        help=(
            "For relative_h2c translation composition, use the predicted relative rotation "
            "or the GT relative rotation. 'gt' is an oracle-R diagnostic."
        ),
    )
    p.add_argument(
        "--vggt_translation_target_type",
        choices=["relative_se3", "delta_t", "depth_norm_delta_t", "projective_delta_t"],
        default="relative_se3",
        help=(
            "How to interpret the translation slot for relative_h2c checkpoints. "
            "relative_se3 composes t_rel with R_rel, delta_t decodes it as t_query - t_anchor, "
            "depth_norm_delta_t decodes [(x1-x0)/z0, (y1-y0)/z0, log(z1/z0)], "
            "projective_delta_t decodes [x1/z1-x0/z0, y1/z1-y0/z0, log(z1/z0)]."
        ),
    )
    p.add_argument("--enable_track", action="store_true", help="Load pretrained track head (no query_points -> dormant at eval)")
    p.add_argument(
        "--translation_diagnostics",
        action="store_true",
        help="Evaluate alternate translation decode conventions without changing the primary VGGT metric.",
    )
    p.add_argument(
        "--translation_prediction_diagnostics",
        action="store_true",
        help="Save per-frame translation predictions and correlation/bias summaries.",
    )
    p.add_argument(
        "--pose_prediction_diagnostics",
        action="store_true",
        help="Save per-frame VGGT pose predictions with rotation matrices and translations.",
    )
    p.add_argument(
        "--multi_anchor_eval",
        action="store_true",
        help="Collect repeated anchor predictions per query and write fused multi-anchor metrics.",
    )
    p.add_argument(
        "--multi_anchor_count",
        type=int,
        default=1,
        help="Number of evenly spaced anchors per BIWI subject when no --pair_csv is provided.",
    )
    p.add_argument(
        "--multi_anchor_variant",
        default="REL_pred_anchor",
        help="anchor_abs_relative_h2c variant to fuse across anchors.",
    )
    p.add_argument(
        "--multi_anchor_fusion",
        choices=["medoid", "trimmed_mean"],
        default="trimmed_mean",
        help="Fusion rule for repeated anchor predictions.",
    )
    p.add_argument(
        "--multi_anchor_min_predictions",
        type=int,
        default=2,
        help="Minimum number of anchor predictions required for a query to enter fused metrics.",
    )

    p.add_argument("--sixdrepnet_dir", default=os.path.join(base.SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet"))
    p.add_argument(
        "--sixdrepnet_checkpoint",
        default=os.path.join(base.SCRIPT_DIR, "..", "6DRepNet", "sixdrepnet", "6DRepNet_70_30_BIWI.pth"),
    )
    p.add_argument("--no_sixdrepnet", action="store_true")

    p.add_argument("--sixdof_face_repo", default="third_party/6dof_face")
    p.add_argument(
        "--sixdof_face_data_root",
        default="",
        help="Preprocessed 6DoF Face BIWI evaluation root (contains csv/, image/, info/). Required if not under sixdof_face_repo/dataset/ARKitFace.",
    )
    p.add_argument("--sixdof_face_checkpoints_dir", default="")
    p.add_argument(
        "--sixdof_face_cache_root",
        default=os.path.join(base.SCRIPT_DIR, "cache", "sixdof_face_biwi_eval"),
        help="Where to auto-build the 6DoF Face BIWI evaluation root if it is missing.",
    )
    p.add_argument(
        "--prepare_sixdof_face_if_missing",
        action="store_true",
        help="Auto-build the 6DoF Face BIWI evaluation root from --biwi_dir when it is missing.",
    )
    p.add_argument("--sixdof_face_name", default="run1")
    p.add_argument("--sixdof_face_epoch", default="latest")
    p.add_argument("--sixdof_face_model", default="perspnet")
    p.add_argument("--sixdof_face_img_size", type=int, default=192)
    p.add_argument("--sixdof_face_batch_size", type=int, default=64)
    p.add_argument("--sixdof_face_num_workers", type=int, default=4)
    p.add_argument("--no_sixdof_face", action="store_true")

    p.add_argument("--mtcnn_ad", type=float, default=0.4, help="Shared MTCNN crop enlargement margin")

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--visualize", action="store_true")
    p.add_argument("--vis_dir", default=os.path.join(base.SCRIPT_DIR, "vis_output_mtcnn_square_compare"))
    p.add_argument("--vis_every_n", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
