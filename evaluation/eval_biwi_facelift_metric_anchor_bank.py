#!/usr/bin/env python3
"""Evaluate fixed enrollment-derived FaceLift metric anchors on BIWI queries."""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT / "eval_scripts"
sys.path.insert(0, str(EVAL_ROOT))
sys.path.insert(0, str(ROOT))

import eval_biwi_crop as base  
import eval_biwi_crop_mtcnn_square_translation as translation  
from eval_biwi_crop_mtcnn_square_compare import (  
    _square_box_rgb,
    _union_mtcnn_boxes,
)


LOG = logging.getLogger("facelift_metric_bank")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biwi-dir", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument(
        "--bank-name",
        default="metric_anchor_bank",
        help="Single bank name or comma-separated bank names to evaluate as one anchor pool.",
    )
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument("--translation-target-type", default="depth_norm_delta_t")
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--mtcnn-ad", type=float, default=0.4)
    parser.add_argument("--ransac-rot-threshold-deg", type=float, default=10.0)
    parser.add_argument("--ransac-trans-threshold-mm", type=float, default=50.0)
    parser.add_argument("--ransac-min-inliers", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument(
        "--frame-selection",
        choices=("post_enrollment", "enrollment_only", "all"),
        default="post_enrollment",
    )
    parser.add_argument("--max-visualizations", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def rotation_distance_deg(first, second):
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def project_rotation(matrix):
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def ransac_fuse(rotations, translations, rot_threshold_deg, trans_threshold_mm, min_inliers):
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    best = None
    for seed in range(len(rotations)):
        rotation_residual = np.asarray(
            [rotation_distance_deg(rotation, rotations[seed]) for rotation in rotations]
        )
        translation_residual = np.linalg.norm(translations - translations[seed], axis=1)
        inliers = (rotation_residual <= rot_threshold_deg) & (
            translation_residual <= trans_threshold_mm
        )
        ids = np.flatnonzero(inliers)
        normalized = (
            rotation_residual[ids] / max(rot_threshold_deg, 1e-6)
            + translation_residual[ids] / max(trans_threshold_mm, 1e-6)
        )
        score = (len(ids), -float(normalized.mean()) if len(ids) else -np.inf)
        if best is None or score > best[0]:
            best = (score, ids, seed)
    consensus_ids = best[1]
    success = len(consensus_ids) >= min_inliers
    inlier_ids = consensus_ids if success else np.asarray([best[2]], dtype=np.int64)
    rotation = project_rotation(rotations[inlier_ids].mean(axis=0))
    translation_vec = np.median(translations[inlier_ids], axis=0)
    return rotation, translation_vec, inlier_ids, success, len(consensus_ids)


def biwi_errors(rotation, translation_vec, gt_rotation, gt_translation):
    prediction = torch.from_numpy(rotation).float().unsqueeze(0)
    target = torch.from_numpy(gt_rotation).float().unsqueeze(0)
    prediction_euler = torch.stack(base.biwi_euler_deg_batch(prediction), dim=-1)[0]
    target_euler = torch.stack(base.biwi_euler_deg_batch(target), dim=-1)[0]
    angular = base.angular_error(target_euler, prediction_euler).numpy()
    translation_abs = np.abs(translation_vec - gt_translation)
    return angular, translation_abs, float(np.linalg.norm(translation_vec - gt_translation))


def local_box_from_xyxy(box):
    return [float(box[0]), float(box[2]), float(box[1]), float(box[3])]


def load_bank(bank_root, subject, bank_name):
    bank_dir = bank_root / f"subject{subject:02d}" / bank_name
    data = json.loads((bank_dir / "anchors.json").read_text())
    tagged_anchors = []
    images_bgr = []
    boxes = []
    rotations = []
    translations_m = []
    for anchor in data["anchors"]:
        image = cv2.imread(str(bank_dir / anchor["image"]))
        if image is None:
            raise RuntimeError(f"Could not read anchor image {anchor['image']}")
        anchor = dict(anchor)
        anchor["bank_name"] = bank_name
        anchor["local_anchor_id"] = int(anchor["anchor_id"])
        anchor["base_depth_scale"] = data.get("schedule", {}).get("base_depth_scale", 1.0)
        anchor["base_depth_m"] = data.get("schedule", {}).get("base_depth_m")
        tagged_anchors.append(anchor)
        images_bgr.append(image)
        boxes.append(local_box_from_xyxy(anchor["head_box_xyxy"]))
        rotations.append(np.asarray(anchor["rotation_h2c_opencv"], dtype=np.float64))
        translations_m.append(np.asarray(anchor["translation_h2c_m"], dtype=np.float64))
    data["bank_name"] = bank_name
    data["anchors"] = tagged_anchors
    return data, images_bgr, boxes, np.asarray(rotations), np.asarray(translations_m)


def load_anchor_pool(bank_root, subject, bank_names):
    combined = {
        "bank_names": list(bank_names),
        "anchors": [],
        "source_banks": [],
    }
    all_images = []
    all_boxes = []
    all_rotations = []
    all_translations = []
    for bank_name in bank_names:
        bank, images, boxes, rotations, translations = load_bank(
            bank_root,
            subject,
            bank_name,
        )
        combined["source_banks"].append(
            {
                "bank_name": bank_name,
                "anchors_json": str(
                    bank_root / f"subject{subject:02d}" / bank_name / "anchors.json"
                ),
                "anchor_count": len(bank["anchors"]),
                "schedule": bank.get("schedule", {}),
            }
        )
        for anchor in bank["anchors"]:
            anchor = dict(anchor)
            anchor["global_anchor_id"] = len(combined["anchors"])
            anchor["anchor_id"] = len(combined["anchors"])
            combined["anchors"].append(anchor)
        all_images.extend(images)
        all_boxes.extend(boxes)
        all_rotations.append(rotations)
        all_translations.append(translations)
    if not all_images:
        raise RuntimeError("No anchors loaded")
    return (
        combined,
        all_images,
        all_boxes,
        np.concatenate(all_rotations, axis=0),
        np.concatenate(all_translations, axis=0),
    )


def save_montage(path, query_bgr, anchor_images, bank, inlier_ids):
    width, height = 240, 180
    label_height = 34
    inlier_set = set(int(index) for index in inlier_ids)
    panels = [cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)] + [
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in anchor_images
    ]
    labels = ["query"]
    for anchor in bank["anchors"]:
        marker = "IN" if anchor["anchor_id"] in inlier_set else "OUT"
        translation_vec = anchor["translation_h2c_m"]
        labels.append(
            f"a{anchor['anchor_id']:02d} {marker} "
            f"t=[{translation_vec[0]:+.2f},{translation_vec[1]:+.2f},{translation_vec[2]:.2f}]"
        )
    columns = 5
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + label_height)
        resized = Image.fromarray(panel).resize((width, height))
        canvas.paste(resized, (x, y))
        draw.text((x + 3, y + height + 3), label, fill="black")
    canvas.save(path)


def summarize(rows, expected_queries):
    if not rows:
        return {}
    fields = (
        "yaw_error_deg",
        "pitch_error_deg",
        "roll_error_deg",
        "rot_mean_error_deg",
        "tx_error_mm",
        "ty_error_mm",
        "tz_error_mm",
        "translation_l2_error_mm",
        "ransac_inliers",
        "inference_runtime_s",
    )
    summary = {
        "expected_queries": expected_queries,
        "queries": len(rows),
        "coverage": len(rows) / max(expected_queries, 1),
        "ransac_success_rate": float(np.mean([row["ransac_success"] for row in rows])),
    }
    for field in fields:
        summary[field] = float(np.mean([row[field] for row in rows]))
    return summary


def evaluate(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    torch.manual_seed(0)
    np.random.seed(0)

    bank_names = [name.strip() for name in args.bank_name.split(",") if name.strip()]
    bank, anchor_images, anchor_boxes, anchor_rot_rgb, anchor_trans_rgb_m = load_anchor_pool(
        args.bank_root, args.subject, bank_names
    )
    model = base.load_vggt_model(
        str(args.base_checkpoint),
        str(args.lora_checkpoint),
        device,
        pose_encoding_type=args.pose_encoding_type,
    )
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, device=device)
    subject_dir = args.biwi_dir / f"{args.subject:02d}"
    frames = base.get_subject_frames(str(subject_dir))
    if args.frame_selection == "post_enrollment":
        frames = frames[1:]
    elif args.frame_selection == "enrollment_only":
        frames = frames[:1]
    if args.frame_stride > 1:
        frames = frames[:: args.frame_stride]
    if args.max_queries > 0:
        frames = frames[: args.max_queries]
    intrinsics, rgb_rotation, rgb_translation = base.load_rgb_intrinsics(
        str(subject_dir / "rgb.cal")
    )
    rgb_rotation_tensor = torch.from_numpy(rgb_rotation).float()
    rgb_translation_tensor = torch.from_numpy(rgb_translation).float()

    anchor_depth_rotations = []
    anchor_depth_translations = []
    for rotation_rgb, translation_rgb_m in zip(anchor_rot_rgb, anchor_trans_rgb_m):
        rotation_depth = rgb_rotation.T @ rotation_rgb
        translation_depth = rgb_rotation.T @ (
            translation_rgb_m * 1000.0 - rgb_translation
        )
        anchor_depth_rotations.append(torch.from_numpy(rotation_depth).float())
        anchor_depth_translations.append(torch.from_numpy(translation_depth).float())

    LOG.info(
        "Subject %02d: %d queries, %d fixed enrollment anchors from %d bank(s): %s",
        args.subject,
        len(frames),
        len(anchor_images),
        len(bank_names),
        ",".join(bank_names),
    )
    rows = []
    pair_rows = []
    previous_query_box = None
    for query_index, (frame_num, rgb_path, pose_path) in enumerate(frames):
        query_start = time.perf_counter()
        query_bgr = cv2.imread(rgb_path)
        if query_bgr is None:
            LOG.warning("Could not read %s", rgb_path)
            continue
        _, previous_query_box = base.crop_face_mtcnn(
            query_bgr,
            mtcnn,
            previous_query_box,
            crop_size=args.crop_size,
            ad=args.mtcnn_ad,
        )

        predicted_rotations = []
        predicted_translations = []
        inference_start = time.perf_counter()
        for start in range(0, len(anchor_images), args.pair_batch_size):
            end = min(start + args.pair_batch_size, len(anchor_images))
            anchor_tensors = []
            query_tensors = []
            for anchor_id in range(start, end):
                pair_box = _union_mtcnn_boxes(anchor_boxes[anchor_id], previous_query_box)
                anchor_crop = _square_box_rgb(
                    anchor_images[anchor_id], pair_box, args.crop_size
                )
                query_crop = _square_box_rgb(query_bgr, pair_box, args.crop_size)
                anchor_tensors.append(base.vggt_transform(anchor_crop))
                query_tensors.append(base.vggt_transform(query_crop))
            images = torch.stack(
                [torch.stack(anchor_tensors), torch.stack(query_tensors)], dim=1
            ).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                output = model(images=images)
            variants, translations = translation.decode_vggt_predictions(
                output["pose_enc"],
                torch.stack(anchor_depth_rotations[start:end]),
                torch.stack(anchor_depth_translations[start:end]),
                rgb_rotation_tensor,
                rgb_translation_tensor,
                pose_mode="relative_h2c",
                translation_scale=1000.0,
                pose_encoding_type=args.pose_encoding_type,
                translation_rotation_mode="predicted",
                translation_target_type=args.translation_target_type,
            )
            predicted_rotations.extend(variants["E_h2c_correct"].cpu().numpy())
            predicted_translations.extend(translations.cpu().numpy())
        inference_runtime = time.perf_counter() - inference_start

        fused_rotation, fused_translation, inlier_ids, success, consensus_size = ransac_fuse(
            predicted_rotations,
            predicted_translations,
            args.ransac_rot_threshold_deg,
            args.ransac_trans_threshold_mm,
            args.ransac_min_inliers,
        )
        gt_rotation, gt_translation = base.parse_pose_txt(pose_path)
        angle_error, translation_error, l2_error = biwi_errors(
            fused_rotation, fused_translation, gt_rotation, gt_translation
        )
        row = {
            "subject": f"{args.subject:02d}",
            "frame_num": frame_num,
            "anchors": len(anchor_images),
            "ransac_inliers": len(inlier_ids),
            "ransac_consensus_size": consensus_size,
            "ransac_success": int(success),
            "inlier_ids": " ".join(str(int(index)) for index in inlier_ids),
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
            "inference_runtime_s": inference_runtime,
            "total_runtime_s": time.perf_counter() - query_start,
        }
        rows.append(row)
        inlier_set = set(int(index) for index in inlier_ids)
        for anchor_id, (rotation_pred, translation_pred) in enumerate(
            zip(predicted_rotations, predicted_translations)
        ):
            pair_angle, pair_translation, pair_l2 = biwi_errors(
                rotation_pred, translation_pred, gt_rotation, gt_translation
            )
            pair_rows.append(
                {
                    "subject": f"{args.subject:02d}",
                    "frame_num": frame_num,
                    "anchor_id": anchor_id,
                    "bank_name": bank["anchors"][anchor_id].get("bank_name", ""),
                    "local_anchor_id": bank["anchors"][anchor_id].get("local_anchor_id", ""),
                    "base_depth_scale": bank["anchors"][anchor_id].get("base_depth_scale", ""),
                    "base_depth_m": bank["anchors"][anchor_id].get("base_depth_m", ""),
                    "inlier": int(anchor_id in inlier_set),
                    "yaw_error_deg": float(pair_angle[0]),
                    "pitch_error_deg": float(pair_angle[1]),
                    "roll_error_deg": float(pair_angle[2]),
                    "rot_mean_error_deg": float(pair_angle.mean()),
                    "tx_error_mm": float(pair_translation[0]),
                    "ty_error_mm": float(pair_translation[1]),
                    "tz_error_mm": float(pair_translation[2]),
                    "translation_l2_error_mm": pair_l2,
                    "pred_tx_mm": float(translation_pred[0]),
                    "pred_ty_mm": float(translation_pred[1]),
                    "pred_tz_mm": float(translation_pred[2]),
                    "pred_r00": float(rotation_pred[0, 0]),
                    "pred_r01": float(rotation_pred[0, 1]),
                    "pred_r02": float(rotation_pred[0, 2]),
                    "pred_r10": float(rotation_pred[1, 0]),
                    "pred_r11": float(rotation_pred[1, 1]),
                    "pred_r12": float(rotation_pred[1, 2]),
                    "pred_r20": float(rotation_pred[2, 0]),
                    "pred_r21": float(rotation_pred[2, 1]),
                    "pred_r22": float(rotation_pred[2, 2]),
                    "anchor_direction_x": float(
                        bank["anchors"][anchor_id]["icosphere_camera_direction"][0]
                    ),
                    "anchor_direction_y": float(
                        bank["anchors"][anchor_id]["icosphere_camera_direction"][1]
                    ),
                    "anchor_direction_z": float(
                        bank["anchors"][anchor_id]["icosphere_camera_direction"][2]
                    ),
                }
            )
        if query_index < args.max_visualizations:
            save_montage(
                args.out_dir / f"subject_{args.subject:02d}_frame_{frame_num:05d}_anchors.png",
                query_bgr,
                anchor_images,
                bank,
                inlier_ids,
            )
        LOG.info(
            "[%02d/%05d] inliers=%d/%d rot=%.3fdeg trans=%.3fmm runtime=%.1fs",
            args.subject,
            frame_num,
            len(inlier_ids),
            len(anchor_images),
            angle_error.mean(),
            l2_error,
            time.perf_counter() - query_start,
        )

    if rows:
        with (args.out_dir / f"subject_{args.subject:02d}_predictions.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (args.out_dir / f"subject_{args.subject:02d}_pairwise.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
            writer.writeheader()
            writer.writerows(pair_rows)
    summary = summarize(rows, len(frames))
    if summary:
        with (args.out_dir / f"subject_{args.subject:02d}_summary.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
    metadata = {
        "subject": f"{args.subject:02d}",
        "bank_names": bank_names,
        "source_banks": bank.get("source_banks", []),
        "uses_query_pose_or_depth_as_model_input": False,
        "uses_biwi_rgb_calibration_for_reporting_frame_only": True,
        "frame_selection": args.frame_selection,
        "excludes_enrollment_frame_from_queries": args.frame_selection == "post_enrollment",
        "summary": summary,
    }
    (args.out_dir / f"subject_{args.subject:02d}_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    LOG.info("Summary: %s", summary)


if __name__ == "__main__":
    evaluate(parse_args())
