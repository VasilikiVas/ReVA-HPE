#!/usr/bin/env python3
"""Validate multi-anchor fusion on synthetic identities excluded from training."""

import argparse
import csv
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "eval_scripts"))
sys.path.insert(0, str(ROOT))

import eval_biwi_crop as base 
import eval_biwi_crop_mtcnn_square_translation as translation  
from eval_biwi_crop_mtcnn_square_compare import _square_box_rgb  
from eval_biwi_facelift_metric_anchor_bank import (  
    biwi_errors,
    ransac_fuse,
)


LOG = logging.getLogger("synthetic_anchor_fusion")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--identity", action="append", default=[])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-identities", type=int, default=5)
    parser.add_argument("--lighting", default="lighting_000")
    parser.add_argument("--query-expression", default="expr_009")
    parser.add_argument("--anchor-count", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--bbox-ad", type=float, default=0.1)
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument("--translation-target-type", default="depth_norm_delta_t")
    parser.add_argument("--ransac-rot-threshold-deg", type=float, default=10.0)
    parser.add_argument("--ransac-trans-threshold-mm", type=float, default=50.0)
    parser.add_argument("--ransac-min-inliers", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_h2c(view_dir):
    with (view_dir / "output_opencv_camera.pkl").open("rb") as handle:
        camera = pickle.load(handle)
    with (view_dir / "output_object_transform_post_render.pkl").open("rb") as handle:
        head = pickle.load(handle)
    rotation_w2c = np.asarray(camera["R"], dtype=np.float64)
    translation_w2c = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    rotation_h2w = np.asarray(head["R_world"], dtype=np.float64)
    translation_h2w = np.asarray(head["t_world"], dtype=np.float64).reshape(3)
    rotation = rotation_w2c @ rotation_h2w
    translation_vec = rotation_w2c @ translation_h2w + translation_w2c
    return rotation, translation_vec


def load_view(view_dir):
    image = cv2.imread(str(view_dir / "output.png"))
    if image is None:
        raise RuntimeError(f"Could not read {view_dir / 'output.png'}")
    bbox = np.load(view_dir / "face_bbox.npy").astype(np.float64).reshape(-1)[:4]
    rotation, translation_vec = load_h2c(view_dir)
    camera_center = -rotation.T @ translation_vec
    direction = camera_center / np.linalg.norm(camera_center)
    return {
        "path": view_dir,
        "image": image,
        "bbox": bbox,
        "rotation": rotation,
        "translation_m": translation_vec,
        "direction": direction,
    }


def heldout_identity_dirs(args):
    identity_dirs = sorted(args.synthetic_root.glob("identity_*"))
    if args.identity:
        by_name = {path.name: path for path in identity_dirs}
        missing = [name for name in args.identity if name not in by_name]
        if missing:
            raise FileNotFoundError(f"Missing synthetic identities: {missing}")
        return [by_name[name] for name in args.identity]
    names = [path.name for path in identity_dirs]
    rng = np.random.RandomState(args.split_seed)
    rng.shuffle(names)
    selected = set(names[: min(args.val_identities, len(names))])
    return [path for path in identity_dirs if path.name in selected]


def farthest_direction_subset(views, count):
    if len(views) < count:
        raise RuntimeError(f"Only {len(views)} anchor candidates for requested bank of {count}")
    directions = np.stack([view["direction"] for view in views])
    canonical = directions.mean(axis=0)
    canonical /= np.linalg.norm(canonical)
    selected = [int(np.argmax(directions @ canonical))]
    min_distance = np.full(len(views), np.inf)
    while len(selected) < count:
        distance = 1.0 - directions @ directions[selected[-1]]
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
        selected.append(int(np.argmax(min_distance)))
    return [views[index] for index in selected]


def expanded_pair_box(first, second, image_shape, ad):
    xmin = min(float(first[0]), float(second[0]))
    xmax = max(float(first[1]), float(second[1]))
    ymin = min(float(first[2]), float(second[2]))
    ymax = max(float(first[3]), float(second[3]))
    width = xmax - xmin
    height = ymax - ymin
    x1, x2 = xmin - ad * width, xmax + ad * width
    y1, y2 = ymin - ad * height, ymax + ad * height
    side = max(x2 - x1, y2 - y1)
    center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    x1, x2 = center_x - 0.5 * side, center_x + 0.5 * side
    y1, y2 = center_y - 0.5 * side, center_y + 0.5 * side
    image_h, image_w = image_shape[:2]
    col_start = max(int(math.floor(x1)), 0)
    col_end = min(int(math.ceil(x2)), image_w - 1)
    row_start = max(int(math.floor(y1)), 0)
    row_end = min(int(math.ceil(y2)), image_h - 1)
    crop_width = col_end - col_start + 1
    crop_height = row_end - row_start + 1
    if crop_width > crop_height:
        col_start += (crop_width - crop_height) // 2
        col_end = col_start + crop_height - 1
    elif crop_height > crop_width:
        row_start += (crop_height - crop_width) // 2
        row_end = row_start + crop_width - 1
    return [col_start, col_end, row_start, row_end]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_query(args, model, device, amp_dtype, anchors, query):
    predicted_rotations = []
    predicted_translations = []
    identity_rotation = torch.eye(3)
    zero_translation = torch.zeros(3)
    for start in range(0, len(anchors), args.pair_batch_size):
        batch = anchors[start : start + args.pair_batch_size]
        anchor_tensors = []
        query_tensors = []
        for anchor in batch:
            box = expanded_pair_box(
                anchor["bbox"], query["bbox"], query["image"].shape, args.bbox_ad
            )
            anchor_tensors.append(
                base.vggt_transform(_square_box_rgb(anchor["image"], box, args.crop_size))
            )
            query_tensors.append(
                base.vggt_transform(_square_box_rgb(query["image"], box, args.crop_size))
            )
        images = torch.stack(
            [torch.stack(anchor_tensors), torch.stack(query_tensors)], dim=1
        ).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
            output = model(images=images)
        rotations = torch.from_numpy(np.stack([anchor["rotation"] for anchor in batch])).float()
        translations_mm = torch.from_numpy(
            np.stack([anchor["translation_m"] for anchor in batch]) * 1000.0
        ).float()
        variants, decoded_translation = translation.decode_vggt_predictions(
            output["pose_enc"],
            rotations,
            translations_mm,
            identity_rotation,
            zero_translation,
            pose_mode="relative_h2c",
            translation_scale=1000.0,
            pose_encoding_type=args.pose_encoding_type,
            translation_rotation_mode="predicted",
            translation_target_type=args.translation_target_type,
        )
        predicted_rotations.extend(variants["E_h2c_correct"].cpu().numpy())
        predicted_translations.extend(decoded_translation.cpu().numpy())
    return np.asarray(predicted_rotations), np.asarray(predicted_translations)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    torch.manual_seed(0)
    np.random.seed(0)
    model = base.load_vggt_model(
        str(args.base_checkpoint),
        str(args.lora_checkpoint),
        device,
        pose_encoding_type=args.pose_encoding_type,
    )

    combined_rows = []
    all_pair_rows = []
    identities = heldout_identity_dirs(args)
    for subject_index, identity_dir in enumerate(identities):
        lighting_dir = identity_dir / args.lighting
        query_dirs = sorted((lighting_dir / args.query_expression).glob("view_*"))
        query_dirs = query_dirs[: args.max_queries]
        anchor_dirs = sorted(lighting_dir.glob("expr_*/view_*"))
        anchor_dirs = [path for path in anchor_dirs if path.parent.name != args.query_expression]
        queries = [load_view(path) for path in query_dirs]
        anchors = farthest_direction_subset([load_view(path) for path in anchor_dirs], args.anchor_count)
        subject = f"{subject_index:02d}"
        LOG.info("%s: %d queries x %d anchors", identity_dir.name, len(queries), len(anchors))
        subject_rows = []
        subject_pair_rows = []
        for query_index, query in enumerate(queries):
            rotations, translations_mm = evaluate_query(
                args, model, device, amp_dtype, anchors, query
            )
            fused_rotation, fused_translation, inliers, success, consensus_size = ransac_fuse(
                rotations,
                translations_mm,
                args.ransac_rot_threshold_deg,
                args.ransac_trans_threshold_mm,
                args.ransac_min_inliers,
            )
            gt_rotation = query["rotation"]
            gt_translation = query["translation_m"] * 1000.0
            angle, trans_abs, trans_l2 = biwi_errors(
                fused_rotation, fused_translation, gt_rotation, gt_translation
            )
            row = {
                "subject": subject,
                "frame_num": query_index,
                "identity": identity_dir.name,
                "query_path": str(query["path"]),
                "anchors": len(anchors),
                "ransac_inliers": len(inliers),
                "ransac_consensus_size": consensus_size,
                "ransac_success": int(success),
                "inlier_ids": " ".join(str(int(index)) for index in inliers),
                "yaw_error_deg": float(angle[0]),
                "pitch_error_deg": float(angle[1]),
                "roll_error_deg": float(angle[2]),
                "rot_mean_error_deg": float(angle.mean()),
                "tx_error_mm": float(trans_abs[0]),
                "ty_error_mm": float(trans_abs[1]),
                "tz_error_mm": float(trans_abs[2]),
                "translation_l2_error_mm": trans_l2,
                "pred_tx_mm": float(fused_translation[0]),
                "pred_ty_mm": float(fused_translation[1]),
                "pred_tz_mm": float(fused_translation[2]),
                "gt_tx_mm": float(gt_translation[0]),
                "gt_ty_mm": float(gt_translation[1]),
                "gt_tz_mm": float(gt_translation[2]),
                "inference_runtime_s": 0.0,
                "total_runtime_s": 0.0,
            }
            subject_rows.append(row)
            combined_rows.append(row)
            inlier_set = set(int(index) for index in inliers)
            for anchor_id, (anchor, rotation_pred, translation_pred) in enumerate(
                zip(anchors, rotations, translations_mm)
            ):
                pair_angle, pair_abs, pair_l2 = biwi_errors(
                    rotation_pred, translation_pred, gt_rotation, gt_translation
                )
                pair_row = {
                    "subject": subject,
                    "frame_num": query_index,
                    "anchor_id": anchor_id,
                    "anchor_path": str(anchor["path"]),
                    "inlier": int(anchor_id in inlier_set),
                    "yaw_error_deg": float(pair_angle[0]),
                    "pitch_error_deg": float(pair_angle[1]),
                    "roll_error_deg": float(pair_angle[2]),
                    "rot_mean_error_deg": float(pair_angle.mean()),
                    "tx_error_mm": float(pair_abs[0]),
                    "ty_error_mm": float(pair_abs[1]),
                    "tz_error_mm": float(pair_abs[2]),
                    "translation_l2_error_mm": pair_l2,
                    "pred_tx_mm": float(translation_pred[0]),
                    "pred_ty_mm": float(translation_pred[1]),
                    "pred_tz_mm": float(translation_pred[2]),
                    **{
                        f"pred_r{row_idx}{column_idx}": float(rotation_pred[row_idx, column_idx])
                        for row_idx in range(3)
                        for column_idx in range(3)
                    },
                    "anchor_direction_x": float(anchor["direction"][0]),
                    "anchor_direction_y": float(anchor["direction"][1]),
                    "anchor_direction_z": float(anchor["direction"][2]),
                }
                subject_pair_rows.append(pair_row)
                all_pair_rows.append(pair_row)
        write_csv(args.output_dir / f"subject_{subject}_predictions.csv", subject_rows)
        write_csv(args.output_dir / f"subject_{subject}_pairwise.csv", subject_pair_rows)

    write_csv(args.output_dir / "combined_predictions.csv", combined_rows)
    metadata = {
        "synthetic_root": str(args.synthetic_root),
        "heldout_identities": [path.name for path in identities],
        "split_seed": args.split_seed,
        "query_expression": args.query_expression,
        "anchor_count": args.anchor_count,
        "queries": len(combined_rows),
        "uses_biwi_data": False,
        "uses_training_identities": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    LOG.info("Saved %d queries and %d pair predictions", len(combined_rows), len(all_pair_rows))


if __name__ == "__main__":
    main()
