#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = PROJECT_ROOT / "evaluation" / "eval_scripts"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_biwi_crop as base 
import eval_biwi_crop_mtcnn_square_translation as vggt_translation 


DEFAULT_FAMOS_ROOT = Path("data/famos_head_pose_test")
DEFAULT_OUT_ROOT = Path("outputs/eval_famos_head_pose")
DEFAULT_EXP = "flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit"
HEAD_YZ_FLIP = np.diag([1.0, -1.0, -1.0])


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="FaMoS GT-anchor VGGT-HPE relative evaluation")
    parser.add_argument("--famos-root", type=Path, default=DEFAULT_FAMOS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "VGGT-1B" / "model.pt",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "training" / "logs" / DEFAULT_EXP / "ckpts" / "checkpoint_410.pt",
    )
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument(
        "--translation-target-type",
        choices=("relative_se3", "delta_t", "depth_norm_delta_t", "projective_delta_t"),
        default="depth_norm_delta_t",
    )
    parser.add_argument("--translation-scale", type=float, default=1000.0)
    parser.add_argument(
        "--crop-schema",
        choices=("mtcnn_square_pair_union", "mtcnn_square_per_view", "full_frame_square"),
        default="mtcnn_square_pair_union",
    )
    parser.add_argument("--crop-posenc-dim", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--mtcnn-ad", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--views", type=int, nargs="*", default=None)
    parser.add_argument(
        "--subset-indices-csv",
        type=Path,
        default=None,
        help="Optional CSV containing label_index rows to evaluate.",
    )
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-targets-per-group", type=int, default=0)
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Convention for FaMoS GT/pred rotations. head_yz_flip uses R @ diag(1,-1,-1).",
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_labels_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.keys()}


def apply_famos_rotation_convention(rotation, convention):
    if convention == "raw":
        return rotation
    if convention == "head_yz_flip":
        return rotation @ HEAD_YZ_FLIP
    raise ValueError(f"Unknown FaMoS rotation convention: {convention}")


def load_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def full_frame_square_rgb(img_bgr, crop_size):
    img_h, img_w = img_bgr.shape[:2]
    side = max(img_w, img_h)
    canvas = np.zeros((side, side, 3), dtype=img_bgr.dtype)
    y0 = (side - img_h) // 2
    x0 = (side - img_w) // 2
    canvas[y0 : y0 + img_h, x0 : x0 + img_w] = img_bgr
    crop_bgr = cv2.resize(canvas, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def square_box_rgb(img_bgr, box, crop_size):
    x1, x2, y1, y2 = [int(v) for v in box]
    img_h, img_w = img_bgr.shape[:2]
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    side = max(box_w, box_h)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = sx1 + side
    sy2 = sy1 + side
    src_x1 = max(0, sx1)
    src_y1 = max(0, sy1)
    src_x2 = min(img_w, sx2)
    src_y2 = min(img_h, sy2)
    canvas = np.zeros((side, side, 3), dtype=img_bgr.dtype)
    dst_x1 = src_x1 - sx1
    dst_y1 = src_y1 - sy1
    canvas[dst_y1 : dst_y1 + (src_y2 - src_y1), dst_x1 : dst_x1 + (src_x2 - src_x1)] = img_bgr[
        src_y1:src_y2, src_x1:src_x2
    ]
    crop_bgr = cv2.resize(canvas, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)


def identity_crop_params(img_bgr):
    return torch.tensor([0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0], dtype=torch.float32)


def square_box_crop_params(img_bgr, box):
    x1, x2, y1, y2 = [float(v) for v in box]
    img_h, img_w = img_bgr.shape[:2]
    box_w = max(1.0, x2 - x1 + 1.0)
    box_h = max(1.0, y2 - y1 + 1.0)
    side = max(box_w, box_h)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    sq_x1 = cx - side / 2.0
    sq_y1 = cy - side / 2.0
    return torch.tensor(
        [
            cx / img_w,
            cy / img_h,
            side / img_w,
            side / img_h,
            (sq_x1 + side / 2.0) / img_w,
            (sq_y1 + side / 2.0) / img_h,
            side / img_w,
            side / img_h,
        ],
        dtype=torch.float32,
    )


def union_boxes(box_a, box_b):
    return [
        min(int(box_a[0]), int(box_b[0])),
        max(int(box_a[1]), int(box_b[1])),
        min(int(box_a[2]), int(box_b[2])),
        max(int(box_a[3]), int(box_b[3])),
    ]


def geodesic_deg_torch(R_pred, R_gt):
    rel = R_pred @ R_gt.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    return torch.rad2deg(torch.acos(cos))


def load_subset_indices(path):
    if path is None:
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Subset CSV is empty: {path}")
    field = "label_index" if "label_index" in rows[0] else "index"
    return {int(row[field]) for row in rows}


def group_indices(labels, subject_filter, sequence_filter, view_filter, allowed_indices=None):
    groups = defaultdict(list)
    n = len(labels["subject"])
    for idx in range(n):
        if allowed_indices is not None and idx not in allowed_indices:
            continue
        subject = str(labels["subject"][idx])
        sequence = str(labels["sequence"][idx])
        view = int(labels["view_idx"][idx])
        if subject_filter and subject not in subject_filter:
            continue
        if sequence_filter and sequence not in sequence_filter:
            continue
        if view_filter and view not in view_filter:
            continue
        groups[(subject, sequence, view)].append(idx)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=lambda i: int(str(labels["frame"][i])))
    return dict(sorted(groups.items()))


def init_rot_acc():
    return {"count": 0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "maer": 0.0, "geodesic": 0.0}


def init_trans_acc():
    return {"count": 0, "tx": 0.0, "ty": 0.0, "tz": 0.0, "l2": 0.0}


def add_rot(acc, yaw_err, pitch_err, roll_err, geodesic):
    n = len(geodesic)
    acc["count"] += n
    acc["yaw"] += float(yaw_err.sum().item())
    acc["pitch"] += float(pitch_err.sum().item())
    acc["roll"] += float(roll_err.sum().item())
    acc["maer"] += float(((yaw_err + pitch_err + roll_err) / 3.0).sum().item())
    acc["geodesic"] += float(geodesic.sum().item())


def add_trans(acc, pred_t, gt_t):
    err = (pred_t - gt_t).abs()
    acc["count"] += pred_t.shape[0]
    acc["tx"] += float(err[:, 0].sum().item())
    acc["ty"] += float(err[:, 1].sum().item())
    acc["tz"] += float(err[:, 2].sum().item())
    acc["l2"] += float((pred_t - gt_t).norm(dim=-1).sum().item())


def metric_rows(rot_accs, trans_accs):
    rows = []
    for key in sorted(rot_accs):
        rot = rot_accs[key]
        n = rot["count"]
        if n <= 0:
            continue
        row = {
            "scope": key[0],
            "subject": key[1],
            "sequence": key[2],
            "view": key[3],
            "variant": key[4],
            "frames": n,
            "yaw_mae_deg": f"{rot['yaw'] / n:.6f}",
            "pitch_mae_deg": f"{rot['pitch'] / n:.6f}",
            "roll_mae_deg": f"{rot['roll'] / n:.6f}",
            "maer_deg": f"{rot['maer'] / n:.6f}",
            "geodesic_mean_deg": f"{rot['geodesic'] / n:.6f}",
        }
        trans = trans_accs.get(key)
        if trans and trans["count"] > 0:
            tn = trans["count"]
            row.update(
                {
                    "tx_mae_mm": f"{trans['tx'] / tn:.6f}",
                    "ty_mae_mm": f"{trans['ty'] / tn:.6f}",
                    "tz_mae_mm": f"{trans['tz'] / tn:.6f}",
                    "translation_l2_mae_mm": f"{trans['l2'] / tn:.6f}",
                }
            )
        else:
            row.update(
                {
                    "tx_mae_mm": "",
                    "ty_mae_mm": "",
                    "tz_mae_mm": "",
                    "translation_l2_mae_mm": "",
                }
            )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    labels = load_labels_npz(args.famos_root / "labels.npz")
    csv_rows = load_csv_rows(args.famos_root / "labels.csv")
    if len(csv_rows) != len(labels["subject"]):
        raise RuntimeError("labels.csv and labels.npz have different row counts")

    allowed_indices = load_subset_indices(args.subset_indices_csv)
    groups = group_indices(
        labels,
        set(args.subjects) if args.subjects else None,
        set(args.sequences) if args.sequences else None,
        set(args.views) if args.views is not None else None,
        allowed_indices,
    )
    if args.max_groups > 0:
        groups = dict(list(groups.items())[: args.max_groups])
    if not groups:
        raise RuntimeError("No FaMoS groups selected")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    logging.info("Device: %s | AMP: %s", device, amp_dtype)
    logging.info("Groups: %d", len(groups))

    model = base.load_vggt_model(
        str(args.base_checkpoint),
        None if args.no_lora else str(args.lora_checkpoint),
        device,
        pose_encoding_type=args.pose_encoding_type,
        crop_posenc_dim=args.crop_posenc_dim,
    )

    from facenet_pytorch import MTCNN as FacenetMTCNN

    mtcnn = FacenetMTCNN(keep_all=True, device=device)
    logging.info("MTCNN detector initialized")

    R_cam_t = torch.eye(3, dtype=torch.float32)
    t_cam_t = torch.zeros(3, dtype=torch.float32)
    primary_variant = base.get_primary_variant_name("relative_h2c")
    rot_accs = defaultdict(init_rot_acc)
    trans_accs = defaultdict(init_trans_acc)
    prediction_rows = []

    total_pairs = 0
    for group_num, ((subject, sequence, view), indices) in enumerate(groups.items(), start=1):
        if len(indices) < 2:
            continue
        anchor_idx = indices[0]
        target_indices = indices[1:: max(1, args.sample_stride)]
        if args.max_targets_per_group > 0:
            target_indices = target_indices[: args.max_targets_per_group]
        if not target_indices:
            continue
        total_pairs += len(target_indices)

        logging.info(
            "Group %d/%d: %s %s view%02d anchor=%s targets=%d",
            group_num,
            len(groups),
            subject,
            sequence,
            view,
            labels["frame"][anchor_idx],
            len(target_indices),
        )

        anchor_bgr = load_bgr(args.famos_root / str(labels["image_relpath"][anchor_idx]))
        R_anchor_np = apply_famos_rotation_convention(labels["R_cam_head"][anchor_idx], args.famos_rotation_convention)
        R_anchor = torch.from_numpy(R_anchor_np).float()
        t_anchor = torch.from_numpy(labels["t_cam_head_mm"][anchor_idx]).float()

        anchor_tensor_static = None
        anchor_crop_params_static = None
        anchor_box = None
        if args.crop_schema == "full_frame_square":
            anchor_tensor_static = base.vggt_transform(full_frame_square_rgb(anchor_bgr, args.crop_size))
            anchor_crop_params_static = identity_crop_params(anchor_bgr)
        else:
            _, anchor_box = base.crop_face_mtcnn(
                anchor_bgr,
                mtcnn,
                None,
                crop_size=args.crop_size,
                ad=args.mtcnn_ad,
            )
            if args.crop_schema == "mtcnn_square_per_view":
                anchor_tensor_static = base.vggt_transform(square_box_rgb(anchor_bgr, anchor_box, args.crop_size))
                anchor_crop_params_static = square_box_crop_params(anchor_bgr, anchor_box)

        prev_target_box = None
        for start in range(0, len(target_indices), args.batch_size):
            batch_indices = target_indices[start : start + args.batch_size]
            anchor_tensors = []
            target_tensors = []
            anchor_crop_params = []
            target_crop_params = []
            kept_indices = []

            for idx in batch_indices:
                target_bgr = load_bgr(args.famos_root / str(labels["image_relpath"][idx]))
                if args.crop_schema == "full_frame_square":
                    target_tensors.append(base.vggt_transform(full_frame_square_rgb(target_bgr, args.crop_size)))
                    target_crop_params.append(identity_crop_params(target_bgr))
                else:
                    _, prev_target_box = base.crop_face_mtcnn(
                        target_bgr,
                        mtcnn,
                        prev_target_box,
                        crop_size=args.crop_size,
                        ad=args.mtcnn_ad,
                    )
                    if args.crop_schema == "mtcnn_square_pair_union":
                        pair_box = union_boxes(anchor_box, prev_target_box)
                        anchor_tensors.append(base.vggt_transform(square_box_rgb(anchor_bgr, pair_box, args.crop_size)))
                        anchor_crop_params.append(square_box_crop_params(anchor_bgr, pair_box))
                        target_tensors.append(base.vggt_transform(square_box_rgb(target_bgr, pair_box, args.crop_size)))
                        target_crop_params.append(square_box_crop_params(target_bgr, pair_box))
                    else:
                        target_tensors.append(base.vggt_transform(square_box_rgb(target_bgr, prev_target_box, args.crop_size)))
                        target_crop_params.append(square_box_crop_params(target_bgr, prev_target_box))
                kept_indices.append(idx)

            if not target_tensors:
                continue

            bs = len(target_tensors)
            target_stack = torch.stack(target_tensors)
            target_crop_stack = torch.stack(target_crop_params)
            if args.crop_schema == "mtcnn_square_pair_union":
                anchor_stack = torch.stack(anchor_tensors)
                anchor_crop_stack = torch.stack(anchor_crop_params)
            else:
                anchor_stack = anchor_tensor_static.unsqueeze(0).expand(bs, -1, -1, -1)
                anchor_crop_stack = anchor_crop_params_static.unsqueeze(0).expand(bs, -1)

            images = torch.stack([anchor_stack, target_stack], dim=1).to(device)
            crop_params = torch.stack([anchor_crop_stack, target_crop_stack], dim=1).to(device)
            R_gt_np = apply_famos_rotation_convention(labels["R_cam_head"][kept_indices], args.famos_rotation_convention)
            R_gt = torch.from_numpy(R_gt_np).float()
            t_gt = torch.from_numpy(labels["t_cam_head_mm"][kept_indices]).float()

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                kwargs = {"images": images}
                if args.crop_posenc_dim > 0:
                    kwargs["crop_params"] = crop_params
                out = model(**kwargs)

            decoded = vggt_translation.decode_vggt_predictions(
                out["pose_enc"],
                R_anchor,
                t_anchor,
                R_cam_t,
                t_cam_t,
                R_target_t=R_gt,
                pose_mode="relative_h2c",
                translation_scale=args.translation_scale,
                pose_encoding_type=args.pose_encoding_type,
                translation_rotation_mode="predicted",
                translation_target_type=args.translation_target_type,
            )
            variants, t_primary = decoded
            y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(R_gt)

            for variant, R_pred in variants.items():
                y_pred, p_pred, r_pred = base.biwi_euler_deg_batch(R_pred)
                yaw_err = base.angular_error(y_gt, y_pred)
                pitch_err = base.angular_error(p_gt, p_pred)
                roll_err = base.angular_error(r_gt, r_pred)
                geod = geodesic_deg_torch(R_pred, R_gt)
                keys = [
                    ("overall", "all", "all", "all", variant),
                    ("subject", subject, "all", "all", variant),
                    ("sequence", "all", sequence, "all", variant),
                    ("view", "all", "all", f"{view:02d}", variant),
                    ("subject_sequence_view", subject, sequence, f"{view:02d}", variant),
                ]
                for key in keys:
                    add_rot(rot_accs[key], yaw_err, pitch_err, roll_err, geod)
                    if variant == primary_variant:
                        add_trans(trans_accs[key], t_primary, t_gt)

                if args.save_predictions:
                    R_pred_np = R_pred.detach().cpu().numpy()
                    t_pred_np = t_primary.detach().cpu().numpy()
                    R_gt_np = R_gt.detach().cpu().numpy()
                    t_gt_np = t_gt.detach().cpu().numpy()
                    for local_i, idx in enumerate(kept_indices):
                        row = {
                            "subject": subject,
                            "sequence": sequence,
                            "view": view,
                            "anchor_frame": str(labels["frame"][anchor_idx]),
                            "frame": str(labels["frame"][idx]),
                            "variant": variant,
                            "yaw_error_deg": f"{float(yaw_err[local_i]):.6f}",
                            "pitch_error_deg": f"{float(pitch_err[local_i]):.6f}",
                            "roll_error_deg": f"{float(roll_err[local_i]):.6f}",
                            "rot_mean_error_deg": f"{float((yaw_err[local_i] + pitch_err[local_i] + roll_err[local_i]) / 3.0):.6f}",
                            "geodesic_error_deg": f"{float(geod[local_i]):.6f}",
                        }
                        for rr in range(3):
                            for cc in range(3):
                                row[f"pred_r{rr}{cc}"] = f"{R_pred_np[local_i, rr, cc]:.9f}"
                                row[f"gt_r{rr}{cc}"] = f"{R_gt_np[local_i, rr, cc]:.9f}"
                        if variant == primary_variant:
                            err = np.abs(t_pred_np[local_i] - t_gt_np[local_i])
                            row.update(
                                {
                                    "pred_tx_mm": f"{t_pred_np[local_i, 0]:.6f}",
                                    "pred_ty_mm": f"{t_pred_np[local_i, 1]:.6f}",
                                    "pred_tz_mm": f"{t_pred_np[local_i, 2]:.6f}",
                                    "gt_tx_mm": f"{t_gt_np[local_i, 0]:.6f}",
                                    "gt_ty_mm": f"{t_gt_np[local_i, 1]:.6f}",
                                    "gt_tz_mm": f"{t_gt_np[local_i, 2]:.6f}",
                                    "tx_error_mm": f"{err[0]:.6f}",
                                    "ty_error_mm": f"{err[1]:.6f}",
                                    "tz_error_mm": f"{err[2]:.6f}",
                                    "translation_l2_error_mm": f"{float(np.linalg.norm(t_pred_np[local_i] - t_gt_np[local_i])):.6f}",
                                }
                            )
                        prediction_rows.append(row)

    summary_rows = metric_rows(rot_accs, trans_accs)
    summary_path = args.out_root / "famos_gt_anchor_summary.csv"
    write_csv(summary_path, summary_rows)
    if args.save_predictions:
        write_csv(args.out_root / "famos_gt_anchor_predictions.csv", prediction_rows)

    metadata = {
        "protocol": "FaMoS same-view first-frame GT-anchor VGGT-HPE relative evaluation",
        "famos_root": str(args.famos_root),
        "base_checkpoint": str(args.base_checkpoint),
        "lora_checkpoint": None if args.no_lora else str(args.lora_checkpoint),
        "crop_schema": args.crop_schema,
        "translation_target_type": args.translation_target_type,
        "translation_scale": args.translation_scale,
        "famos_rotation_convention": args.famos_rotation_convention,
        "groups": len(groups),
        "pairs": total_pairs,
        "subjects": args.subjects,
        "sequences": args.sequences,
        "views": args.views,
        "subset_indices_csv": None if args.subset_indices_csv is None else str(args.subset_indices_csv),
        "subset_indices": None if allowed_indices is None else len(allowed_indices),
        "sample_stride": args.sample_stride,
        "max_groups": args.max_groups,
        "max_targets_per_group": args.max_targets_per_group,
        "note": "Final AvatarBox hard-gate refinement is not run because FaMoS export lacks 68 landmark/FLAME enrollment alignment assets.",
    }
    (args.out_root / "famos_gt_anchor_metadata.json").write_text(json.dumps(metadata, indent=2))

    overall = [
        row for row in summary_rows
        if row["scope"] == "overall" and row["variant"] == primary_variant
    ]
    md_path = args.out_root / "famos_gt_anchor_summary.md"
    with md_path.open("w") as handle:
        handle.write("# FaMoS GT-Anchor VGGT-HPE Evaluation\n\n")
        handle.write(
            "Protocol: first frame per subject/sequence/view is the GT anchor, "
            "queries are later frames from the same camera view.\n\n"
        )
        handle.write("## Overall Primary Variant\n\n")
        handle.write("| variant | frames | yaw | pitch | roll | MAEr | GE | Tx | Ty | Tz | Trans L2 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in overall:
            handle.write(
                f"| {row['variant']} | {row['frames']} | "
                f"{float(row['yaw_mae_deg']):.3f} | {float(row['pitch_mae_deg']):.3f} | "
                f"{float(row['roll_mae_deg']):.3f} | {float(row['maer_deg']):.3f} | "
                f"{float(row['geodesic_mean_deg']):.3f} | "
                f"{float(row['tx_mae_mm']):.3f} | {float(row['ty_mae_mm']):.3f} | "
                f"{float(row['tz_mae_mm']):.3f} | {float(row['translation_l2_mae_mm']):.3f} |\n"
            )
        handle.write("\n## Files\n\n")
        handle.write("- `famos_gt_anchor_summary.csv`\n")
        if args.save_predictions:
            handle.write("- `famos_gt_anchor_predictions.csv`\n")
        handle.write("- `famos_gt_anchor_metadata.json`\n")

    print(md_path)


if __name__ == "__main__":
    main()
