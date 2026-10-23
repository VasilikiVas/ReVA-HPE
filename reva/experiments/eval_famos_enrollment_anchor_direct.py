#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "evaluation" / "eval_scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_biwi_crop as base  
import eval_biwi_crop_mtcnn_square_translation as vggt_translation  
from reva.experiments.build_cross_dataset_refinement_summary_table import (  
    convert_famos_rotation_convention,
    make_anchor_from_z,
)
from reva.experiments.eval_cross_dataset_avatarbox_median import (
    angular_metrics,
    avatarbox_median_depth,
    load_intrinsics,
    pnp_rotation_to_eval_head_frame,
    read_csv,
    row_matrix,
    row_translation,
    solve_enrollment_pnp,
    summarize,
    write_csv,
    write_json,
)
from reva.experiments.eval_famos_gt_anchor import square_box_rgb, union_boxes 


OUT_ROOT = Path("outputs")
DEFAULT_EXP = "flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_view00_mtcnn_visible_no091")
    parser.add_argument("--method-prefix", default="FaMoS")
    parser.add_argument(
        "--ready-root",
        type=Path,
        default=OUT_ROOT / "refinement_inputs_cross_dataset",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=OUT_ROOT / "famos_baseline_predictions",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "refinement_summary_tables_direct_enrollment",
    )
    parser.add_argument(
        "--canonical-reference",
        choices=("person_neutral", "person_expression", "mean_neutral"),
        default="mean_neutral",
    )
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
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument("--translation-target-type", default="depth_norm_delta_t")
    parser.add_argument("--translation-scale", type=float, default=1000.0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--mtcnn-ad", type=float, default=0.4)
    parser.add_argument(
        "--landmark-box-ad",
        type=float,
        default=0.65,
        help="Fallback box expansion around 68 detected landmarks when MTCNN is unavailable.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--raw-subject", default=None)
    parser.add_argument("--adapter-subject", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="head_yz_flip",
        help="Scoring/evaluation rotation convention.",
    )
    parser.add_argument(
        "--ready-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Rotation convention stored in ready_pose_predictions.csv.",
    )
    return parser.parse_args()


def format_cell(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def write_markdown(path, title, rows):
    lines = [
        f"# {title}",
        "",
        "| Category | Method | Frames | Yaw | Pitch | Roll | MAEr | GE | Tx | Ty | Tz | MAEt | ADD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {method} | {frames} | {yaw} | {pitch} | {roll} | {maer} | {ge} | "
            "{tx} | {ty} | {tz} | {maet} | {add} |".format(
                category=row["category"],
                method=row["method"],
                frames=row["frames"],
                yaw=format_cell(row["yaw"]),
                pitch=format_cell(row["pitch"]),
                roll=format_cell(row["roll"]),
                maer=format_cell(row["maer"]),
                ge=format_cell(row["ge"]),
                tx=format_cell(row["tx"]),
                ty=format_cell(row["ty"]),
                tz=format_cell(row["tz"]),
                maet=format_cell(row["maet"]),
                add=format_cell(row.get("add", "")),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def read_direct_baselines(path):
    if not path.is_file():
        return []
    pair_csv = path.parent / "first_frame_query_pairs.csv"
    frame_count = ""
    if pair_csv.is_file():
        with pair_csv.open(newline="") as handle:
            frame_count = sum(1 for _ in csv.DictReader(handle))
    rows = []
    label_map = {
        "vggt_ABS_single": "VGGT-HPE-Abs",
        "sixdrepnet": "6DRepNet",
        "tokenhpe": "TokenHPE-v1",
        "trg": "TRG",
    }
    for row in read_csv(path):
        if row.get("subject") != "overall":
            continue
        model = row.get("model", "")
        rows.append(
            {
                "category": "Direct baselines",
                "method": label_map.get(model, model),
                "frames": frame_count,
                "yaw": row.get("yaw_mae", ""),
                "pitch": row.get("pitch_mae", ""),
                "roll": row.get("roll_mae", ""),
                "maer": row.get("mean_mae", ""),
                "ge": "",
                "tx": row.get("tx_mae_mm", ""),
                "ty": row.get("ty_mae_mm", ""),
                "tz": row.get("tz_mae_mm", ""),
                "maet": row.get("translation_l2_mae_mm", ""),
                "add": "",
                "note": "same BIWI-compatible direct baseline evaluator",
            }
        )
    order = {"6DRepNet": 0, "TokenHPE-v1": 1, "TRG": 2, "VGGT-HPE-Abs": 3}
    rows.sort(key=lambda row: order.get(row["method"], 99))
    return rows


def load_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def gt_rotation_from_ready(row, args):
    return convert_famos_rotation_convention(
        row_matrix(row, "gt"),
        args.ready_rotation_convention,
        args.famos_rotation_convention,
    )


def add_prediction_row(method_rows, method, category, source_row, R_pred, t_pred, args):
    R_gt = gt_rotation_from_ready(source_row, args)
    t_gt = row_translation(source_row, "gt")
    signed = t_pred - t_gt
    out = {
        "category": category,
        "method": method,
        "protocol": source_row.get("protocol", args.protocol),
        "subject": source_row["subject"],
        "enrollment_id": source_row.get("enrollment_id", ""),
        "source_key": source_row.get("source_key", ""),
        "anchor_frame": source_row.get("anchor_frame", ""),
        "frame": source_row["frame"],
        "query_original_frame": source_row.get("query_original_frame", ""),
        "query_image_path": source_row.get("query_image_path", ""),
        "gt_tx_mm": float(t_gt[0]),
        "gt_ty_mm": float(t_gt[1]),
        "gt_tz_mm": float(t_gt[2]),
        "pred_tx_mm": float(t_pred[0]),
        "pred_ty_mm": float(t_pred[1]),
        "pred_tz_mm": float(t_pred[2]),
        "tx_error_mm": float(abs(signed[0])),
        "ty_error_mm": float(abs(signed[1])),
        "tz_error_mm": float(abs(signed[2])),
        "translation_l2_error_mm": float(np.linalg.norm(signed)),
    }
    for rr in range(3):
        for cc in range(3):
            out[f"pred_r{rr}{cc}"] = float(R_pred[rr, cc])
            out[f"gt_r{rr}{cc}"] = float(R_gt[rr, cc])
    method_rows[method].append(out)


def finalize_method_rows(method_rows, method_categories):
    summary_rows = []
    prediction_rows = []
    for method, rows in method_rows.items():
        pred_rotations = [
            np.asarray([[row[f"pred_r{rr}{cc}"] for cc in range(3)] for rr in range(3)], dtype=np.float64)
            for row in rows
        ]
        gt_rotations = [
            np.asarray([[row[f"gt_r{rr}{cc}"] for cc in range(3)] for rr in range(3)], dtype=np.float64)
            for row in rows
        ]
        yaw, pitch, roll, geod = angular_metrics(pred_rotations, gt_rotations)
        for row, y, p, r, g in zip(rows, yaw, pitch, roll, geod):
            row["yaw_error_deg"] = float(y)
            row["pitch_error_deg"] = float(p)
            row["roll_error_deg"] = float(r)
            row["rot_mean_error_deg"] = float((y + p + r) / 3.0)
            row["geodesic_error_deg"] = float(g)
            prediction_rows.append(row)
        metric = summarize(rows)
        summary_rows.append(
            {
                "category": method_categories[method],
                "method": method,
                "frames": metric["frames"],
                "yaw": metric["yaw_mae_deg"],
                "pitch": metric["pitch_mae_deg"],
                "roll": metric["roll_mae_deg"],
                "maer": metric["maer_deg"],
                "ge": metric["geodesic_mean_deg"],
                "tx": metric["tx_mae_mm"],
                "ty": metric["ty_mae_mm"],
                "tz": metric["tz_mae_mm"],
                "maet": metric["translation_l2_mae_mm"],
                "add": "",
                "note": "",
            }
        )
    return summary_rows, prediction_rows


def box_from_points(points, image_bgr, ad):
    img_h, img_w = image_bgr.shape[:2]
    points = np.asarray(points, dtype=np.float64)
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    xw1 = max(int(round(x1 - ad * width)), 0)
    yw1 = max(int(round(y1 - ad * height)), 0)
    xw2 = min(int(round(x2 + ad * width)), img_w - 1)
    yw2 = min(int(round(y2 + ad * height)), img_h - 1)
    return [xw1, xw2, yw1, yw2]


def center_box(image_bgr, crop_size):
    img_h, img_w = image_bgr.shape[:2]
    side = min(max(crop_size, min(img_w, img_h)), max(img_w, img_h))
    cx = img_w // 2
    cy = img_h // 2
    return [
        max(cx - side // 2, 0),
        min(cx + side // 2, img_w - 1),
        max(cy - side // 2, 0),
        min(cy + side // 2, img_h - 1),
    ]


def try_create_face_alignment(device):
    try:
        import face_alignment
    except Exception:
        return None
    landmark_type = getattr(face_alignment.LandmarksType, "TWO_D", None)
    if landmark_type is None:
        landmark_type = getattr(face_alignment.LandmarksType, "_2D")
    device_name = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
    return face_alignment.FaceAlignment(landmark_type, flip_input=False, device=device_name)


def detect_landmark_points(image_bgr, detector):
    if detector is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictions = detector.get_landmarks(image_rgb)
    if not predictions:
        return None
    candidates = [np.asarray(points, dtype=np.float64) for points in predictions if points is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda points: float((np.max(points[:, 0]) - np.min(points[:, 0])) * (np.max(points[:, 1]) - np.min(points[:, 1]))),
    )


def detect_box(image_bgr, detector, prev_box, crop_size, mtcnn_ad, landmark_ad, fallback_points=None):
    if detector is not None:
        points = detect_landmark_points(image_bgr, detector)
        if points is not None:
            return box_from_points(points, image_bgr, landmark_ad), "face_alignment_landmarks"
    if fallback_points is not None:
        return box_from_points(fallback_points, image_bgr, landmark_ad), "saved_enrollment_landmarks"
    if prev_box is not None:
        return prev_box, "previous_box"
    return center_box(image_bgr, crop_size), "center_box"


def build_enrollment_cache(row, args):
    image_bgr = load_bgr(row["enrollment_image_path"])
    fit_json = json.loads(Path(row["canonical_flame_json"]).read_text())
    enrollment_points = np.asarray(fit_json["fit"]["detected_landmarks_xy"], dtype=np.float64)
    anchor_box, anchor_box_source = detect_box(
        image_bgr,
        None,
        None,
        args.crop_size,
        args.mtcnn_ad,
        args.landmark_box_ad,
        fallback_points=enrollment_points,
    )
    intrinsics = load_intrinsics(Path(row["enrollment_rgb_cal_path"]))
    pnp = solve_enrollment_pnp(Path(row["canonical_flame_json"]), intrinsics, args.canonical_reference)
    R_anchor_eval = convert_famos_rotation_convention(
        pnp_rotation_to_eval_head_frame(pnp["rotation"]),
        "raw",
        args.famos_rotation_convention,
    )
    z_avatar, avatar_pair_estimates = avatarbox_median_depth(pnp, intrinsics)
    if not np.isfinite(z_avatar):
        raise RuntimeError(f"AvatarBox depth failed for enrollment {row.get('enrollment_id')}")
    pnp_t = np.asarray(pnp["translation_mm"], dtype=np.float64)
    anchors = {"pnp": pnp_t, "avatar_median": make_anchor_from_z(pnp_t, z_avatar, "rayxy")}
    if len(avatar_pair_estimates) >= 3:
        ordered = sorted(avatar_pair_estimates)
        z_trimmed = float(np.mean(ordered[1:-1]))
    else:
        z_trimmed = z_avatar
    anchors["avatar_trimmed_mean"] = make_anchor_from_z(pnp_t, z_trimmed, "rayxy")
    return {
        "image_bgr": image_bgr,
        "anchor_box": anchor_box,
        "R_anchor_eval": R_anchor_eval,
        "anchors": anchors,
        "debug": {
            "subject": row["subject"],
            "enrollment_id": row.get("enrollment_id", ""),
            "source_key": row.get("source_key", ""),
            "canonical_flame_json": row["canonical_flame_json"],
            "z_pnp_mm": float(pnp_t[2]),
            "z_avatarbox_median_mm": float(z_avatar),
            "z_avatarbox_trimmed_mean_mm": float(z_trimmed),
            "pnp_residual_mean_px": float(pnp["residual_mean_px"]),
            "pnp_residual_median_px": float(pnp["residual_median_px"]),
            "pnp_residual_p95_px": float(pnp["residual_p95_px"]),
            "avatar_pair_estimates_mm": " ".join(f"{value:.6f}" for value in avatar_pair_estimates),
            "anchor_box_source": anchor_box_source,
            "anchor_box": " ".join(str(value) for value in anchor_box),
        },
    }


def main():
    args = parse_args()
    ready_csv = args.ready_root / args.protocol / "ready_pose_predictions.csv"
    rows = [row for row in read_csv(ready_csv) if row.get("ready_for_refinement", "1") == "1"]
    if args.raw_subject is not None:
        rows = [row for row in rows if row.get("source_key", "").split("|")[0] == args.raw_subject]
    if args.adapter_subject is not None:
        subject = f"{int(args.adapter_subject):03d}" if str(args.adapter_subject).isdigit() else args.adapter_subject
        rows = [row for row in rows if row["subject"] == subject]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if not rows:
        raise RuntimeError(f"No ready rows selected from {ready_csv}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.manual_seed(0)
    np.random.seed(0)

    landmark_detector = try_create_face_alignment(device)
    model = base.load_vggt_model(
        str(args.base_checkpoint),
        str(args.lora_checkpoint),
        device,
        pose_encoding_type=args.pose_encoding_type,
    )

    method_specs = (
        ("Enrollment-PnP + relative", "No-known-anchor practical", "pnp"),
        ("AvatarBox-median", "AvatarBox", "avatar_median"),
        ("AvatarBox-trimmed-mean", "AvatarBox", "avatar_trimmed_mean"),
    )
    method_rows = defaultdict(list)
    method_categories = {method: category for method, category, _ in method_specs}
    debug_rows = []
    failure_rows = []
    cache = {}
    R_cam_t = torch.eye(3, dtype=torch.float32)
    t_cam_t = torch.zeros(3, dtype=torch.float32)

    by_enrollment = defaultdict(list)
    for row in rows:
        by_enrollment[(row["subject"], row.get("enrollment_id", ""), row.get("source_key", ""))].append(row)

    start_all = time.perf_counter()
    for group_key, group_rows in sorted(by_enrollment.items()):
        try:
            data = cache.get(group_key)
            if data is None:
                data = build_enrollment_cache(group_rows[0], args)
                cache[group_key] = data
                debug_rows.append(data["debug"])
        except Exception as exc:
            for row in group_rows:
                failure_rows.append({**row, "failure_reason": f"enrollment setup failed: {exc}"})
            continue

        prev_query_box = None
        for start in range(0, len(group_rows), args.batch_size):
            batch_rows = group_rows[start : start + args.batch_size]
            kept_rows = []
            anchor_tensors = []
            query_tensors = []
            for row in batch_rows:
                try:
                    query_bgr = load_bgr(row["query_image_path"])
                    prev_query_box, query_box_source = detect_box(
                        query_bgr,
                        landmark_detector,
                        prev_query_box,
                        args.crop_size,
                        args.mtcnn_ad,
                        args.landmark_box_ad,
                    )
                    pair_box = union_boxes(data["anchor_box"], prev_query_box)
                    anchor_rgb = square_box_rgb(data["image_bgr"], pair_box, args.crop_size)
                    query_rgb = square_box_rgb(query_bgr, pair_box, args.crop_size)
                    anchor_tensors.append(base.vggt_transform(anchor_rgb))
                    query_tensors.append(base.vggt_transform(query_rgb))
                    kept_rows.append(row)
                except Exception as exc:
                    failure_rows.append({**row, "failure_reason": f"query preprocessing failed: {exc}"})
            if not kept_rows:
                continue

            images = torch.stack([torch.stack(anchor_tensors), torch.stack(query_tensors)], dim=1).to(device)
            amp_context = torch.cuda.amp.autocast(dtype=amp_dtype) if torch.cuda.is_available() else nullcontext()
            with torch.no_grad(), amp_context:
                output = model(images=images)

            batch_size = len(kept_rows)
            R_anchor_t = torch.from_numpy(data["R_anchor_eval"]).float().unsqueeze(0).expand(batch_size, -1, -1)
            for method, category, anchor_key in method_specs:
                t_anchor = torch.from_numpy(data["anchors"][anchor_key]).float().unsqueeze(0).expand(batch_size, -1)
                variants, decoded_t = vggt_translation.decode_vggt_predictions(
                    output["pose_enc"],
                    R_anchor_t,
                    t_anchor,
                    R_cam_t,
                    t_cam_t,
                    pose_mode="relative_h2c",
                    translation_scale=args.translation_scale,
                    pose_encoding_type=args.pose_encoding_type,
                    translation_rotation_mode="predicted",
                    translation_target_type=args.translation_target_type,
                )
                R_pred_np = variants["E_h2c_correct"].cpu().numpy()
                t_pred_np = decoded_t.cpu().numpy()
                for local_i, row in enumerate(kept_rows):
                    add_prediction_row(
                        method_rows,
                        method,
                        category,
                        row,
                        R_pred_np[local_i],
                        t_pred_np[local_i],
                        args,
                    )

    refinement_summary, prediction_rows = finalize_method_rows(method_rows, method_categories)
    direct_rows = read_direct_baselines(args.baseline_root / args.protocol / "biwi_eval_results.csv")
    category_order = {"Direct baselines": 0, "No-known-anchor practical": 1, "AvatarBox": 2}
    method_order = {"6DRepNet": 0, "TokenHPE-v1": 1, "TRG": 2, "VGGT-HPE-Abs": 3}
    all_rows = direct_rows + refinement_summary
    all_rows.sort(key=lambda row: (category_order.get(row["category"], 99), method_order.get(row["method"], 50), row["method"]))

    out_dir = args.out_root / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "famos_refinement_full_summary.csv"
    summary_md = out_dir / "famos_refinement_full_summary.md"
    predictions_csv = out_dir / "famos_refinement_full_predictions.csv"
    debug_csv = out_dir / "famos_refinement_subject_debug.csv"
    failures_csv = out_dir / "famos_refinement_failures.csv"
    metadata_json = out_dir / "famos_refinement_full_metadata.json"

    fields = ["category", "method", "frames", "yaw", "pitch", "roll", "maer", "ge", "tx", "ty", "tz", "maet", "add", "note"]
    write_csv(summary_csv, all_rows, fields)
    write_csv(predictions_csv, prediction_rows, sorted({key for row in prediction_rows for key in row}))
    write_csv(debug_csv, debug_rows, list(debug_rows[0]) if debug_rows else [])
    if failure_rows:
        write_csv(failures_csv, failure_rows, sorted({key for row in failure_rows for key in row}))
    write_markdown(summary_md, f"{args.method_prefix} {args.protocol} Direct Enrollment Refinement Summary", all_rows)
    metadata = {
        "protocol": args.protocol,
        "ready_pose_predictions": str(ready_csv),
        "direct_baseline_csv": str(args.baseline_root / args.protocol / "biwi_eval_results.csv"),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "predictions_csv": str(predictions_csv),
        "debug_csv": str(debug_csv),
        "failures_csv": str(failures_csv) if failure_rows else "",
        "frames_requested": len(rows),
        "prediction_rows": len(prediction_rows),
        "failure_rows": len(failure_rows),
        "runtime_s": time.perf_counter() - start_all,
        "canonical_reference": args.canonical_reference,
        "famos_rotation_convention": args.famos_rotation_convention,
        "ready_rotation_convention": args.ready_rotation_convention,
        "ground_truth_usage": "GT is used only for final metric scoring. Anchor R/t are estimated from enrollment RGB/intrinsics/FLAME landmarks.",
        "notes": [
            "This direct evaluator reruns VGGT-HPE on enrollment/query RGB pairs.",
        ],
    }
    write_json(metadata_json, metadata)
    print(json.dumps({"summary_csv": str(summary_csv), "rows": len(all_rows), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
