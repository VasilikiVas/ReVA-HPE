#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLENDED_ROOT = Path(os.environ.get("BLENDED_IN_FLAMES_ROOT", str(PROJECT_ROOT.parent / "blended_in_flames")))
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "evaluation",
    PROJECT_ROOT / "evaluation" / "eval_scripts",
    Path(__file__).resolve().parent,
    BLENDED_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_biwi_crop as base 
import eval_biwi_crop_mtcnn_square_translation as vggt_translation  
from eval_biwi_query_flame_icosphere_ransac import (  
    _biwi_errors,
    _ransac_fuse,
    _render_anchors,
    _save_montage,
)
_TRIAL_MODULE = None


def _trial():
    global _TRIAL_MODULE
    if _TRIAL_MODULE is None:
        import importlib.util

        script = Path(BLENDED_ROOT) / "trial_photo_to_5views.py"
        if not script.is_file():
            raise FileNotFoundError(
                f"trial_photo_to_5views.py not found under {BLENDED_ROOT}, set "
                "BLENDED_IN_FLAMES_ROOT to the directory containing it."
            )
        spec = importlib.util.spec_from_file_location("trial_photo_to_5views", str(script))
        _TRIAL_MODULE = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(_TRIAL_MODULE)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"{exc}. The FLAME helpers need the packages from requirements.txt"
            ) from exc
    return _TRIAL_MODULE
from reva.experiments.eval_famos_gt_anchor import square_box_rgb, union_boxes


OUT_ROOT = Path("outputs")
DEFAULT_EXP = "flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit"
LOG = logging.getLogger("famos_flame_bank")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_view00_mtcnn_visible_no091")
    parser.add_argument(
        "--ready-root",
        type=Path,
        default=OUT_ROOT / "refinement_inputs_cross_dataset",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "flame_bank_hybrid",
    )
    parser.add_argument("--adapter-subject", default=None, help="Optional adapter subject id, e.g. 001.")
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "VGGT-1B" / "model.pt")
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "training" / "logs" / DEFAULT_EXP / "ckpts" / "checkpoint_410.pt",
    )
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument("--translation-target-type", default="depth_norm_delta_t")
    parser.add_argument("--translation-scale", type=float, default=1000.0)
    parser.add_argument("--anchor-count", type=int, default=20)
    parser.add_argument("--anchor-pose-mode", choices=("centered_icosphere", "fit_conditioned_se3"), default="fit_conditioned_se3")
    parser.add_argument("--anchor-dx-over-z-range", type=float, default=0.10)
    parser.add_argument("--anchor-dy-over-z-range", type=float, default=0.05)
    parser.add_argument("--anchor-log-z-range", type=float, default=0.10)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--mtcnn-ad", type=float, default=0.4)
    parser.add_argument("--ransac-rot-threshold-deg", type=float, default=10.0)
    parser.add_argument("--ransac-trans-threshold-mm", type=float, default=50.0)
    parser.add_argument("--ransac-min-inliers", type=int, default=3)
    parser.add_argument("--method-name", default="FaMoS FLAME-bank RANSAC")
    parser.add_argument(
        "--bank-translation-mode",
        choices=("metric", "rotation_only"),
        default="metric",
        help="Use metric only when bank translations are valid. rotation_only disables direct translation reporting.",
    )
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="GT rotation convention for FaMoS scoring. head_yz_flip uses R_gt @ diag(1,-1,-1).",
    )
    parser.add_argument("--max-visualizations", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_intrinsics(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return np.asarray([[float(value) for value in lines[idx].split()] for idx in range(3)], dtype=np.float64)


def matrix_from_row(row, prefix):
    return np.asarray(
        [[float(row[f"{prefix}_r{rr}{cc}"]) for cc in range(3)] for rr in range(3)],
        dtype=np.float64,
    )


def translation_from_row(row, prefix):
    return np.asarray([float(row[f"{prefix}_{axis}_mm"]) for axis in ("tx", "ty", "tz")], dtype=np.float64)


def apply_famos_rotation_convention(rotation, convention):
    if convention == "raw":
        return rotation
    if convention == "head_yz_flip":
        head_flip = np.diag([1.0, -1.0, -1.0])
        return rotation @ head_flip
    raise ValueError(f"Unknown FaMoS rotation convention: {convention}")


def tensor(value, device, shape=None):
    out = torch.as_tensor(value, dtype=torch.float32, device=device)
    if shape is not None:
        out = out.reshape(*shape)
    return out


def reconstruct_fit(flame, fit_json, device):
    fit = fit_json["fit"]
    shape = tensor(fit["shape_coefficients"], device)
    expr = tensor(fit["expression_coefficients"], device)
    global_pose = tensor(fit["estimated_global_pose_axis_angle"], device)
    jaw_pose = tensor(fit["estimated_jaw_pose_axis_angle"], device)
    scale = tensor([float(fit["weak_perspective_scale"])], device)
    trans = tensor(fit["weak_perspective_translation_xy"], device, shape=(2,))
    with torch.no_grad():
        pose = torch.cat([global_pose, jaw_pose], dim=1)
        verts, _, _, _ = flame(shape_params=shape, expression_params=expr, pose_params=pose)
    return {
        "shape": shape,
        "expr": expr,
        "global_pose": global_pose,
        "jaw_pose": jaw_pose,
        "scale": scale,
        "trans": trans,
        "verts_fit": verts.detach(),
    }


def load_or_render_anchors(args, flame, rows, device):
    first = rows[0]
    fit_path = Path(first["canonical_flame_json"])
    fit_json = json.loads(fit_path.read_text())
    fit = reconstruct_fit(flame, fit_json, device)
    image_rgb = _trial().load_image(first["enrollment_image_path"])
    intrinsics = load_intrinsics(first["enrollment_rgb_cal_path"])
    colors = _trial().sample_vertex_colors(
        flame,
        fit["verts_fit"],
        fit["scale"],
        fit["trans"],
        image_rgb,
        device,
    )
    return _render_anchors(
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


def summarize(rows, expected):
    if not rows:
        return {}
    metrics = (
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
        "total_runtime_s",
    )
    out = {
        "subject": rows[0]["subject"],
        "enrollment_id": rows[0]["enrollment_id"],
        "source_key": rows[0]["source_key"],
        "expected_queries": expected,
        "queries": len(rows),
        "coverage": len(rows) / max(expected, 1),
        "ransac_success_rate": float(np.mean([int(row["ransac_success"]) for row in rows])),
    }
    for metric in metrics:
        out[metric] = float(np.mean([float(row[metric]) for row in rows]))
    return out


def evaluate_group(args, model, flame, mtcnn, subject, rows, out_dir, device, amp_dtype):
    start_all = time.perf_counter()
    anchors = load_or_render_anchors(args, flame, rows, device)
    bank_metadata = getattr(args, "_bank_metadata", {})
    (
        anchors_rgb,
        anchor_boxes,
        anchor_rot_rgb,
        anchor_trans_rgb_m,
        directions,
        translation_offsets,
        anchor_distance,
        head_radius,
    ) = anchors

    R_cam_t = torch.eye(3, dtype=torch.float32)
    t_cam_t = torch.zeros(3, dtype=torch.float32)
    anchor_rot_t = torch.from_numpy(anchor_rot_rgb.astype(np.float64)).float()
    anchor_trans_t = torch.from_numpy(anchor_trans_rgb_m.astype(np.float64) * 1000.0).float()
    prediction_rows = []
    pair_rows = []
    failure_rows = []
    prev_query_box = None
    visualized = 0

    for row_index, source in enumerate(rows):
        query_start = time.perf_counter()
        image_bgr = cv2.imread(source["query_image_path"])
        if image_bgr is None:
            failure_rows.append({**source, "error": f"could not read query image {source['query_image_path']}"})
            continue
        try:
            _, prev_query_box = base.crop_face_mtcnn(
                image_bgr,
                mtcnn,
                prev_query_box,
                crop_size=args.crop_size,
                ad=args.mtcnn_ad,
            )
        except Exception as exc:
            failure_rows.append({**source, "error": f"query MTCNN failed: {exc}"})
            continue

        predicted_rotations = []
        predicted_translations = []
        inference_start = time.perf_counter()
        for start in range(0, args.anchor_count, args.pair_batch_size):
            end = min(start + args.pair_batch_size, args.anchor_count)
            anchor_tensors = []
            query_tensors = []
            for anchor_id in range(start, end):
                pair_box = union_boxes(anchor_boxes[anchor_id], prev_query_box)
                anchor_bgr = cv2.cvtColor(anchors_rgb[anchor_id], cv2.COLOR_RGB2BGR)
                anchor_crop = square_box_rgb(anchor_bgr, pair_box, args.crop_size)
                query_crop = square_box_rgb(image_bgr, pair_box, args.crop_size)
                anchor_tensors.append(base.vggt_transform(anchor_crop))
                query_tensors.append(base.vggt_transform(query_crop))
            images = torch.stack([torch.stack(anchor_tensors), torch.stack(query_tensors)], dim=1).to(device)
            amp_context = torch.cuda.amp.autocast(dtype=amp_dtype) if torch.cuda.is_available() else nullcontext()
            with torch.no_grad(), amp_context:
                output = model(images=images)
            variants, decoded_t = vggt_translation.decode_vggt_predictions(
                output["pose_enc"],
                anchor_rot_t[start:end],
                anchor_trans_t[start:end],
                R_cam_t,
                t_cam_t,
                pose_mode="relative_h2c",
                translation_scale=args.translation_scale,
                pose_encoding_type=args.pose_encoding_type,
                translation_rotation_mode="predicted",
                translation_target_type=args.translation_target_type,
            )
            predicted_rotations.extend(variants["E_h2c_correct"].cpu().numpy())
            predicted_translations.extend(decoded_t.cpu().numpy())
        inference_runtime = time.perf_counter() - inference_start

        fused_rotation, fused_translation, inlier_ids, success = _ransac_fuse(
            predicted_rotations,
            predicted_translations,
            args.ransac_rot_threshold_deg,
            args.ransac_trans_threshold_mm,
            args.ransac_min_inliers,
        )
        gt_rotation_raw = matrix_from_row(source, "gt")
        gt_rotation = apply_famos_rotation_convention(gt_rotation_raw, args.famos_rotation_convention)
        gt_translation = translation_from_row(source, "gt")
        angle_error, translation_error, l2_error = _biwi_errors(
            fused_rotation,
            fused_translation,
            gt_rotation,
            gt_translation,
        )
        out = {
            "method": args.method_name,
            "protocol": args.protocol,
            "subject": source["subject"],
            "enrollment_id": source["enrollment_id"],
            "source_key": source["source_key"],
            "anchor_frame": source["anchor_frame"],
            "frame": source["frame"],
            "query_original_frame": source.get("query_original_frame", ""),
            "query_image_path": source["query_image_path"],
            "anchors": args.anchor_count,
            "anchor_pose_mode": args.anchor_pose_mode,
            "bank_translation_mode": args.bank_translation_mode,
            "direct_bank_translation_valid": int(args.bank_translation_mode == "metric"),
            "famos_rotation_convention": args.famos_rotation_convention,
            "ransac_inliers": len(inlier_ids),
            "ransac_success": int(success),
            "inlier_ids": " ".join(str(int(idx)) for idx in inlier_ids),
            "yaw_error_deg": float(angle_error[0]),
            "pitch_error_deg": float(angle_error[1]),
            "roll_error_deg": float(angle_error[2]),
            "rot_mean_error_deg": float(angle_error.mean()),
            "tx_error_mm": float(translation_error[0]),
            "ty_error_mm": float(translation_error[1]),
            "tz_error_mm": float(translation_error[2]),
            "translation_l2_error_mm": float(l2_error),
            "pred_tx_mm": float(fused_translation[0]),
            "pred_ty_mm": float(fused_translation[1]),
            "pred_tz_mm": float(fused_translation[2]),
            "gt_tx_mm": float(gt_translation[0]),
            "gt_ty_mm": float(gt_translation[1]),
            "gt_tz_mm": float(gt_translation[2]),
            "anchor_distance_model_units": float(anchor_distance),
            "head_radius_model_units": float(head_radius),
            "inference_runtime_s": inference_runtime,
            "total_runtime_s": time.perf_counter() - query_start,
        }
        out.update(bank_metadata)
        for rr in range(3):
            for cc in range(3):
                out[f"pred_r{rr}{cc}"] = float(fused_rotation[rr, cc])
                out[f"gt_r{rr}{cc}"] = float(gt_rotation[rr, cc])
                out[f"gt_raw_r{rr}{cc}"] = float(gt_rotation_raw[rr, cc])
        prediction_rows.append(out)

        inlier_set = set(int(idx) for idx in inlier_ids)
        for anchor_id, (rotation_pred, translation_pred) in enumerate(zip(predicted_rotations, predicted_translations)):
            pair_angle, pair_translation, pair_l2 = _biwi_errors(
                rotation_pred,
                translation_pred,
                gt_rotation,
                gt_translation,
            )
            pair = {
                "protocol": args.protocol,
                "subject": source["subject"],
                "enrollment_id": source["enrollment_id"],
                "source_key": source["source_key"],
                "frame": source["frame"],
                "query_original_frame": source.get("query_original_frame", ""),
                "anchor_id": anchor_id,
                "inlier": int(anchor_id in inlier_set),
                "anchor_dx_over_z_offset": float(translation_offsets[anchor_id, 0]),
                "anchor_dy_over_z_offset": float(translation_offsets[anchor_id, 1]),
                "anchor_log_z_offset": float(translation_offsets[anchor_id, 2]),
                "anchor_tx_mm": float(anchor_trans_rgb_m[anchor_id, 0] * 1000.0),
                "anchor_ty_mm": float(anchor_trans_rgb_m[anchor_id, 1] * 1000.0),
                "anchor_tz_mm": float(anchor_trans_rgb_m[anchor_id, 2] * 1000.0),
                "anchor_direction_x": float(directions[anchor_id, 0]),
                "anchor_direction_y": float(directions[anchor_id, 1]),
                "anchor_direction_z": float(directions[anchor_id, 2]),
                "famos_rotation_convention": args.famos_rotation_convention,
                "bank_translation_mode": args.bank_translation_mode,
                "direct_bank_translation_valid": int(args.bank_translation_mode == "metric"),
                "yaw_error_deg": float(pair_angle[0]),
                "pitch_error_deg": float(pair_angle[1]),
                "roll_error_deg": float(pair_angle[2]),
                "rot_mean_error_deg": float(pair_angle.mean()),
                "tx_error_mm": float(pair_translation[0]),
                "ty_error_mm": float(pair_translation[1]),
                "tz_error_mm": float(pair_translation[2]),
                "translation_l2_error_mm": float(pair_l2),
                "pred_tx_mm": float(translation_pred[0]),
                "pred_ty_mm": float(translation_pred[1]),
                "pred_tz_mm": float(translation_pred[2]),
            }
            pair.update(bank_metadata)
            for rr in range(3):
                for cc in range(3):
                    pair[f"pred_r{rr}{cc}"] = float(rotation_pred[rr, cc])
            pair_rows.append(pair)

        if visualized < args.max_visualizations:
            _save_montage(
                out_dir / f"subject_{subject}_frame_{source['frame']}_flame_bank.png",
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                anchors_rgb,
                inlier_ids,
                directions,
                translation_offsets,
            )
            visualized += 1
        LOG.info(
            "[%s frame %s] inliers=%d/%d rot=%.3f trans=%.3f runtime=%.1fs",
            subject,
            source["frame"],
            len(inlier_ids),
            args.anchor_count,
            float(angle_error.mean()),
            float(l2_error),
            time.perf_counter() - query_start,
        )

    if prediction_rows:
        write_csv(out_dir / f"subject_{subject}_predictions.csv", prediction_rows)
    if pair_rows:
        write_csv(out_dir / f"subject_{subject}_pairwise.csv", pair_rows)
    if failure_rows:
        write_csv(out_dir / f"subject_{subject}_failures.csv", failure_rows)
    summary = summarize(prediction_rows, len(rows))
    if summary:
        write_csv(out_dir / f"subject_{subject}_summary.csv", [summary])
    metadata = {
        "protocol": args.protocol,
        "subject": subject,
        "enrollment_id": rows[0]["enrollment_id"],
        "source_key": rows[0]["source_key"],
        "query_rows": len(rows),
        "prediction_rows": len(prediction_rows),
        "failure_rows": len(failure_rows),
        "canonical_flame_json": rows[0]["canonical_flame_json"],
        "enrollment_image_path": rows[0]["enrollment_image_path"],
        "uses_dataset_pose_or_depth_ground_truth_for_prediction": False,
        "bank_translation_mode": args.bank_translation_mode,
        "direct_bank_translation_valid": args.bank_translation_mode == "metric",
        "ground_truth_usage": "GT is used only for final error scoring.",
        "famos_rotation_convention": args.famos_rotation_convention,
        "runtime_s": time.perf_counter() - start_all,
        "summary": summary,
    }
    metadata.update(bank_metadata)
    (out_dir / f"subject_{subject}_metadata.json").write_text(json.dumps(metadata, indent=2))
    return prediction_rows, failure_rows


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out_dir = args.out_root / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    ready_csv = args.ready_root / args.protocol / "ready_pose_predictions.csv"
    rows = read_csv(ready_csv)
    if args.adapter_subject is not None:
        subject = f"{int(args.adapter_subject):03d}" if str(args.adapter_subject).isdigit() else args.adapter_subject
        rows = [row for row in rows if row["subject"] == subject]
        if not rows:
            raise RuntimeError(f"No ready rows for adapter subject {subject} in {ready_csv}")
    raw_subject = getattr(args, "raw_subject", None)
    if raw_subject is not None:
        rows = [row for row in rows if row["source_key"].split("|")[0] == raw_subject]
        if not rows:
            raise RuntimeError(f"No ready rows for raw subject {raw_subject} in {ready_csv}")
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.manual_seed(0)
    np.random.seed(0)
    LOG.info("Device=%s amp=%s groups=%d", device, amp_dtype, len(by_subject))

    flame, _ = _trial().build_flame(device, BLENDED_ROOT / "assets")
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(keep_all=True, device=device)
    model = base.load_vggt_model(
        str(args.base_checkpoint),
        str(args.lora_checkpoint),
        device,
        pose_encoding_type=args.pose_encoding_type,
    )

    all_predictions = []
    all_failures = []
    for subject, subject_rows in sorted(by_subject.items()):
        preds, fails = evaluate_group(args, model, flame, mtcnn, subject, subject_rows, out_dir, device, amp_dtype)
        all_predictions.extend(preds)
        all_failures.extend(fails)

    if args.adapter_subject is None and raw_subject is None and all_predictions:
        write_csv(out_dir / "combined_predictions.csv", all_predictions)
    if args.adapter_subject is None and raw_subject is not None and all_predictions:
        safe_raw_subject = raw_subject.replace("/", "_").replace("|", "_")
        write_csv(out_dir / f"{safe_raw_subject}_combined_predictions.csv", all_predictions)
    if args.adapter_subject is None and raw_subject is None and all_failures:
        write_csv(out_dir / "combined_failures.csv", all_failures)
    if args.adapter_subject is None and raw_subject is not None and all_failures:
        safe_raw_subject = raw_subject.replace("/", "_").replace("|", "_")
        write_csv(out_dir / f"{safe_raw_subject}_combined_failures.csv", all_failures)
    print(json.dumps({"out_dir": str(out_dir), "predictions": len(all_predictions), "failures": len(all_failures)}, indent=2))


if __name__ == "__main__":
    main()
