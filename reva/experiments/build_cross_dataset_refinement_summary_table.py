#!/usr/bin/env python3
"""Build a BIWI-style summary table for cross-dataset refinement protocols."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "eval_scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "eval_scripts"))

from reva.experiments.eval_cross_dataset_avatarbox_median import (  
    AVATAR_MEDIAN_PAIRS,
    angular_metrics,
    avatarbox_median_depth,
    decode_depth_norm_slot,
    load_intrinsics,
    load_pose_txt,
    pnp_rotation_to_eval_head_frame,
    read_csv,
    recover_depth_norm_slot,
    row_matrix,
    row_translation,
    solve_enrollment_pnp,
    summarize,
    write_csv,
    write_json,
)


OUT_ROOT = Path("outputs")
HEAD_YZ_FLIP = np.diag([1.0, -1.0, -1.0])


PAIR_NAMES = {
    (36, 45): "outer-eye (36,45)",
    (39, 42): "inner-eye (39,42)",
    (31, 35): "nose-width (31,35)",
    (48, 54): "mouth (48,54)",
    (27, 33): "nose-bridge-tip (27,33)",
    (30, 33): "lower-nose-tip (30,33)",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_view00_mtcnn_visible_no091")
    parser.add_argument("--method-prefix", default="FaMoS")
    parser.add_argument(
        "--refinement-root",
        type=Path,
        default=OUT_ROOT / "refinement_inputs_cross_dataset",
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=OUT_ROOT / "biwi_compatible_eval_roots",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=OUT_ROOT / "biwi_compatible_baseline_outputs",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "refinement_summary_tables",
    )
    parser.add_argument(
        "--canonical-reference",
        choices=("person_neutral", "person_expression", "mean_neutral"),
        default="mean_neutral",
    )
    parser.add_argument("--box-scale-mm", type=float, default=200.0)
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Convention for FaMoS GT/pred rotations. head_yz_flip uses R @ diag(1,-1,-1).",
    )
    parser.add_argument(
        "--ready-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Rotation convention used by ready_pose_predictions.csv.",
    )
    parser.add_argument(
        "--adapter-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Rotation convention already stored in the BIWI-compatible adapter pose.txt files.",
    )
    return parser.parse_args()


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


def make_anchor_from_z(base_translation, z_value, mode="rayxy"):
    z_base = max(float(base_translation[2]), 1e-6)
    if mode == "fixedxy":
        return np.asarray([base_translation[0], base_translation[1], z_value], dtype=np.float64)
    alpha = z_value / z_base
    return np.asarray([alpha * base_translation[0], alpha * base_translation[1], z_value], dtype=np.float64)


def apply_famos_rotation_convention(rotation, convention):
    if convention == "raw":
        return rotation
    if convention == "head_yz_flip":
        return rotation @ HEAD_YZ_FLIP
    raise ValueError(f"Unknown FaMoS rotation convention: {convention}")


def convert_famos_rotation_convention(rotation, source_convention, target_convention):
    if source_convention == target_convention:
        return rotation
    if {source_convention, target_convention} == {"raw", "head_yz_flip"}:
        return rotation @ HEAD_YZ_FLIP
    raise ValueError(
        f"Unsupported FaMoS rotation convention conversion: "
        f"{source_convention} -> {target_convention}"
    )


def landmark_box_depth(pnp_data, intrinsics, box_scale_mm):
    image_points = pnp_data["image_points"]
    x1, y1 = np.min(image_points, axis=0)
    x2, y2 = np.max(image_points, axis=0)
    box_b = max(float(x2 - x1), float(y2 - y1), 1e-6)
    focal = 0.5 * (float(intrinsics[0, 0]) + float(intrinsics[1, 1]))
    return float(box_scale_mm * focal / box_b), box_b


def add_method_row(
    method_rows,
    method,
    category,
    source_row,
    R_pred,
    t_pred,
    rotation_convention="raw",
    ready_rotation_convention="raw",
):
    R_gt = convert_famos_rotation_convention(
        row_matrix(source_row, "gt"),
        ready_rotation_convention,
        rotation_convention,
    )
    t_gt = row_translation(source_row, "gt")
    signed = t_pred - t_gt
    row = {
        "category": category,
        "method": method,
        "subject": source_row["subject"],
        "frame": source_row["frame"],
        "gt_tx_mm": t_gt[0],
        "gt_ty_mm": t_gt[1],
        "gt_tz_mm": t_gt[2],
        "pred_tx_mm": t_pred[0],
        "pred_ty_mm": t_pred[1],
        "pred_tz_mm": t_pred[2],
        "tx_error_mm": abs(signed[0]),
        "ty_error_mm": abs(signed[1]),
        "tz_error_mm": abs(signed[2]),
        "translation_l2_error_mm": float(np.linalg.norm(signed)),
    }
    for rr in range(3):
        for cc in range(3):
            row[f"pred_r{rr}{cc}"] = R_pred[rr, cc]
            row[f"gt_r{rr}{cc}"] = R_gt[rr, cc]
    method_rows[method].append(row)


def finalize_method_rows(method_rows, method_categories):
    summary_rows = []
    prediction_rows = []
    for method, rows in method_rows.items():
        pred_rotations = []
        gt_rotations = []
        for row in rows:
            pred_rotations.append(
                np.asarray([[row[f"pred_r{rr}{cc}"] for cc in range(3)] for rr in range(3)], dtype=np.float64)
            )
            gt_rotations.append(
                np.asarray([[row[f"gt_r{rr}{cc}"] for cc in range(3)] for rr in range(3)], dtype=np.float64)
            )
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
                add=format_cell(row["add"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    protocol_dir = args.refinement_root / args.protocol
    ready_csv = protocol_dir / "ready_pose_predictions.csv"
    if not ready_csv.is_file():
        raise FileNotFoundError(ready_csv)
    rows = read_csv(ready_csv)
    if not rows:
        raise RuntimeError(f"No rows in {ready_csv}")

    subject_cache = {}
    method_rows = defaultdict(list)
    method_categories = {}
    subject_debug = []

    def register(method, category):
        method_categories.setdefault(method, category)
        return method

    for row in rows:
        subject = row["subject"]
        if subject not in subject_cache:
            adapter_subject_dir = args.adapter_root / args.protocol / subject
            R_anchor_gt_raw, t_anchor_gt = load_pose_txt(adapter_subject_dir / f"frame_{int(row['anchor_frame']):05d}_pose.txt")
            R_anchor_gt = convert_famos_rotation_convention(
                R_anchor_gt_raw,
                args.adapter_rotation_convention,
                args.famos_rotation_convention,
            )
            intrinsics = load_intrinsics(adapter_subject_dir / "rgb.cal")
            pnp = solve_enrollment_pnp(Path(row["canonical_flame_json"]), intrinsics, args.canonical_reference)
            R_anchor_eval = convert_famos_rotation_convention(
                pnp_rotation_to_eval_head_frame(pnp["rotation"]),
                "raw",
                args.famos_rotation_convention,
            )
            z_avatar, avatar_pair_estimates = avatarbox_median_depth(pnp, intrinsics)
            z_box, box_b = landmark_box_depth(pnp, intrinsics, args.box_scale_mm)
            pnp_t = pnp["translation_mm"]
            anchors = {
                "pnp": pnp_t,
                "avatar_median": make_anchor_from_z(pnp_t, z_avatar, "rayxy"),
                "box_fixedxy": make_anchor_from_z(pnp_t, z_box, "fixedxy"),
                "box_rayxy": make_anchor_from_z(pnp_t, z_box, "rayxy"),
            }
            if len(avatar_pair_estimates) >= 3:
                ordered = sorted(avatar_pair_estimates)
                z_trimmed = float(np.mean(ordered[1:-1]))
            else:
                z_trimmed = z_avatar
            anchors["avatar_trimmed_mean"] = make_anchor_from_z(pnp_t, z_trimmed, "rayxy")
            for pair, z_pair in zip(AVATAR_MEDIAN_PAIRS, avatar_pair_estimates):
                anchors[f"avatar_pair_{PAIR_NAMES[pair]}"] = make_anchor_from_z(pnp_t, z_pair, "rayxy")

            subject_cache[subject] = {
                "R_anchor_gt": R_anchor_gt,
                "t_anchor_gt": t_anchor_gt,
                "R_anchor_eval": R_anchor_eval,
                "t_anchor_pnp": pnp_t,
                "anchors": anchors,
            }
            subject_debug.append(
                {
                    "subject": subject,
                    "enrollment_id": row.get("enrollment_id", ""),
                    "famos_rotation_convention": args.famos_rotation_convention,
                    "z_gt_anchor_mm": t_anchor_gt[2],
                    "z_pnp_mm": pnp_t[2],
                    "z_avatarbox_median_mm": z_avatar,
                    "z_avatarbox_trimmed_mean_mm": z_trimmed,
                    "z_landmark_box_mm": z_box,
                    "landmark_box_b_px": box_b,
                    "pnp_residual_median_px": pnp["residual_median_px"],
                    "pnp_residual_p95_px": pnp["residual_p95_px"],
                    "avatar_pair_estimates_mm": " ".join(f"{value:.6f}" for value in avatar_pair_estimates),
                }
            )

        data = subject_cache[subject]
        R_gt_anchor_pred = convert_famos_rotation_convention(
            row_matrix(row, "pred"),
            args.ready_rotation_convention,
            args.famos_rotation_convention,
        )
        t_gt_anchor_pred = row_translation(row, "pred")
        R_rel = R_gt_anchor_pred @ data["R_anchor_gt"].T
        slot = recover_depth_norm_slot(row, data["t_anchor_gt"])

        add_method_row(
            method_rows,
            register("D0 GT-anchor relative (known anchor)", "Known-anchor/oracle"),
            "Known-anchor/oracle",
            row,
            R_gt_anchor_pred,
            t_gt_anchor_pred,
            args.famos_rotation_convention,
            args.ready_rotation_convention,
        )

        R_pnp = R_rel @ data["R_anchor_eval"]
        t_pnp = decode_depth_norm_slot(slot, data["anchors"]["pnp"])
        add_method_row(
            method_rows,
            register("Enrollment-PnP + relative", "No-known-anchor practical"),
            "No-known-anchor practical",
            row,
            R_pnp,
            t_pnp,
            args.famos_rotation_convention,
            args.ready_rotation_convention,
        )
        add_method_row(
            method_rows,
            register("D1 PnP-R + GT anchor t", "GT diagnostic"),
            "GT diagnostic",
            row,
            R_pnp,
            t_gt_anchor_pred,
            args.famos_rotation_convention,
            args.ready_rotation_convention,
        )
        add_method_row(
            method_rows,
            register("D2 GT-R + PnP anchor t", "GT diagnostic"),
            "GT diagnostic",
            row,
            R_gt_anchor_pred,
            t_pnp,
            args.famos_rotation_convention,
            args.ready_rotation_convention,
        )
        d3_anchor = np.asarray([data["t_anchor_pnp"][0], data["t_anchor_pnp"][1], data["t_anchor_gt"][2]], dtype=np.float64)
        d4_anchor = np.asarray([data["t_anchor_gt"][0], data["t_anchor_gt"][1], data["t_anchor_pnp"][2]], dtype=np.float64)
        add_method_row(
            method_rows,
            register("D3 PnP xy + GT z", "GT diagnostic"),
            "GT diagnostic",
            row,
            R_pnp,
            decode_depth_norm_slot(slot, d3_anchor),
            args.famos_rotation_convention,
            args.ready_rotation_convention,
        )
        add_method_row(
            method_rows,
            register("D4 GT xy + PnP z", "GT diagnostic"),
            "GT diagnostic",
            row,
            R_pnp,
            decode_depth_norm_slot(slot, d4_anchor),
            args.famos_rotation_convention,
            args.ready_rotation_convention,
        )

        for anchor_name, method, category in (
            ("avatar_median", "AvatarBox-median", "AvatarBox"),
            ("avatar_trimmed_mean", "AvatarBox-trimmed-mean", "AvatarBox"),
            ("box_fixedxy", "LandmarkBox-z-fixedxy", "Generic box controls"),
            ("box_rayxy", "LandmarkBox-z-rayxy", "Generic box controls"),
        ):
            add_method_row(
                method_rows,
                register(method, category),
                category,
                row,
                R_pnp,
                decode_depth_norm_slot(slot, data["anchors"][anchor_name]),
                args.famos_rotation_convention,
                args.ready_rotation_convention,
            )
        for pair in AVATAR_MEDIAN_PAIRS:
            anchor_name = f"avatar_pair_{PAIR_NAMES[pair]}"
            if anchor_name not in data["anchors"]:
                continue
            method = f"AvatarBox-single-pair {PAIR_NAMES[pair]}"
            add_method_row(
                method_rows,
                register(method, "AvatarBox pair ablation"),
                "AvatarBox pair ablation",
                row,
                R_pnp,
                decode_depth_norm_slot(slot, data["anchors"][anchor_name]),
                args.famos_rotation_convention,
                args.ready_rotation_convention,
            )

    refinement_summary, prediction_rows = finalize_method_rows(method_rows, method_categories)
    direct_rows = read_direct_baselines(args.baseline_root / args.protocol / "biwi_eval_results.csv")

    category_order = {
        "Direct baselines": 0,
        "Known-anchor/oracle": 1,
        "No-known-anchor practical": 2,
        "AvatarBox": 3,
        "AvatarBox pair ablation": 4,
        "Generic box controls": 5,
        "GT diagnostic": 6,
    }
    method_order = {
        "6DRepNet": 0,
        "TokenHPE-v1": 1,
        "TRG": 2,
        "VGGT-HPE-Abs": 3,
        "D0 GT-anchor relative (known anchor)": 0,
        "Enrollment-PnP + relative": 0,
        "AvatarBox-median": 0,
        "AvatarBox-trimmed-mean": 1,
    }
    all_rows = direct_rows + refinement_summary
    all_rows.sort(key=lambda row: (category_order.get(row["category"], 99), method_order.get(row["method"], 50), row["method"]))

    out_dir = args.out_root / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "famos_refinement_full_summary.csv"
    summary_md = out_dir / "famos_refinement_full_summary.md"
    predictions_csv = out_dir / "famos_refinement_full_predictions.csv"
    debug_csv = out_dir / "famos_refinement_subject_debug.csv"
    metadata_json = out_dir / "famos_refinement_full_metadata.json"

    fields = ["category", "method", "frames", "yaw", "pitch", "roll", "maer", "ge", "tx", "ty", "tz", "maet", "add", "note"]
    write_csv(summary_csv, all_rows, fields)
    pred_fields = sorted({key for row in prediction_rows for key in row})
    write_csv(predictions_csv, prediction_rows, pred_fields)
    write_csv(debug_csv, subject_debug, list(subject_debug[0]))
    write_markdown(summary_md, f"{args.method_prefix} {args.protocol} Full Refinement Summary", all_rows)
    write_json(
        metadata_json,
        {
            "protocol": args.protocol,
            "ready_pose_predictions": str(ready_csv),
            "direct_baseline_csv": str(args.baseline_root / args.protocol / "biwi_eval_results.csv"),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "predictions_csv": str(predictions_csv),
            "subject_debug_csv": str(debug_csv),
            "canonical_reference": args.canonical_reference,
            "famos_rotation_convention": args.famos_rotation_convention,
            "ready_rotation_convention": args.ready_rotation_convention,
            "adapter_rotation_convention": args.adapter_rotation_convention,
            "fixed_avatarbox_pairs": AVATAR_MEDIAN_PAIRS,
        },
    )
    print(json.dumps({"summary_md": str(summary_md), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
