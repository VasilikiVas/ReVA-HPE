#!/usr/bin/env python3

import argparse
import csv
import json
import math
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

import eval_biwi_crop as base  # noqa: E402
from reva.experiments.eval_famos_gt_anchor import geodesic_deg_torch  


OUT_ROOT = Path("outputs")
HEAD_YZ_FLIP = np.diag([1.0, -1.0, -1.0])
AVATAR_MEDIAN_PAIRS = (
    (36, 45),  # outer eye corners
    (39, 42),  # inner eye corners
    (31, 35),  # nose width
    (48, 54),  # mouth corners
    (27, 33),  # nose bridge to nose tip
    (30, 33),  # lower nose bridge / nose tip
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_mtcnn_visible_rel45")
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
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "avatarbox_median_cross_dataset",
    )
    parser.add_argument(
        "--canonical-reference",
        choices=("person_neutral", "person_expression", "mean_neutral"),
        default="mean_neutral",
    )
    parser.add_argument("--method-name", default="AvatarBox-median")
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Convention for FaMoS GT/pred rotations. head_yz_flip uses R @ diag(1,-1,-1).",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_pose_txt(path):
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape != (4, 3):
        raise ValueError(f"Expected 4x3 pose file, got {values.shape}: {path}")
    return values[:3, :], values[3, :]


def load_intrinsics(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split()])
            if len(rows) == 3:
                break
    k = np.asarray(rows, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics in {path}, got {k.shape}")
    return k


def row_matrix(row, prefix):
    return np.asarray(
        [[float(row[f"{prefix}_r{rr}{cc}"]) for cc in range(3)] for rr in range(3)],
        dtype=np.float64,
    )


def row_translation(row, prefix):
    return np.asarray(
        [float(row[f"{prefix}_tx_mm"]), float(row[f"{prefix}_ty_mm"]), float(row[f"{prefix}_tz_mm"])],
        dtype=np.float64,
    )


def solve_enrollment_pnp(canonical_json, intrinsics, reference):
    data = json.loads(canonical_json.read_text())
    object_points = np.asarray(
        data["canonical_references"][reference]["landmarks_xyz_opencv"],
        dtype=np.float64,
    )
    image_points = np.asarray(data["fit"]["detected_landmarks_xy"], dtype=np.float64)
    if object_points.shape != (68, 3) or image_points.shape != (68, 2):
        raise ValueError(f"Unexpected landmark shapes in {canonical_json}")

    valid = np.isfinite(object_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
    valid &= image_points[:, 0] >= 0
    valid &= image_points[:, 1] >= 0
    obj = object_points[valid].astype(np.float64)
    img = image_points[valid].astype(np.float64)
    if len(obj) < 6:
        raise RuntimeError(f"Too few valid landmarks for PnP: {canonical_json}")

    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        intrinsics,
        dist,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=200,
        reprojectionError=6.0,
        confidence=0.999,
    )
    if ok:
        inlier_idx = inliers[:, 0] if inliers is not None and len(inliers) >= 6 else np.arange(len(obj))
        ok_refine, rvec, tvec = cv2.solvePnP(
            obj[inlier_idx],
            img[inlier_idx],
            intrinsics,
            dist,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        ok = bool(ok_refine)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            intrinsics,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None
    if not ok:
        raise RuntimeError(f"solvePnP failed for {canonical_json}")

    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics, dist)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)
    inlier_count = int(len(inliers)) if inliers is not None else int(len(obj))
    return {
        "rotation": rotation.astype(np.float64),
        "translation_mm": (tvec.reshape(3) * 1000.0).astype(np.float64),
        "object_points_m": object_points,
        "image_points": image_points,
        "inliers": inlier_count,
        "residual_mean_px": float(np.mean(residuals)),
        "residual_median_px": float(np.median(residuals)),
        "residual_p95_px": float(np.percentile(residuals, 95)),
    }


def pnp_rotation_to_eval_head_frame(rotation_opencv):
    """Convert FLAME/OpenCV canonical head axes into the eval head-pose basis."""
    flame_to_opencv = np.diag([1.0, -1.0, -1.0])
    return rotation_opencv @ flame_to_opencv


def apply_famos_rotation_convention(rotation, convention):
    if convention == "raw":
        return rotation
    if convention == "head_yz_flip":
        return rotation @ HEAD_YZ_FLIP
    raise ValueError(f"Unknown FaMoS rotation convention: {convention}")


def avatarbox_median_depth(pnp_data, intrinsics):
    object_points = pnp_data["object_points_m"]
    image_points = pnp_data["image_points"]
    rotation = pnp_data["rotation"]
    focal = 0.5 * (float(intrinsics[0, 0]) + float(intrinsics[1, 1]))
    rotated = (rotation @ object_points.T).T
    estimates = []
    for first, second in AVATAR_MEDIAN_PAIRS:
        observed_px = float(np.linalg.norm(image_points[first] - image_points[second]))
        metric_xy_mm = float(np.linalg.norm(rotated[first, :2] - rotated[second, :2]) * 1000.0)
        if observed_px > 3.0 and metric_xy_mm > 1.0:
            estimates.append(focal * metric_xy_mm / observed_px)
    if not estimates:
        return float("nan"), []
    return float(np.median(estimates)), estimates


def recover_depth_norm_slot(row, anchor_gt):
    pred = row_translation(row, "pred")
    z0 = max(float(anchor_gt[2]), 1e-6)
    return {
        "dx": (pred[0] - anchor_gt[0]) / z0,
        "dy": (pred[1] - anchor_gt[1]) / z0,
        "log_z_ratio": float(math.log(max(float(pred[2]), 1e-6) / z0)),
    }


def decode_depth_norm_slot(slot, anchor_avatar):
    z0 = max(float(anchor_avatar[2]), 1e-6)
    return np.asarray(
        [
            anchor_avatar[0] + slot["dx"] * z0,
            anchor_avatar[1] + slot["dy"] * z0,
            z0 * math.exp(float(np.clip(slot["log_z_ratio"], -20.0, 20.0))),
        ],
        dtype=np.float64,
    )


def angular_metrics(pred_rotations, gt_rotations):
    pred_t = torch.from_numpy(np.stack(pred_rotations)).float()
    gt_t = torch.from_numpy(np.stack(gt_rotations)).float()
    y_gt, p_gt, r_gt = base.biwi_euler_deg_batch(gt_t)
    y_pred, p_pred, r_pred = base.biwi_euler_deg_batch(pred_t)
    yaw = base.angular_error(y_gt, y_pred).numpy()
    pitch = base.angular_error(p_gt, p_pred).numpy()
    roll = base.angular_error(r_gt, r_pred).numpy()
    geod = geodesic_deg_torch(pred_t, gt_t).numpy()
    return yaw, pitch, roll, geod


def summarize(rows):
    if not rows:
        raise RuntimeError("No rows to summarize")
    fields = (
        "yaw_error_deg",
        "pitch_error_deg",
        "roll_error_deg",
        "rot_mean_error_deg",
        "geodesic_error_deg",
        "tx_error_mm",
        "ty_error_mm",
        "tz_error_mm",
        "translation_l2_error_mm",
    )
    values = {field: np.asarray([float(row[field]) for row in rows], dtype=np.float64) for field in fields}
    return {
        "frames": len(rows),
        "yaw_mae_deg": float(np.mean(values["yaw_error_deg"])),
        "pitch_mae_deg": float(np.mean(values["pitch_error_deg"])),
        "roll_mae_deg": float(np.mean(values["roll_error_deg"])),
        "maer_deg": float(np.mean(values["rot_mean_error_deg"])),
        "geodesic_mean_deg": float(np.mean(values["geodesic_error_deg"])),
        "tx_mae_mm": float(np.mean(values["tx_error_mm"])),
        "ty_mae_mm": float(np.mean(values["ty_error_mm"])),
        "tz_mae_mm": float(np.mean(values["tz_error_mm"])),
        "translation_l2_mae_mm": float(np.mean(values["translation_l2_error_mm"])),
    }


def write_summary_md(path, method_name, protocol, summary, metadata):
    lines = [
        f"# {method_name} on {protocol}",
        "",
        "| Method | Frames | Yaw | Pitch | Roll | MAEr | GE | Tx | Ty | Tz | MAEt | ADD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {method_name} | {summary['frames']} | {summary['yaw_mae_deg']:.4f} | "
            f"{summary['pitch_mae_deg']:.4f} | {summary['roll_mae_deg']:.4f} | "
            f"{summary['maer_deg']:.4f} | {summary['geodesic_mean_deg']:.4f} | "
            f"{summary['tx_mae_mm']:.4f} | {summary['ty_mae_mm']:.4f} | "
            f"{summary['tz_mae_mm']:.4f} | {summary['translation_l2_mae_mm']:.4f} | n/a |"
        ),
        "",
        "## Protocol",
        "",
        "- Enrollment anchor is estimated from enrollment RGB landmarks and subject-specific FLAME geometry.",
        "- AvatarBox depth uses the frozen six landmark pairs.",
        "- Translation is decoded from the saved VGGT-HPE depth-normalized relative slot with the AvatarBox anchor.",
        "- Rotation is composed from the recovered VGGT-HPE relative rotation and the enrollment PnP rotation.",
        "- GT pose is used only for metric scoring and to reconstruct the saved relative slot from the previous GT-anchor export.",
        "",
        "## Files",
        "",
        f"- Predictions: `{metadata['predictions_csv']}`",
        f"- Subject debug: `{metadata['subject_debug_csv']}`",
        f"- Summary CSV: `{metadata['summary_csv']}`",
    ]
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

    out_dir = args.out_root / args.protocol
    pred_csv = out_dir / "avatarbox_median_predictions.csv"
    debug_csv = out_dir / "avatarbox_median_subject_debug.csv"
    summary_csv = out_dir / "avatarbox_median_summary.csv"
    summary_md = out_dir / "avatarbox_median_summary.md"
    metadata_json = out_dir / "avatarbox_median_metadata.json"

    cache = {}
    pred_rows = []
    debug_rows = []
    pred_rotations = []
    gt_rotations = []
    per_subject_counts = defaultdict(int)

    for row in rows:
        subject = row["subject"]
        if subject not in cache:
            adapter_subject_dir = args.adapter_root / args.protocol / subject
            anchor_pose = adapter_subject_dir / f"frame_{int(row['anchor_frame']):05d}_pose.txt"
            intrinsics_path = adapter_subject_dir / "rgb.cal"
            canonical_json = Path(row["canonical_flame_json"])
            if not canonical_json.is_file():
                raise FileNotFoundError(canonical_json)
            R_anchor_gt_raw, t_anchor_gt = load_pose_txt(anchor_pose)
            R_anchor_gt = apply_famos_rotation_convention(R_anchor_gt_raw, args.famos_rotation_convention)
            intrinsics = load_intrinsics(intrinsics_path)
            pnp = solve_enrollment_pnp(canonical_json, intrinsics, args.canonical_reference)
            z_avatar, pair_estimates = avatarbox_median_depth(pnp, intrinsics)
            if not np.isfinite(z_avatar):
                raise RuntimeError(f"AvatarBox depth failed for subject {subject}")
            z_pnp = max(float(pnp["translation_mm"][2]), 1e-6)
            alpha = z_avatar / z_pnp
            anchor_avatar = np.asarray(
                [
                    alpha * pnp["translation_mm"][0],
                    alpha * pnp["translation_mm"][1],
                    z_avatar,
                ],
                dtype=np.float64,
            )
            cache[subject] = {
                "R_anchor_gt": R_anchor_gt,
                "t_anchor_gt": t_anchor_gt,
                "R_anchor_pnp": pnp["rotation"],
                "R_anchor_eval": apply_famos_rotation_convention(
                    pnp_rotation_to_eval_head_frame(pnp["rotation"]),
                    args.famos_rotation_convention,
                ),
                "t_anchor_pnp": pnp["translation_mm"],
                "anchor_avatar": anchor_avatar,
                "z_avatar": z_avatar,
                "pair_estimates": pair_estimates,
                "pnp": pnp,
                "canonical_json": str(canonical_json),
            }
            debug_rows.append(
                {
                    "subject": subject,
                    "enrollment_id": row.get("enrollment_id", ""),
                    "source_key": row.get("source_key", ""),
                    "famos_rotation_convention": args.famos_rotation_convention,
                    "canonical_flame_json": str(canonical_json),
                    "z_gt_anchor_mm": f"{t_anchor_gt[2]:.9f}",
                    "z_pnp_mm": f"{pnp['translation_mm'][2]:.9f}",
                    "z_avatarbox_mm": f"{z_avatar:.9f}",
                    "anchor_avatar_x_mm": f"{anchor_avatar[0]:.9f}",
                    "anchor_avatar_y_mm": f"{anchor_avatar[1]:.9f}",
                    "anchor_avatar_z_mm": f"{anchor_avatar[2]:.9f}",
                    "pnp_inliers": pnp["inliers"],
                    "pnp_residual_mean_px": f"{pnp['residual_mean_px']:.9f}",
                    "pnp_residual_median_px": f"{pnp['residual_median_px']:.9f}",
                    "pnp_residual_p95_px": f"{pnp['residual_p95_px']:.9f}",
                    "avatar_pair_estimates_mm": " ".join(f"{value:.6f}" for value in pair_estimates),
                }
            )

        data = cache[subject]
        R_gt = apply_famos_rotation_convention(row_matrix(row, "gt"), args.famos_rotation_convention)
        t_gt = row_translation(row, "gt")
        R_pred_gt_anchor = apply_famos_rotation_convention(row_matrix(row, "pred"), args.famos_rotation_convention)
        R_rel = R_pred_gt_anchor @ data["R_anchor_gt"].T
        R_pred = R_rel @ data["R_anchor_eval"]
        slot = recover_depth_norm_slot(row, data["t_anchor_gt"])
        t_pred = decode_depth_norm_slot(slot, data["anchor_avatar"])

        pred_rotations.append(R_pred)
        gt_rotations.append(R_gt)
        signed = t_pred - t_gt
        out = dict(row)
        out["method"] = args.method_name
        out["variant"] = args.method_name
        for rr in range(3):
            for cc in range(3):
                out[f"pred_r{rr}{cc}"] = f"{R_pred[rr, cc]:.9f}"
                out[f"gt_r{rr}{cc}"] = f"{R_gt[rr, cc]:.9f}"
        out["pred_tx_mm"] = f"{t_pred[0]:.9f}"
        out["pred_ty_mm"] = f"{t_pred[1]:.9f}"
        out["pred_tz_mm"] = f"{t_pred[2]:.9f}"
        out["tx_error_mm"] = f"{abs(signed[0]):.9f}"
        out["ty_error_mm"] = f"{abs(signed[1]):.9f}"
        out["tz_error_mm"] = f"{abs(signed[2]):.9f}"
        out["translation_l2_error_mm"] = f"{float(np.linalg.norm(signed)):.9f}"
        out["anchor_avatar_x_mm"] = f"{data['anchor_avatar'][0]:.9f}"
        out["anchor_avatar_y_mm"] = f"{data['anchor_avatar'][1]:.9f}"
        out["anchor_avatar_z_mm"] = f"{data['anchor_avatar'][2]:.9f}"
        pred_rows.append(out)
        per_subject_counts[subject] += 1

    yaw, pitch, roll, geod = angular_metrics(pred_rotations, gt_rotations)
    for row, y, p, r, g in zip(pred_rows, yaw, pitch, roll, geod):
        row["yaw_error_deg"] = f"{float(y):.9f}"
        row["pitch_error_deg"] = f"{float(p):.9f}"
        row["roll_error_deg"] = f"{float(r):.9f}"
        row["rot_mean_error_deg"] = f"{float((y + p + r) / 3.0):.9f}"
        row["geodesic_error_deg"] = f"{float(g):.9f}"

    summary = summarize(pred_rows)
    summary_row = {
        "scope": "overall",
        "subject": "all",
        "sequence": "all",
        "view": "all",
        "variant": args.method_name,
        **summary,
    }
    metadata = {
        "protocol": args.protocol,
        "method": args.method_name,
        "canonical_reference": args.canonical_reference,
        "famos_rotation_convention": args.famos_rotation_convention,
        "ready_pose_predictions": str(ready_csv),
        "adapter_root": str(args.adapter_root / args.protocol),
        "frames": len(pred_rows),
        "subjects": len(per_subject_counts),
        "fixed_avatarbox_pairs": AVATAR_MEDIAN_PAIRS,
        "predictions_csv": str(pred_csv),
        "subject_debug_csv": str(debug_csv),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "note": (
            "The saved GT-anchor prediction is used only to recover the raw depth_norm_delta_t "
            "relative slot "
        ),
    }

    fieldnames = list(pred_rows[0])
    write_csv(pred_csv, pred_rows, fieldnames)
    write_csv(debug_csv, debug_rows, list(debug_rows[0]))
    write_csv(summary_csv, [summary_row], list(summary_row))
    write_json(metadata_json, metadata)
    write_summary_md(summary_md, args.method_name, args.protocol, summary, metadata)

    print(json.dumps({"summary": summary_row, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
