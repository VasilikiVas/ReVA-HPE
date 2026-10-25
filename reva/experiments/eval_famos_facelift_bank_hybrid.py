#!/usr/bin/env python3
"""Evaluate completed FaceLift metric anchor banks on FaMoS ready rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reva.experiments import eval_famos_flame_bank_hybrid as flame_eval


OUT_ROOT = Path("outputs")
FAMOS_ALIGNMENT_ROOT = Path("data/famos_enrollment_alignment")
DEFAULT_EXP = "flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit"
HEAD_YZ_FLIP = np.diag([1.0, -1.0, -1.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="famos_view00_mtcnn_visible_no091")
    parser.add_argument(
        "--ready-root",
        type=Path,
        default=OUT_ROOT / "refinement_inputs_cross_dataset",
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=FAMOS_ALIGNMENT_ROOT,
        help="Root containing BIWI-style metric FaceLift banks, or manual-view00 banks when --bank-layout=manual_view00.",
    )
    parser.add_argument(
        "--bank-name",
        default="metric_anchor_bank_front20",
        help="Single bank name or comma-separated names to evaluate as one pool.",
    )
    parser.add_argument(
        "--bank-layout",
        choices=("auto", "metric_famos_alignment", "manual_metric_raw_subject", "manual_view00"),
        default="auto",
        help=(
            "metric_famos_alignment reads rotation_h2c_opencv/translation_h2c_m "
            "directly from anchors.json (BIWI-style banks). manual_metric_raw_subject "
            "reads per-raw-subject metric banks. manual_view00 reads label-anchored "
            "view-00 banks."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT / "eval_famos_head_pose" / "facelift_bank_hybrid",
    )
    parser.add_argument("--adapter-subject", default=None, help="Optional adapter subject id, e.g. 001.")
    parser.add_argument(
        "--raw-subject",
        default=None,
        help="Optional raw FaMoS subject id, e.g. FaMoS_subject_079. This applies one manual avatar to all ready rows of that person.",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=flame_eval.PROJECT_ROOT / "checkpoints" / "VGGT-1B" / "model.pt")
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=flame_eval.PROJECT_ROOT / "training" / "logs" / DEFAULT_EXP / "ckpts" / "checkpoint_410.pt",
    )
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument("--translation-target-type", default="depth_norm_delta_t")
    parser.add_argument("--translation-scale", type=float, default=1000.0)
    parser.add_argument("--anchor-count", type=int, default=20)
    parser.add_argument(
        "--anchor-pose-mode",
        choices=("centered_icosphere", "fit_conditioned_se3"),
        default="fit_conditioned_se3",
    )
    parser.add_argument("--anchor-dx-over-z-range", type=float, default=0.10)
    parser.add_argument("--anchor-dy-over-z-range", type=float, default=0.05)
    parser.add_argument("--anchor-log-z-range", type=float, default=0.10)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--mtcnn-ad", type=float, default=0.4)
    parser.add_argument("--ransac-rot-threshold-deg", type=float, default=10.0)
    parser.add_argument("--ransac-trans-threshold-mm", type=float, default=50.0)
    parser.add_argument("--ransac-min-inliers", type=int, default=3)
    parser.add_argument("--method-name", default="ReVA-HPE (Direct-5r)")
    parser.add_argument("--bank-translation-mode", choices=("metric", "rotation_only"), default="metric")
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="GT rotation convention for FaMoS scoring. head_yz_flip uses R_gt @ diag(1,-1,-1).",
    )
    parser.add_argument(
        "--strict-manual-source-key",
        action="store_true",
        help="Require ready rows to use exactly the same subject/sequence/view as the manual anchor manifest row.",
    )
    parser.add_argument(
        "--require-gt-free-bank",
        action="store_true",
        help="Fail if anchors.json declares dataset GT pose/depth usage. Useful for practical/final protocol checks.",
    )
    parser.add_argument("--max-visualizations", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def _local_box_from_xyxy(box):
    return [float(box[0]), float(box[2]), float(box[1]), float(box[3])]


def _matrix_from_6d(text):
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected 6 values for rotation-6D, got {values.shape}: {text}")
    first_two_columns = values.reshape(3, 2)
    first = first_two_columns[:, 0]
    second = first_two_columns[:, 1]
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second - first * float(np.dot(first, second))
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _vector_from_text(text):
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected 3 values for vector, got {values.shape}: {text}")
    return values


def _apply_rotation_convention(rotation, convention):
    if convention == "raw":
        return rotation
    if convention == "head_yz_flip":
        return rotation @ HEAD_YZ_FLIP
    raise ValueError(f"Unknown FaMoS rotation convention: {convention}")


def _manual_label_pose(bank_root: Path, raw_subject: str):
    manifest_path = bank_root / "famos_manual_anchor_manifest.csv"
    labels_path = bank_root / "labels_valid_views.csv"
    with manifest_path.open(newline="") as handle:
        manifest_rows = {row["subject"]: row for row in csv.DictReader(handle)}
    manifest = manifest_rows.get(raw_subject)
    if manifest is None:
        raise KeyError(f"No manual anchor manifest row for {raw_subject}: {manifest_path}")
    key = (manifest["subject"], manifest["sequence"], manifest["frame"], manifest["view_idx"])
    with labels_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["subject"], row["sequence"], row["frame"], row["view_idx"]) == key:
                return {
                    "manifest": manifest,
                    "manual_source_key": f"{manifest['subject']}|{manifest['sequence']}|{manifest['view_idx']}",
                    "rotation_raw": _matrix_from_6d(row["R_cam_head_6d"]),
                    "translation_mm": _vector_from_text(row["t_cam_head_mm"]),
                }
    raise KeyError(f"No label row matching manual anchor {key}: {labels_path}")


def _metric_bank_dir(args, adapter_subject: str, bank_name: str):
    subject_name = f"subject{int(adapter_subject):03d}" if str(adapter_subject).isdigit() else str(adapter_subject)
    candidates = [
        args.bank_root / args.protocol / subject_name / bank_name,
        args.bank_root / subject_name / bank_name,
    ]
    for candidate in candidates:
        if (candidate / "anchors.json").is_file():
            return candidate
    return candidates[0]


def _manual_metric_bank_dir(args, raw_subject: str, bank_name: str):
    candidates = [
        args.bank_root / args.protocol / raw_subject / bank_name,
        args.bank_root / raw_subject / bank_name,
    ]
    for candidate in candidates:
        if (candidate / "anchors.json").is_file():
            return candidate
    return candidates[0]


def _load_metric_bank(args, adapter_subject: str, bank_name: str):
    bank_dir = _metric_bank_dir(args, adapter_subject, bank_name)
    anchors_json = bank_dir / "anchors.json"
    if not anchors_json.is_file():
        raise FileNotFoundError(f"Missing BIWI-style metric FaceLift bank: {anchors_json}")
    data = json.loads(anchors_json.read_text())
    if args.require_gt_free_bank and bool(data.get("uses_dataset_pose_or_depth_ground_truth", False)):
        raise RuntimeError(
            f"{anchors_json} declares uses_dataset_pose_or_depth_ground_truth=True. "
            "This is not a clean practical bank."
        )
    images_rgb = []
    boxes = []
    rotations = []
    translations_m = []
    directions = []
    offsets = []
    for anchor in data["anchors"]:
        image_bgr = cv2.imread(str(bank_dir / anchor["image"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read FaceLift anchor image: {bank_dir / anchor['image']}")
        translation_m = anchor.get("translation_h2c_m")
        if translation_m is None:
            raise ValueError(
                f"{anchors_json} is not a BIWI-style metric bank: anchor {anchor.get('anchor_id')} "
                "does not contain translation_h2c_m."
            )
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        boxes.append(_local_box_from_xyxy(anchor["head_box_xyxy"]))
        rotations.append(np.asarray(anchor["rotation_h2c_opencv"], dtype=np.float64))
        translations_m.append(np.asarray(translation_m, dtype=np.float64))
        directions.append(np.asarray(anchor["icosphere_camera_direction"], dtype=np.float64))
        offsets.append(
            np.asarray(
                [
                    float(anchor.get("dx_over_z", 0.0)),
                    float(anchor.get("dy_over_z", 0.0)),
                    float(anchor.get("log_z_offset", 0.0)),
                ],
                dtype=np.float64,
            )
        )
    quality = data.get("quality", {})
    schedule = data.get("schedule", {})
    return {
        "layout": "metric_famos_alignment",
        "images_rgb": images_rgb,
        "boxes": boxes,
        "rotations": np.asarray(rotations, dtype=np.float64),
        "translations_m": np.asarray(translations_m, dtype=np.float64),
        "directions": np.asarray(directions, dtype=np.float64),
        "offsets": np.asarray(offsets, dtype=np.float64),
        "anchor_distance": float(schedule.get("base_depth_m") or 0.0),
        "head_radius": float(quality.get("head_radius_model_units") or 0.0),
        "uses_dataset_pose_or_depth_ground_truth": bool(data.get("uses_dataset_pose_or_depth_ground_truth", False)),
        "uses_famos_enrollment_gt_rotation": bool(data.get("uses_famos_enrollment_gt_rotation", False)),
        "rotation_transport_type": str(data.get("rotation_transport", {}).get("type", "")),
        "bank_dir": str(bank_dir),
        "manual_source_key": "",
        "manual_anchor_sequence": "",
        "manual_anchor_frame": "",
        "manual_anchor_view_idx": "",
        "unit_to_mm": float("nan"),
    }


def _load_manual_metric_bank(args, raw_subject: str, bank_name: str):
    bank_dir = _manual_metric_bank_dir(args, raw_subject, bank_name)
    anchors_json = bank_dir / "anchors.json"
    if not anchors_json.is_file():
        raise FileNotFoundError(f"Missing manual-view00 BIWI-style metric FaceLift bank: {anchors_json}")
    data = json.loads(anchors_json.read_text())
    if args.require_gt_free_bank and bool(data.get("uses_dataset_pose_or_depth_ground_truth", False)):
        raise RuntimeError(
            f"{anchors_json} declares uses_dataset_pose_or_depth_ground_truth=True. "
            "This is not a clean practical bank."
        )
    images_rgb = []
    boxes = []
    rotations = []
    translations_m = []
    directions = []
    offsets = []
    for anchor in data["anchors"]:
        image_bgr = cv2.imread(str(bank_dir / anchor["image"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read FaceLift anchor image: {bank_dir / anchor['image']}")
        translation_m = anchor.get("translation_h2c_m")
        if translation_m is None:
            raise ValueError(
                f"{anchors_json} is not a BIWI-style metric bank: anchor {anchor.get('anchor_id')} "
                "does not contain translation_h2c_m."
            )
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        boxes.append(_local_box_from_xyxy(anchor["head_box_xyxy"]))
        rotations.append(np.asarray(anchor["rotation_h2c_opencv"], dtype=np.float64))
        translations_m.append(np.asarray(translation_m, dtype=np.float64))
        directions.append(np.asarray(anchor["icosphere_camera_direction"], dtype=np.float64))
        offsets.append(
            np.asarray(
                [
                    float(anchor.get("dx_over_z", 0.0)),
                    float(anchor.get("dy_over_z", 0.0)),
                    float(anchor.get("log_z_offset", 0.0)),
                ],
                dtype=np.float64,
            )
        )
    quality = data.get("quality", {})
    schedule = data.get("schedule", {})
    return {
        "layout": "manual_metric_raw_subject",
        "images_rgb": images_rgb,
        "boxes": boxes,
        "rotations": np.asarray(rotations, dtype=np.float64),
        "translations_m": np.asarray(translations_m, dtype=np.float64),
        "directions": np.asarray(directions, dtype=np.float64),
        "offsets": np.asarray(offsets, dtype=np.float64),
        "anchor_distance": float(schedule.get("base_depth_m") or 0.0),
        "head_radius": float(quality.get("head_radius_model_units") or 0.0),
        "uses_dataset_pose_or_depth_ground_truth": bool(data.get("uses_dataset_pose_or_depth_ground_truth", False)),
        "uses_famos_enrollment_gt_rotation": bool(data.get("uses_famos_enrollment_gt_rotation", False)),
        "rotation_transport_type": str(data.get("rotation_transport", {}).get("type", "")),
        "bank_dir": str(bank_dir),
        "raw_subject": raw_subject,
        "manual_source_key": "",
        "manual_anchor_sequence": "",
        "manual_anchor_frame": "",
        "manual_anchor_view_idx": "",
        "unit_to_mm": float("nan"),
    }


def _load_manual_bank(args, raw_subject: str, bank_name: str):
    bank_root = args.bank_root
    bank_dir = bank_root / raw_subject / bank_name
    anchors_json = bank_dir / "anchors.json"
    if not anchors_json.is_file():
        raise FileNotFoundError(
            f"Missing FaceLift render bank for raw subject {raw_subject}: {anchors_json}. "
            "This evaluator intentionally does not fall back to FLAME anchors."
        )
    data = json.loads(anchors_json.read_text())
    if args.require_gt_free_bank and bool(data.get("uses_dataset_pose_or_depth_ground_truth", False)):
        raise RuntimeError(
            f"{anchors_json} declares uses_dataset_pose_or_depth_ground_truth=True. "
            "This is not a clean practical bank."
        )
    label_pose = _manual_label_pose(bank_root, raw_subject)
    label_rotation = _apply_rotation_convention(label_pose["rotation_raw"], args.famos_rotation_convention)
    label_translation_mm = label_pose["translation_mm"]
    anchors = data["anchors"]
    canonical_id = int(data.get("schedule", {}).get("canonical_anchor_id", 0))
    canonical_units = np.asarray(
        anchors[canonical_id].get("translation_h2c_facelift_units"),
        dtype=np.float64,
    )
    if canonical_units.shape != (3,) or abs(float(canonical_units[2])) < 1e-9:
        raise ValueError(f"Invalid canonical FaceLift translation units in {anchors_json}")
    unit_to_mm = float(label_translation_mm[2]) / float(canonical_units[2])
    unit_offset_mm = label_translation_mm - unit_to_mm * canonical_units
    images_rgb = []
    boxes = []
    rotations = []
    translations_m = []
    directions = []
    offsets = []
    for anchor in anchors:
        image_bgr = cv2.imread(str(bank_dir / anchor["image"]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read FaceLift anchor image: {bank_dir / anchor['image']}")
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        boxes.append(_local_box_from_xyxy(anchor["head_box_xyxy"]))
        relative_rotation = np.asarray(anchor["relative_rotation_h2c_opencv"], dtype=np.float64)
        rotations.append(relative_rotation @ label_rotation)
        translation_m = anchor.get("translation_h2c_m")
        if translation_m is None:
            units = np.asarray(anchor["translation_h2c_facelift_units"], dtype=np.float64)
            translation_mm = unit_to_mm * units + unit_offset_mm
            translation_m = translation_mm / 1000.0
        translations_m.append(np.asarray(translation_m, dtype=np.float64))
        directions.append(np.asarray(anchor["icosphere_camera_direction"], dtype=np.float64))
        offsets.append(
            np.asarray(
                [
                    float(anchor.get("dx_over_z", 0.0)),
                    float(anchor.get("dy_over_z", 0.0)),
                    float(anchor.get("log_z_offset", 0.0)),
                ],
                dtype=np.float64,
            )
        )
    quality = data.get("quality", {})
    schedule = data.get("schedule", {})
    return {
        "layout": "manual_view00",
        "images_rgb": images_rgb,
        "boxes": boxes,
        "rotations": np.asarray(rotations, dtype=np.float64),
        "translations_m": np.asarray(translations_m, dtype=np.float64),
        "directions": np.asarray(directions, dtype=np.float64),
        "offsets": np.asarray(offsets, dtype=np.float64),
        "anchor_distance": float(schedule.get("base_depth_m") or 0.0),
        "head_radius": float(quality.get("head_radius_model_units") or 0.0),
        "uses_dataset_pose_or_depth_ground_truth": bool(data.get("uses_dataset_pose_or_depth_ground_truth", False)),
        "uses_famos_enrollment_gt_rotation": bool(data.get("uses_famos_enrollment_gt_rotation", False)),
        "rotation_transport_type": str(data.get("rotation_transport", {}).get("type", "")),
        "bank_dir": str(bank_dir),
        "raw_subject": raw_subject,
        "manual_source_key": label_pose["manual_source_key"],
        "manual_anchor_sequence": label_pose["manifest"]["sequence"],
        "manual_anchor_frame": label_pose["manifest"]["frame"],
        "manual_anchor_view_idx": label_pose["manifest"]["view_idx"],
        "unit_to_mm": unit_to_mm,
    }


def load_or_render_anchors(args, _flame, rows, _device):
    adapter_subject = rows[0]["subject"]
    raw_subject = rows[0]["source_key"].split("|")[0]
    names = [name.strip() for name in args.bank_name.split(",") if name.strip()]
    loaded = []
    for name in names:
        metric_exists = (_metric_bank_dir(args, adapter_subject, name) / "anchors.json").is_file()
        manual_metric_exists = (_manual_metric_bank_dir(args, raw_subject, name) / "anchors.json").is_file()
        if args.bank_layout == "metric_famos_alignment" or (
            args.bank_layout == "auto" and metric_exists
        ):
            loaded.append(_load_metric_bank(args, adapter_subject, name))
        elif args.bank_layout == "manual_metric_raw_subject" or (
            args.bank_layout == "auto" and manual_metric_exists
        ):
            loaded.append(_load_manual_metric_bank(args, raw_subject, name))
        elif args.bank_layout == "manual_view00" or args.bank_layout == "auto":
            loaded.append(_load_manual_bank(args, raw_subject, name))
        else:
            raise RuntimeError(f"Could not resolve bank layout for {name}")
    manual_source_key = loaded[0]["manual_source_key"]
    if args.strict_manual_source_key:
        if loaded[0]["layout"] != "manual_view00":
            raise RuntimeError("--strict-manual-source-key is only valid for --bank-layout=manual_view00")
        source_keys = sorted({row["source_key"] for row in rows})
        if source_keys != [manual_source_key]:
            raise RuntimeError(
                "Manual FaceLift anchor source-key mismatch. "
                f"ready source_keys={source_keys}, manual_source_key={manual_source_key}"
            )
    images_rgb = [image for bank in loaded for image in bank["images_rgb"]]
    boxes = [box for bank in loaded for box in bank["boxes"]]
    rotations = np.concatenate([bank["rotations"] for bank in loaded], axis=0)
    translations_m = np.concatenate([bank["translations_m"] for bank in loaded], axis=0)
    directions = np.concatenate([bank["directions"] for bank in loaded], axis=0)
    offsets = np.concatenate([bank["offsets"] for bank in loaded], axis=0)
    args.anchor_count = len(images_rgb)
    anchor_distance = float(np.mean([bank["anchor_distance"] for bank in loaded]))
    head_radius = float(np.mean([bank["head_radius"] for bank in loaded]))
    unit_values = [bank["unit_to_mm"] for bank in loaded if not np.isnan(bank["unit_to_mm"])]
    if loaded[0]["layout"] == "metric_famos_alignment":
        source_match_mode = "adapter_subject"
    elif loaded[0]["layout"] == "manual_metric_raw_subject":
        source_match_mode = "raw_subject_metric"
    elif args.strict_manual_source_key:
        source_match_mode = "strict_manual_source_key"
    else:
        source_match_mode = "raw_subject"
    args._bank_metadata = {
        "bank_layout": ",".join(sorted({bank["layout"] for bank in loaded})),
        "bank_adapter_subject": adapter_subject,
        "bank_raw_subject": raw_subject,
        "bank_root": str(args.bank_root),
        "bank_name": args.bank_name,
        "bank_dirs": " ".join(bank["bank_dir"] for bank in loaded),
        "bank_manual_source_key": manual_source_key,
        "bank_manual_anchor_sequence": loaded[0]["manual_anchor_sequence"],
        "bank_manual_anchor_frame": loaded[0]["manual_anchor_frame"],
        "bank_manual_anchor_view_idx": loaded[0]["manual_anchor_view_idx"],
        "bank_uses_dataset_pose_or_depth_ground_truth": int(any(bank["uses_dataset_pose_or_depth_ground_truth"] for bank in loaded)),
        "bank_uses_famos_enrollment_gt_rotation": int(any(bank["uses_famos_enrollment_gt_rotation"] for bank in loaded)),
        "bank_rotation_transport_type": ",".join(sorted({bank["rotation_transport_type"] for bank in loaded})),
        "bank_unit_to_mm": float(np.mean(unit_values)) if unit_values else "",
        "bank_source_match_mode": source_match_mode,
    }
    return (
        images_rgb,
        boxes,
        rotations,
        translations_m,
        directions,
        offsets,
        anchor_distance,
        head_radius,
    )


def main():
    flame_eval.parse_args = parse_args
    flame_eval.load_or_render_anchors = load_or_render_anchors
    flame_eval.main()


if __name__ == "__main__":
    main()
