#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_FAMOS_ROOT = Path("data/famos_head_pose_test")
DEFAULT_OUT_ROOT = Path("outputs/eval_famos_head_pose/enrollment")


def parse_args():
    parser = argparse.ArgumentParser(description="Build FaMoS enrollment manifests")
    parser.add_argument("--famos-root", type=Path, default=DEFAULT_FAMOS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--protocol",
        choices=("group_anchors", "subject_single", "both"),
        default="both",
        help=(
            "group_anchors matches the GT-anchor evaluator "
        ),
    )
    return parser.parse_args()


def safe_id(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def load_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.keys()}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_rgb_cal(path, intrinsics):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{float(value):.12g}" for value in row) for row in intrinsics]
    path.write_text("\n".join(lines) + "\n")


def write_entry_metadata(path, row):
    payload = {
        "protocol": row["protocol"],
        "enrollment_id": row["enrollment_id"],
        "label_index": int(row["label_index"]),
        "subject": row["subject"],
        "sequence": row["sequence"],
        "view_idx": int(row["view_idx"]),
        "frame": row["frame"],
        "image_path": row["image_path"],
        "camera_path": row["camera_path"],
        "target_count": int(row["target_count"]),
        "uses_dataset_pose_or_depth_ground_truth_for_selection": False,
        "selection_rule": row["selection_rule"],
        "intrinsics": json.loads(row["K_json"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def make_row(args, labels, csv_rows, idx, protocol, enrollment_id, target_count, selection_rule):
    subject = str(labels["subject"][idx])
    sequence = str(labels["sequence"][idx])
    view_idx = int(labels["view_idx"][idx])
    frame = str(labels["frame"][idx])
    image_relpath = str(labels["image_relpath"][idx])
    camera_relpath = str(labels["camera_relpath"][idx])
    entry_root = args.out_root / protocol / enrollment_id
    flame_dir = entry_root / "flame"
    row = {
        "protocol": protocol,
        "enrollment_id": enrollment_id,
        "label_index": str(idx),
        "subject": subject,
        "sequence": sequence,
        "view_idx": str(view_idx),
        "frame": frame,
        "target_count": str(target_count),
        "image_relpath": image_relpath,
        "image_path": str(args.famos_root / image_relpath),
        "camera_relpath": camera_relpath,
        "camera_path": str(args.famos_root / camera_relpath),
        "entry_root": str(entry_root),
        "flame_dir": str(flame_dir),
        "rgb_cal_path": str(entry_root / "rgb.cal"),
        "entry_metadata_path": str(entry_root / "enrollment_entry.json"),
        "canonical_flame_json": str(flame_dir / "canonical_flame_landmarks.json"),
        "K_json": json.dumps(np.asarray(labels["K"][idx], dtype=float).tolist()),
        "R_cam_head_json": json.dumps(np.asarray(labels["R_cam_head"][idx], dtype=float).tolist()),
        "t_cam_head_mm_json": json.dumps(np.asarray(labels["t_cam_head_mm"][idx], dtype=float).tolist()),
        "selection_rule": selection_rule,
    }
    if "procrustes_rms_m" in csv_rows[idx]:
        row["procrustes_rms_m"] = csv_rows[idx]["procrustes_rms_m"]
    return row


def group_anchor_rows(args, labels, csv_rows):
    groups = defaultdict(list)
    for idx in range(len(labels["subject"])):
        key = (str(labels["subject"][idx]), str(labels["sequence"][idx]), int(labels["view_idx"][idx]))
        groups[key].append(idx)

    rows = []
    for subject, sequence, view_idx in sorted(groups):
        indices = sorted(groups[(subject, sequence, view_idx)], key=lambda i: int(str(labels["frame"][i])))
        anchor_idx = indices[0]
        seq_tag = "lr" if sequence == "head_rotation_left_right" else "ud"
        subject_tag = subject.replace("FaMoS_subject_", "s")
        enrollment_id = f"{subject_tag}_{seq_tag}_view{view_idx:02d}"
        rows.append(
            make_row(
                args,
                labels,
                csv_rows,
                anchor_idx,
                "group_anchors",
                enrollment_id,
                target_count=max(0, len(indices) - 1),
                selection_rule="first frame in subject/sequence/view group after sorting by frame id",
            )
        )
    return rows


def subject_single_rows(args, labels, csv_rows):
    by_subject = defaultdict(list)
    for idx in range(len(labels["subject"])):
        by_subject[str(labels["subject"][idx])].append(idx)

    rows = []
    sequence_rank = {"head_rotation_left_right": 0, "head_rotation_up_down": 1}
    for subject in sorted(by_subject):
        indices = sorted(
            by_subject[subject],
            key=lambda i: (
                sequence_rank.get(str(labels["sequence"][i]), 99),
                int(labels["view_idx"][i]),
                int(str(labels["frame"][i])),
            ),
        )
        anchor_idx = indices[0]
        subject_tag = subject.replace("FaMoS_subject_", "s")
        rows.append(
            make_row(
                args,
                labels,
                csv_rows,
                anchor_idx,
                "subject_single",
                subject_tag,
                target_count=sum(1 for i in by_subject[subject] if i != anchor_idx),
                selection_rule=(
                    "first available left-right sequence, lowest view_idx, lowest frame, "
                    "GT pose is not used"
                ),
            )
        )
    return rows


def materialize_assets(rows):
    for row in rows:
        write_rgb_cal(Path(row["rgb_cal_path"]), np.asarray(json.loads(row["K_json"]), dtype=float))
        write_entry_metadata(Path(row["entry_metadata_path"]), row)


def write_summary(path, manifests):
    with path.open("w") as handle:
        handle.write("# FaMoS Enrollment Manifests\n\n")
        handle.write("No GT pose/depth is used to select enrollment images.\n\n")
        for name, rows in manifests.items():
            handle.write(f"## {name}\n\n")
            handle.write(f"- Entries: {len(rows)}\n")
            handle.write(f"- CSV: `{name}_manifest.csv`\n")
            handle.write(f"- First entry: `{rows[0]['enrollment_id']}`\n")
            handle.write(f"- First image: `{rows[0]['image_path']}`\n")
            handle.write(f"- Selection: {rows[0]['selection_rule']}\n\n")


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    labels = load_npz(args.famos_root / "labels.npz")
    csv_rows = load_csv_rows(args.famos_root / "labels.csv")
    if len(csv_rows) != len(labels["subject"]):
        raise RuntimeError("labels.csv and labels.npz row counts differ")

    manifests = {}
    if args.protocol in ("group_anchors", "both"):
        manifests["group_anchors"] = group_anchor_rows(args, labels, csv_rows)
    if args.protocol in ("subject_single", "both"):
        manifests["subject_single"] = subject_single_rows(args, labels, csv_rows)

    for name, rows in manifests.items():
        materialize_assets(rows)
        write_csv(args.out_root / f"{name}_manifest.csv", rows)

    write_summary(args.out_root / "famos_enrollment_manifest_summary.md", manifests)
    metadata = {
        "famos_root": str(args.famos_root),
        "out_root": str(args.out_root),
        "protocols": {name: len(rows) for name, rows in manifests.items()},
        "uses_gt_pose_or_depth_for_selection": False,
    }
    (args.out_root / "famos_enrollment_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
