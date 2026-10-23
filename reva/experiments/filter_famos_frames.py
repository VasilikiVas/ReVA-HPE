#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS = PROJECT_ROOT / "evaluation" / "eval_scripts"
if str(EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS))

import eval_biwi_crop as base  # noqa: E402


DEFAULT_FAMOS_ROOT = Path("data/famos_head_pose_test")
DEFAULT_OUT_ROOT = Path("outputs/eval_famos_head_pose/subsets")


SUBSETS = {
    "front45_pitch30_roll45": {
        "description": "BIWI-like/front-facing FaMoS subset",
        "yaw_abs_max": 45.0,
        "pitch_abs_max": 30.0,
        "roll_abs_max": 45.0,
    },
    "front60_pitch35_roll60": {
        "description": "Extended-front FaMoS subset with moderate profile/roll",
        "yaw_abs_max": 60.0,
        "pitch_abs_max": 35.0,
        "roll_abs_max": 60.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--famos-root", type=Path, default=DEFAULT_FAMOS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def load_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key].copy() for key in data.keys()}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_euler(labels):
    rotations = torch.from_numpy(labels["R_cam_head"]).float()
    yaw, pitch, roll = base.biwi_euler_deg_batch(rotations)
    return yaw.numpy(), pitch.numpy(), roll.numpy()


def within(yaw, pitch, roll, spec):
    return (
        abs(float(yaw)) <= spec["yaw_abs_max"]
        and abs(float(pitch)) <= spec["pitch_abs_max"]
        and abs(float(roll)) <= spec["roll_abs_max"]
    )


def group_indices(labels):
    groups = defaultdict(list)
    for idx in range(len(labels["subject"])):
        key = (
            str(labels["subject"][idx]),
            str(labels["sequence"][idx]),
            int(labels["view_idx"][idx]),
        )
        groups[key].append(idx)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=lambda i: int(str(labels["frame"][i])))
    return dict(sorted(groups.items()))


def make_subset_rows(labels, csv_rows, yaw, pitch, roll, spec):
    rows = []
    kept_groups = 0
    skipped_anchor_groups = 0
    for key, indices in group_indices(labels).items():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        if not within(yaw[anchor], pitch[anchor], roll[anchor], spec):
            skipped_anchor_groups += 1
            continue
        selected = [
            idx for idx in indices[1:]
            if within(yaw[idx], pitch[idx], roll[idx], spec)
        ]
        if not selected:
            continue
        kept_groups += 1
        for idx, role in [(anchor, "anchor")] + [(idx, "query") for idx in selected]:
            source = csv_rows[idx]
            rows.append(
                {
                    "label_index": idx,
                    "role": role,
                    "subject": str(labels["subject"][idx]),
                    "sequence": str(labels["sequence"][idx]),
                    "view_idx": int(labels["view_idx"][idx]),
                    "frame": str(labels["frame"][idx]),
                    "image_relpath": str(labels["image_relpath"][idx]),
                    "yaw_deg": f"{float(yaw[idx]):.9f}",
                    "pitch_deg": f"{float(pitch[idx]):.9f}",
                    "roll_deg": f"{float(roll[idx]):.9f}",
                    "procrustes_rms_m": source.get("procrustes_rms_m", ""),
                }
            )
    return rows, kept_groups, skipped_anchor_groups


def summarize(rows, kept_groups, skipped_anchor_groups):
    query_rows = [row for row in rows if row["role"] == "query"]
    by_sequence = Counter(row["sequence"] for row in query_rows)
    by_view = Counter(row["view_idx"] for row in query_rows)
    return {
        "rows_total_including_anchors": len(rows),
        "query_frames": len(query_rows),
        "anchors": len(rows) - len(query_rows),
        "kept_groups": kept_groups,
        "skipped_anchor_groups": skipped_anchor_groups,
        "by_sequence": dict(sorted(by_sequence.items())),
        "by_view": {str(key): value for key, value in sorted(by_view.items())},
    }


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.famos_root / "labels.npz")
    csv_rows = load_csv_rows(args.famos_root / "labels.csv")
    yaw, pitch, roll = compute_euler(labels)
    fieldnames = (
        "label_index",
        "role",
        "subject",
        "sequence",
        "view_idx",
        "frame",
        "image_relpath",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "procrustes_rms_m",
    )

    metadata = {
        "famos_root": str(args.famos_root),
        "selection_uses_model_predictions": False,
        "selection_uses_model_errors": False,
        "selection_uses_famos_pose_labels": True,
        "subsets": {},
    }
    for name, spec in SUBSETS.items():
        rows, kept_groups, skipped_anchor_groups = make_subset_rows(
            labels, csv_rows, yaw, pitch, roll, spec
        )
        path = args.out_root / f"famos_{name}_indices.csv"
        write_csv(path, rows, fieldnames)
        metadata["subsets"][name] = {
            "path": str(path),
            "description": spec["description"],
            "yaw_abs_max": spec["yaw_abs_max"],
            "pitch_abs_max": spec["pitch_abs_max"],
            "roll_abs_max": spec["roll_abs_max"],
            **summarize(rows, kept_groups, skipped_anchor_groups),
        }

    (args.out_root / "famos_subset_metadata.json").write_text(json.dumps(metadata, indent=2))
    with (args.out_root / "famos_subset_summary.md").open("w") as handle:
        handle.write("# FaMoS Label-Only Face-Visible Subsets\n\n")
        handle.write("Selection uses only FaMoS camera-head pose labels, model predictions/errors are not used.\n\n")
        handle.write("| subset | query frames | anchors | groups | skipped anchor groups | description |\n")
        handle.write("|---|---:|---:|---:|---:|---|\n")
        for name, item in metadata["subsets"].items():
            handle.write(
                f"| {name} | {item['query_frames']} | {item['anchors']} | "
                f"{item['kept_groups']} | {item['skipped_anchor_groups']} | "
                f"{item['description']} |\n"
            )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
