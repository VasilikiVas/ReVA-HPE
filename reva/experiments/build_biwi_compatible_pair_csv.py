#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def frame_number(path: Path) -> int:
    name = path.name
    # frame_00012_pose.txt -> 12
    return int(name.split("_")[1])


def collect_subject_pairs(subject_dir: Path) -> list[dict[str, object]]:
    pose_files = sorted(subject_dir.glob("frame_*_pose.txt"), key=frame_number)
    if len(pose_files) < 2:
        return []

    anchor_pose = pose_files[0]
    anchor_frame = frame_number(anchor_pose)
    rows = []
    for query_pose in pose_files[1:]:
        rows.append(
            {
                "subject": subject_dir.name,
                "anchor_frame_num": anchor_frame,
                "anchor_pose_file": anchor_pose.name,
                "query_frame_num": frame_number(query_pose),
                "query_pose_file": query_pose.name,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    subjects = sorted(
        [p for p in args.dataset_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    rows = []
    for subject_dir in subjects:
        rows.extend(collect_subject_pairs(subject_dir))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "anchor_frame_num",
                "anchor_pose_file",
                "query_frame_num",
                "query_pose_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(args.out_csv)
    print(f"subjects={len(subjects)} pairs={len(rows)}")


if __name__ == "__main__":
    main()
