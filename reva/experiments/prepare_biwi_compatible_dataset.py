#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


FAMOS_ROOT = Path("data/famos_head_pose_test")
HEAD_YZ_FLIP = np.diag([1.0, -1.0, -1.0])


DUMMY_OBJ = """v -0.050000 -0.050000 0.000000
v 0.050000 -0.050000 0.000000
v 0.000000 0.050000 0.000000
f 1 2 3
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("famos",), default="famos")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--famos-root", type=Path, default=FAMOS_ROOT)
    parser.add_argument("--famos-subset-csv", type=Path, default=None)
    parser.add_argument(
        "--famos-rotation-convention",
        choices=("raw", "head_yz_flip"),
        default="raw",
        help="Convention for FaMoS pose files. head_yz_flip writes R_cam_head @ diag(1,-1,-1).",
    )
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--max-targets-per-group", type=int, default=0)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking.")
    return parser.parse_args()


def read_csv(path):
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


def write_pose(path, R, t):
    lines = []
    for row in np.asarray(R, dtype=np.float64).reshape(3, 3):
        lines.append(" ".join(f"{float(value):.12g}" for value in row))
    lines.append(" ".join(f"{float(value):.12g}" for value in np.asarray(t, dtype=np.float64).reshape(3)))
    path.write_text("\n".join(lines) + "\n")


def write_rgb_cal(path, K, R_cam=None, t_cam=None):
    if R_cam is None:
        R_cam = np.eye(3, dtype=np.float64)
    if t_cam is None:
        t_cam = np.zeros(3, dtype=np.float64)
    lines = []
    for row in np.asarray(K, dtype=np.float64).reshape(3, 3):
        lines.append(" ".join(f"{float(value):.12g}" for value in row))
    lines.append("0 0 0")
    for row in np.asarray(R_cam, dtype=np.float64).reshape(3, 3):
        lines.append(" ".join(f"{float(value):.12g}" for value in row))
    lines.append(" ".join(f"{float(value):.12g}" for value in np.asarray(t_cam, dtype=np.float64).reshape(3)))
    path.write_text("\n".join(lines) + "\n")


def link_or_copy(src, dst, copy_images=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def write_dummy_obj(path):
    if not path.exists():
        path.write_text(DUMMY_OBJ)


def parse_subset_indices(path):
    if path is None:
        return None
    rows = read_csv(path)
    if not rows:
        return set()
    field = "label_index" if "label_index" in rows[0] else "index"
    return {int(row[field]) for row in rows}


def apply_famos_convention(R, t, convention):
    if convention == "raw":
        return R, t
    if convention == "head_yz_flip":
        return R @ HEAD_YZ_FLIP, t
    raise ValueError(convention)


def load_famos_entries(args):
    label_path = args.famos_root / "labels.npz"
    allowed = parse_subset_indices(args.famos_subset_csv)
    data = np.load(label_path, allow_pickle=True)
    groups = defaultdict(list)
    for idx in range(len(data["subject"])):
        if allowed is not None and idx not in allowed:
            continue
        key = (str(data["subject"][idx]), str(data["sequence"][idx]), int(data["view_idx"][idx]))
        R, t = apply_famos_convention(
            np.asarray(data["R_cam_head"][idx], dtype=np.float64),
            np.asarray(data["t_cam_head_mm"][idx], dtype=np.float64),
            args.famos_rotation_convention,
        )
        groups[key].append(
            {
                "source_index": idx,
                "source_dataset": "famos",
                "source_key": "|".join(map(str, key)),
                "original_frame": str(data["frame"][idx]),
                "image_path": args.famos_root / str(data["image_relpath"][idx]),
                "R": R,
                "t": t,
                "K": np.asarray(data["K"][idx], dtype=np.float64),
            }
        )
    return groups


def quat_wxyz_to_matrix(q):
    w, x, y, z = [float(value) for value in q]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

def materialize_groups(groups, out_root, copy_images=False, max_groups=0, max_targets_per_group=0):
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    selected_groups = list(sorted(groups.items(), key=lambda item: item[0]))
    if max_groups > 0:
        selected_groups = selected_groups[:max_groups]

    for subject_id, (group_key, entries) in enumerate(selected_groups, start=1):
        entries = sorted(entries, key=lambda item: str(item["original_frame"]))
        if len(entries) < 2:
            continue
        if max_targets_per_group > 0:
            entries = [entries[0]] + entries[1 : 1 + max_targets_per_group]
            if len(entries) < 2:
                continue

        subject_name = f"{subject_id:03d}"
        subject_dir = out_root / subject_name
        subject_dir.mkdir(parents=True, exist_ok=True)
        write_dummy_obj(out_root / f"{subject_name}.obj")
        write_rgb_cal(subject_dir / "rgb.cal", entries[0]["K"])

        for local_idx, entry in enumerate(entries):
            frame_name = f"frame_{local_idx:05d}"
            dst_image = subject_dir / f"{frame_name}_rgb.png"
            link_or_copy(entry["image_path"], dst_image, copy_images=copy_images)
            write_pose(subject_dir / f"{frame_name}_pose.txt", entry["R"], entry["t"])
            manifest_rows.append(
                {
                    "subject": subject_name,
                    "local_frame": local_idx,
                    "role": "anchor" if local_idx == 0 else "query",
                    "source_dataset": entry["source_dataset"],
                    "source_key": entry["source_key"],
                    "source_index": entry["source_index"],
                    "original_frame": entry["original_frame"],
                    "image_path": str(entry["image_path"]),
                }
            )
    return manifest_rows


def main():
    args = parse_args()
    groups = load_famos_entries(args)

    manifest_rows = materialize_groups(
        groups,
        args.out_root,
        copy_images=args.copy_images,
        max_groups=args.max_groups,
        max_targets_per_group=args.max_targets_per_group,
    )
    if not manifest_rows:
        raise RuntimeError("No BIWI-compatible groups were written")

    write_csv(args.out_root / "adapter_manifest.csv", manifest_rows)
    subjects = sorted({row["subject"] for row in manifest_rows})
    metadata = {
        "dataset": args.dataset,
        "out_root": str(args.out_root),
        "subjects": len(subjects),
        "frames": len(manifest_rows),
        "queries": sum(1 for row in manifest_rows if row["role"] == "query"),
        "copy_images": bool(args.copy_images),
        "famos_subset_csv": str(args.famos_subset_csv) if args.famos_subset_csv else None,
        "famos_rotation_convention": args.famos_rotation_convention,
        "note": "Images are symlinked unless copy_images=true. Mesh files are dummy placeholders for non-visual eval.",
    }
    (args.out_root / "adapter_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
