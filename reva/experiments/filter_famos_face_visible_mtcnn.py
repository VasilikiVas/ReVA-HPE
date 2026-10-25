#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_FAMOS_ROOT = Path("data/famos_head_pose_test")
DEFAULT_OUT_ROOT = Path("outputs/eval_famos_head_pose/subsets")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--famos-root", type=Path, default=DEFAULT_FAMOS_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prob-threshold", type=float, default=0.90)
    parser.add_argument("--views", type=int, nargs="+", default=None,
                        help="Keep only these camera views (paper protocol: 0).")
    parser.add_argument("--exclude-subjects", nargs="+", default=(),
                        help="Drop these subjects (paper protocol: FaMoS_subject_091).")
    parser.add_argument("--relative-rot-caps", type=float, nargs="+", default=(45.0, 60.0))
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


def load_image_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def group_indices(labels, views=None, exclude_subjects=()):
    groups = defaultdict(list)
    for idx in range(len(labels["subject"])):
        if views is not None and int(labels["view_idx"][idx]) not in views:
            continue
        if str(labels["subject"][idx]) in exclude_subjects:
            continue
        key = (
            str(labels["subject"][idx]),
            str(labels["sequence"][idx]),
            int(labels["view_idx"][idx]),
        )
        groups[key].append(idx)
    for key in list(groups):
        groups[key] = sorted(groups[key], key=lambda i: int(str(labels["frame"][i])))
    return dict(sorted(groups.items()))


def geodesic_deg(rotation_a, rotation_b):
    rel = rotation_a @ rotation_b.T
    cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def detect_visibility(args, labels):
    from facenet_pytorch import MTCNN

    detector = MTCNN(keep_all=True, device=args.device)
    rows = []
    visibility = {}
    for idx in range(len(labels["subject"])):
        image_path = args.famos_root / str(labels["image_relpath"][idx])
        image = load_image_rgb(image_path)
        boxes, probs = detector.detect(Image.fromarray(image))
        best_prob = float("nan")
        best_box = ["", "", "", ""]
        visible = 0
        if boxes is not None and probs is not None and len(boxes) > 0:
            valid = [(box, float(prob)) for box, prob in zip(boxes, probs) if prob is not None]
            if valid:
                box, best_prob = max(valid, key=lambda item: item[1])
                best_box = [f"{float(value):.6f}" for value in box.tolist()]
                visible = int(best_prob >= args.prob_threshold)
        visibility[idx] = bool(visible)
        rows.append(
            {
                "label_index": idx,
                "subject": str(labels["subject"][idx]),
                "sequence": str(labels["sequence"][idx]),
                "view_idx": int(labels["view_idx"][idx]),
                "frame": str(labels["frame"][idx]),
                "image_relpath": str(labels["image_relpath"][idx]),
                "face_visible": visible,
                "mtcnn_prob": "" if math.isnan(best_prob) else f"{best_prob:.9f}",
                "box_x1": best_box[0],
                "box_y1": best_box[1],
                "box_x2": best_box[2],
                "box_y2": best_box[3],
            }
        )
    return visibility, rows


def make_subset_rows(labels, csv_rows, visibility, relative_cap=None):
    rows = []
    kept_groups = 0
    skipped_anchor_groups = 0
    for key, indices in group_indices(labels, args.views, set(args.exclude_subjects)).items():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        if not visibility.get(anchor, False):
            skipped_anchor_groups += 1
            continue
        anchor_rotation = np.asarray(labels["R_cam_head"][anchor], dtype=np.float64)
        selected = []
        for idx in indices[1:]:
            if not visibility.get(idx, False):
                continue
            rel_deg = geodesic_deg(np.asarray(labels["R_cam_head"][idx], dtype=np.float64), anchor_rotation)
            if relative_cap is not None and rel_deg > relative_cap:
                continue
            selected.append((idx, rel_deg))
        if not selected:
            continue
        kept_groups += 1
        rows.append(make_subset_row(labels, csv_rows, anchor, "anchor", 0.0))
        for idx, rel_deg in selected:
            rows.append(make_subset_row(labels, csv_rows, idx, "query", rel_deg))
    return rows, kept_groups, skipped_anchor_groups


def make_subset_row(labels, csv_rows, idx, role, relative_rotation_deg):
    source = csv_rows[idx]
    return {
        "label_index": idx,
        "role": role,
        "subject": str(labels["subject"][idx]),
        "sequence": str(labels["sequence"][idx]),
        "view_idx": int(labels["view_idx"][idx]),
        "frame": str(labels["frame"][idx]),
        "image_relpath": str(labels["image_relpath"][idx]),
        "relative_rotation_deg": f"{float(relative_rotation_deg):.9f}",
        "procrustes_rms_m": source.get("procrustes_rms_m", ""),
    }


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

    visibility, visibility_rows = detect_visibility(args, labels)
    write_csv(
        args.out_root / "famos_mtcnn_visibility.csv",
        visibility_rows,
        (
            "label_index",
            "subject",
            "sequence",
            "view_idx",
            "frame",
            "image_relpath",
            "face_visible",
            "mtcnn_prob",
            "box_x1",
            "box_y1",
            "box_x2",
            "box_y2",
        ),
    )

    fieldnames = (
        "label_index",
        "role",
        "subject",
        "sequence",
        "view_idx",
        "frame",
        "image_relpath",
        "relative_rotation_deg",
        "procrustes_rms_m",
    )
    metadata = {
        "famos_root": str(args.famos_root),
        "selection_uses_model_predictions": False,
        "selection_uses_model_errors": False,
        "selection_uses_mtcnn_face_visibility": True,
        "mtcnn_prob_threshold": args.prob_threshold,
        "rule": "A group is kept only if its first-frame anchor has an MTCNN face. Query frames require an MTCNN face and optional relative-rotation cap.",
        "subsets": {},
    }

    specs = [("mtcnn_visible", None)]
    specs.extend((f"mtcnn_visible_rel{int(cap)}", float(cap)) for cap in args.relative_rot_caps)
    for name, cap in specs:
        rows, kept_groups, skipped_anchor_groups = make_subset_rows(
            labels, csv_rows, visibility, cap
        )
        path = args.out_root / f"famos_{name}_indices.csv"
        write_csv(path, rows, fieldnames)
        metadata["subsets"][name] = {
            "path": str(path),
            "description": (
                "MTCNN face-visible subset"
                if cap is None
                else f"MTCNN face-visible subset with relative rotation <= {cap:g} deg"
            ),
            "relative_rotation_cap_deg": cap,
            **summarize(rows, kept_groups, skipped_anchor_groups),
        }

    (args.out_root / "famos_mtcnn_subset_metadata.json").write_text(json.dumps(metadata, indent=2))
    with (args.out_root / "famos_mtcnn_subset_summary.md").open("w") as handle:
        handle.write("# FaMoS MTCNN Face-Visible Subsets\n\n")
        handle.write("Selection uses MTCNN visibility and optional GT relative-rotation caps, model predictions/errors are not used.\n\n")
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
