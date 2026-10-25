#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


OUT_ROOT = Path("outputs")
DEFAULT_ADAPTER_ROOT = OUT_ROOT / "biwi_compatible_eval_roots"
DEFAULT_PRED_ROOT = OUT_ROOT / "biwi_compatible_eval_outputs"
DEFAULT_OUT_ROOT = OUT_ROOT / "refinement_inputs_cross_dataset"
DEFAULT_FAMOS_ENROLL = OUT_ROOT / "eval_famos_head_pose/enrollment/group_anchors_manifest.csv"


PROTOCOLS = (
    "famos_full",
    "famos_mtcnn_visible",
    "famos_view00_mtcnn_visible_no091",
    "famos_mtcnn_visible_rel45",
    "famos_mtcnn_visible_rel60",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--famos-enrollment-manifest", type=Path, default=DEFAULT_FAMOS_ENROLL)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def dataset_from_protocol(protocol):
    if protocol.startswith("famos_"):
        return "famos"
    raise ValueError(f"Cannot infer dataset from protocol: {protocol}")


def famos_key(row):
    return f"{row['subject']}|{row['sequence']}|{int(row['view_idx'])}"


def load_enrollment_index(dataset, args):
    if dataset == "famos":
        rows = read_csv(args.famos_enrollment_manifest)
        return {famos_key(row): row for row in rows if row.get("protocol") == "group_anchors"}
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_adapter_subjects(adapter_rows):
    by_subject = defaultdict(list)
    for row in adapter_rows:
        by_subject[row["subject"]].append(row)

    subject_rows = []
    frame_lookup = {}
    for subject, rows in sorted(by_subject.items()):
        rows = sorted(rows, key=lambda row: int(row["local_frame"]))
        anchors = [row for row in rows if row["role"] == "anchor"]
        anchor = anchors[0] if anchors else rows[0]
        query_count = sum(1 for row in rows if row["role"] == "query")
        subject_rows.append(
            {
                "adapter_subject": subject,
                "source_dataset": anchor.get("source_dataset", ""),
                "source_key": anchor.get("source_key", ""),
                "adapter_anchor_frame": anchor.get("local_frame", ""),
                "adapter_anchor_original_frame": anchor.get("original_frame", ""),
                "adapter_anchor_image_path": anchor.get("image_path", ""),
                "adapter_query_count": query_count,
            }
        )
        for row in rows:
            frame_lookup[(subject, str(int(row["local_frame"])))] = row
    return subject_rows, frame_lookup


def canonical_ok(path_text):
    if not path_text:
        return False, "missing_canonical_path"
    path = Path(path_text)
    if not path.exists():
        return False, "missing_canonical_flame_json"
    if path.stat().st_size <= 0:
        return False, "empty_canonical_flame_json"
    return True, ""


def make_subject_manifest(protocol, dataset, adapter_subjects, enrollment_index):
    subject_manifest = []
    enrollment_key_to_subject = {}
    for subject_row in adapter_subjects:
        source_key = subject_row["source_key"]
        enroll = enrollment_index.get(source_key)
        if enroll is None:
            ready = False
            reason = "missing_enrollment_manifest_row"
            canonical_path = ""
        else:
            canonical_path = enroll.get("canonical_flame_json", "")
            ready, reason = canonical_ok(canonical_path)
            enrollment_key_to_subject[source_key] = subject_row["adapter_subject"]

        row = {
            "protocol": protocol,
            "dataset": dataset,
            **subject_row,
            "enrollment_found": int(enroll is not None),
            "enrollment_id": enroll.get("enrollment_id", "") if enroll else "",
            "enrollment_image_path": enroll.get("image_path", "") if enroll else "",
            "enrollment_rgb_cal_path": enroll.get("rgb_cal_path", "") if enroll else "",
            "canonical_flame_json": canonical_path,
            "canonical_flame_exists": int(ready),
            "ready_for_refinement": int(ready and enroll is not None),
            "not_ready_reason": reason,
        }
        if dataset == "famos" and enroll:
            row.update(
                {
                    "famos_subject": enroll.get("subject", ""),
                    "famos_sequence": enroll.get("sequence", ""),
                    "famos_view_idx": enroll.get("view_idx", ""),
                    "famos_anchor_frame": enroll.get("frame", ""),
                    "famos_target_count": enroll.get("target_count", ""),
                }
            )
        subject_manifest.append(row)
    return subject_manifest, enrollment_key_to_subject


def make_frame_manifest(protocol, dataset, prediction_rows, subject_manifest, frame_lookup):
    subject_by_id = {row["adapter_subject"]: row for row in subject_manifest}
    frame_rows = []
    ready_prediction_rows = []

    for pred in prediction_rows:
        subject = pred["subject"]
        frame = str(int(pred["frame"]))
        subject_row = subject_by_id.get(subject, {})
        adapter_frame = frame_lookup.get((subject, frame), {})
        ready = bool(int(subject_row.get("ready_for_refinement", 0))) and bool(adapter_frame)
        reason = subject_row.get("not_ready_reason", "")
        if not adapter_frame:
            reason = "missing_adapter_frame_row"

        metadata = {
            "protocol": protocol,
            "dataset": dataset,
            "ready_for_refinement": int(ready),
            "not_ready_reason": reason,
            "source_key": subject_row.get("source_key", ""),
            "source_dataset": subject_row.get("source_dataset", ""),
            "query_image_path": adapter_frame.get("image_path", ""),
            "query_original_frame": adapter_frame.get("original_frame", ""),
            "enrollment_id": subject_row.get("enrollment_id", ""),
            "enrollment_image_path": subject_row.get("enrollment_image_path", ""),
            "enrollment_rgb_cal_path": subject_row.get("enrollment_rgb_cal_path", ""),
            "canonical_flame_json": subject_row.get("canonical_flame_json", ""),
        }
        merged = {**metadata, **pred}
        frame_rows.append(merged)
        if ready:
            ready_prediction_rows.append(merged)
    return frame_rows, ready_prediction_rows


def write_summary(path, summary, subject_manifest):
    lines = [
        f"# Refinement Inputs: {summary['protocol']}",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Adapter root: `{summary['adapter_root']}`",
        f"- Prediction CSV: `{summary['prediction_csv']}`",
        f"- Subjects: {summary['subjects_total']}",
        f"- Subjects ready: {summary['subjects_ready']}",
        f"- Prediction rows: {summary['prediction_rows_total']}",
        f"- Prediction rows ready: {summary['prediction_rows_ready']}",
        f"- Output root: `{summary['out_dir']}`",
        "",
        "## Not-Ready Reasons",
        "",
    ]
    if summary["not_ready_reasons"]:
        for reason, count in sorted(summary["not_ready_reasons"].items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")

    missing = [row for row in subject_manifest if not int(row["ready_for_refinement"])]
    if missing:
        lines.extend(["", "## First Missing Subjects", ""])
        for row in missing[:20]:
            lines.append(
                f"- adapter subject `{row['adapter_subject']}` / source `{row['source_key']}`: "
                f"{row['not_ready_reason']}"
            )
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    dataset = dataset_from_protocol(args.protocol)
    adapter_dir = args.adapter_root / args.protocol
    prediction_csv = args.prediction_root / args.protocol / "pose_predictions.csv"
    out_dir = args.out_root / args.protocol

    adapter_manifest_csv = adapter_dir / "adapter_manifest.csv"
    if not adapter_manifest_csv.exists():
        raise FileNotFoundError(adapter_manifest_csv)
    if not prediction_csv.exists():
        raise FileNotFoundError(prediction_csv)

    adapter_rows = read_csv(adapter_manifest_csv)
    prediction_rows = read_csv(prediction_csv)
    enrollment_index = load_enrollment_index(dataset, args)
    adapter_subjects, frame_lookup = load_adapter_subjects(adapter_rows)
    subject_manifest, _ = make_subject_manifest(
        args.protocol,
        dataset,
        adapter_subjects,
        enrollment_index,
    )
    frame_manifest, ready_prediction_rows = make_frame_manifest(
        args.protocol,
        dataset,
        prediction_rows,
        subject_manifest,
        frame_lookup,
    )

    not_ready_reasons = defaultdict(int)
    for row in subject_manifest:
        if not int(row["ready_for_refinement"]):
            not_ready_reasons[row["not_ready_reason"]] += 1

    subject_fields = [
        "protocol",
        "dataset",
        "adapter_subject",
        "source_dataset",
        "source_key",
        "adapter_anchor_frame",
        "adapter_anchor_original_frame",
        "adapter_anchor_image_path",
        "adapter_query_count",
        "enrollment_found",
        "enrollment_id",
        "enrollment_image_path",
        "enrollment_rgb_cal_path",
        "canonical_flame_json",
        "canonical_flame_exists",
        "ready_for_refinement",
        "not_ready_reason",
        "famos_subject",
        "famos_sequence",
        "famos_view_idx",
        "famos_anchor_frame",
        "famos_target_count",
    ]
    prediction_fields = list(prediction_rows[0].keys()) if prediction_rows else []
    metadata_fields = [
        "protocol",
        "dataset",
        "ready_for_refinement",
        "not_ready_reason",
        "source_key",
        "source_dataset",
        "query_image_path",
        "query_original_frame",
        "enrollment_id",
        "enrollment_image_path",
        "enrollment_rgb_cal_path",
        "canonical_flame_json",
    ]
    frame_fields = metadata_fields + [field for field in prediction_fields if field not in metadata_fields]

    write_csv(out_dir / "refinement_subject_manifest.csv", subject_manifest, subject_fields)
    write_csv(out_dir / "refinement_frame_manifest.csv", frame_manifest, frame_fields)
    write_csv(out_dir / "ready_pose_predictions.csv", ready_prediction_rows, frame_fields)

    summary = {
        "protocol": args.protocol,
        "dataset": dataset,
        "adapter_root": str(adapter_dir),
        "prediction_csv": str(prediction_csv),
        "out_dir": str(out_dir),
        "subjects_total": len(subject_manifest),
        "subjects_ready": sum(int(row["ready_for_refinement"]) for row in subject_manifest),
        "prediction_rows_total": len(prediction_rows),
        "prediction_rows_ready": len(ready_prediction_rows),
        "not_ready_reasons": dict(not_ready_reasons),
        "subject_manifest_csv": str(out_dir / "refinement_subject_manifest.csv"),
        "frame_manifest_csv": str(out_dir / "refinement_frame_manifest.csv"),
        "ready_pose_predictions_csv": str(out_dir / "ready_pose_predictions.csv"),
    }
    write_json(out_dir / "refinement_input_summary.json", summary)
    write_summary(out_dir / "refinement_input_summary.md", summary, subject_manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
