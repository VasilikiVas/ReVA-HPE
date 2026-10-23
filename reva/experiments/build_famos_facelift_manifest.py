#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import shlex
from pathlib import Path


DEFAULT_READY_CSV = Path(
    "data/famos_ready/famos_mtcnn_visible/ready_pose_predictions.csv"
)
DEFAULT_OUTPUT = Path(__file__).with_name("famos_facelift_enrollment_manifest.csv")


REQUIRED_COLUMNS = (
    "subject",
    "enrollment_id",
    "source_key",
    "enrollment_image_path",
    "canonical_flame_json",
    "enrollment_rgb_cal_path",
)


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    stem = stem.strip("._-")
    return stem or "enrollment"


def _subject_sort_key(subject: str) -> tuple[int, str]:
    try:
        return int(subject), subject
    except ValueError:
        return 10**9, subject


def _is_ready(row: dict[str, str]) -> bool:
    value = row.get("ready_for_refinement", "1").strip().lower()
    return value not in {"0", "false", "no", "n"}


def build_manifest(ready_csv: Path, output: Path) -> list[dict[str, str]]:
    with ready_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns in {ready_csv}: {missing}")

        by_subject: dict[str, dict[str, str]] = {}
        conflicts: list[str] = []
        for row in reader:
            if not _is_ready(row):
                continue

            subject = row["subject"].strip()
            current = {col: row[col].strip() for col in REQUIRED_COLUMNS}
            if subject not in by_subject:
                by_subject[subject] = current
                continue

            previous = by_subject[subject]
            for col in REQUIRED_COLUMNS[1:]:
                if current[col] != previous[col]:
                    conflicts.append(f"subject={subject} column={col}")
                    break

        if conflicts:
            preview = "\n".join(conflicts[:10])
            raise SystemExit(
                "Found subjects with multiple enrollment definitions. "
                f"First conflicts:\n{preview}"
            )

    rows: list[dict[str, str]] = []
    for array_id, subject in enumerate(sorted(by_subject, key=_subject_sort_key), start=1):
        row = dict(by_subject[subject])
        row["array_id"] = str(array_id)
        row["subject_root_name"] = f"subject{subject}"
        row["enrollment_stem"] = _safe_stem(f"subject{subject}_{row['enrollment_id']}")
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "array_id",
        "subject",
        "subject_root_name",
        "enrollment_id",
        "enrollment_stem",
        "source_key",
        "enrollment_image_path",
        "canonical_flame_json",
        "enrollment_rgb_cal_path",
    )
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def load_manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as f:
        return list(csv.DictReader(f))


def emit_shell(manifest: Path, array_id: int) -> None:
    rows = load_manifest(manifest)
    matches = [row for row in rows if int(row["array_id"]) == array_id]
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one row for array_id={array_id}, got {len(matches)}")

    row = matches[0]
    mapping = {
        "FAMOS_ARRAY_ID": row["array_id"],
        "FAMOS_SUBJECT": row["subject"],
        "FAMOS_SUBJECT_ROOT_NAME": row["subject_root_name"],
        "FAMOS_ENROLLMENT_ID": row["enrollment_id"],
        "FAMOS_ENROLLMENT_STEM": row["enrollment_stem"],
        "FAMOS_SOURCE_KEY": row["source_key"],
        "FAMOS_ENROLLMENT_IMAGE": row["enrollment_image_path"],
        "FAMOS_CANONICAL_FLAME_JSON": row["canonical_flame_json"],
        "FAMOS_RGB_CAL": row["enrollment_rgb_cal_path"],
    }
    for key, value in mapping.items():
        print(f"{key}={shlex.quote(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-csv", type=Path, default=DEFAULT_READY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--array-id", type=int)
    parser.add_argument("--emit-shell", action="store_true")
    args = parser.parse_args()

    if args.emit_shell:
        if args.array_id is None:
            raise SystemExit("--array-id is required with --emit-shell")
        emit_shell(args.manifest, args.array_id)
        return

    rows = build_manifest(args.ready_csv, args.output)
    print(f"Wrote {len(rows)} FaMoS FaceLift enrollment rows to {args.output}")


if __name__ == "__main__":
    main()
