#!/usr/bin/env python3
"""Run the FLAME landmark enrollment fit for one FaMoS manifest entry."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one FaMoS FLAME enrollment entry")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--entry-index",
        type=int,
        required=True,
        help="1-based manifest row index, matching SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument(
        "--blended-root",
        type=Path,
        default=Path(os.environ.get("BLENDED_IN_FLAMES_ROOT", "blended_in_flames")),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_manifest_row(path, entry_index):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if entry_index < 1 or entry_index > len(rows):
        raise IndexError(f"entry-index {entry_index} outside manifest length {len(rows)}")
    return rows[entry_index - 1], len(rows)


def main():
    args = parse_args()
    row, total = load_manifest_row(args.manifest, args.entry_index)
    image = Path(row["image_path"])
    output_dir = Path(row["flame_dir"])
    final_json = output_dir / "canonical_flame_landmarks.json"
    if final_json.is_file() and final_json.stat().st_size > 0 and not args.force:
        print(json.dumps({"status": "already_complete", "entry_index": args.entry_index, "total": total, "final": str(final_json)}))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if not image.is_file():
        raise FileNotFoundError(image)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "reva" / "avatar_pipeline" / "fit_enrollment_flame_landmarks.py"),
        "--image",
        str(image),
        "--blended-root",
        str(args.blended_root),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
    ]
    print(json.dumps({
        "status": "running",
        "entry_index": args.entry_index,
        "total": total,
        "enrollment_id": row["enrollment_id"],
        "image": str(image),
        "output_dir": str(output_dir),
    }, indent=2))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    subprocess.run(command, check=True, env=env)
    if not final_json.is_file() or final_json.stat().st_size == 0:
        raise RuntimeError(f"Expected output was not created: {final_json}")
    print(json.dumps({"status": "complete", "entry_index": args.entry_index, "final": str(final_json)}))


if __name__ == "__main__":
    main()
