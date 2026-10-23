#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(ROOT / "evaluation"))

from eval_biwi_facelift_metric_anchor_bank import biwi_errors, ransac_fuse  # noqa: E402


DEFAULT_INPUT = Path("outputs/synthetic_heldout_fusion")
DEFAULT_OUTPUT = Path("outputs/synthetic_threshold_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rot-thresholds", default="5,7.5,10,15,20")
    parser.add_argument("--trans-thresholds", default="25,50,75,100")
    parser.add_argument("--min-inliers", default="2,3,4")
    parser.add_argument("--selected-rot-threshold", type=float, default=10.0)
    parser.add_argument("--selected-trans-threshold", type=float, default=50.0)
    parser.add_argument("--selected-min-inliers", type=int, default=3)
    parser.add_argument("--corruption-fraction", type=float, default=0.0)
    parser.add_argument("--corruption-rot-deg", type=float, default=35.0)
    parser.add_argument("--corruption-trans-mm", type=float, default=120.0)
    parser.add_argument("--corruption-seed", type=int, default=20270629)
    return parser.parse_args()


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_r{rr}{cc}"]) for cc in range(3)] for rr in range(3)],
        dtype=np.float64,
    )


def translation_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_t{axis}_mm"]) for axis in ("x", "y", "z")], dtype=np.float64)


def axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def corrupt_candidates(
    rotations: np.ndarray,
    translations: np.ndarray,
    fraction: float,
    rot_deg: float,
    trans_mm: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fraction <= 0.0:
        return rotations, translations, np.asarray([], dtype=np.int64)
    count = max(1, int(round(fraction * len(rotations))))
    ids = np.sort(rng.choice(len(rotations), size=count, replace=False))
    corrupted_rotations = rotations.copy()
    corrupted_translations = translations.copy()
    for index in ids:
        axis = rng.normal(size=3)
        angle = np.deg2rad(rot_deg) * rng.choice([-1.0, 1.0])
        corrupted_rotations[index] = axis_angle_rotation(axis, angle) @ corrupted_rotations[index]
        corrupted_translations[index] = corrupted_translations[index] + rng.normal(scale=trans_mm, size=3)
    return corrupted_rotations, corrupted_translations, ids


def load_h2c(view_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (view_dir / "output_opencv_camera.pkl").open("rb") as handle:
        camera = pickle.load(handle)
    with (view_dir / "output_object_transform_post_render.pkl").open("rb") as handle:
        head = pickle.load(handle)
    rotation_w2c = np.asarray(camera["R"], dtype=np.float64)
    translation_w2c = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    rotation_h2w = np.asarray(head["R_world"], dtype=np.float64)
    translation_h2w = np.asarray(head["t_world"], dtype=np.float64).reshape(3)
    rotation = rotation_w2c @ rotation_h2w
    translation_m = rotation_w2c @ translation_h2w + translation_w2c
    return rotation, translation_m


def load_candidate_groups(input_dir: Path):
    combined = read_csv(input_dir / "combined_predictions.csv")
    query_by_key = {(row["subject"], row["frame_num"]): row for row in combined}
    pair_rows = []
    for path in sorted(input_dir.glob("subject_*_pairwise.csv")):
        pair_rows.extend(read_csv(path))

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pair_rows:
        groups[(row["subject"], row["frame_num"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["anchor_id"]))
    return query_by_key, groups


def summarize(values: list[float], reducer: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if reducer == "mean":
        return float(arr.mean())
    if reducer == "median":
        return float(np.median(arr))
    if reducer == "p95":
        return float(np.percentile(arr, 95))
    raise ValueError(reducer)


def build_heatmap_rows(rows: list[dict[str, object]], metric: str, min_inliers: int) -> list[dict[str, object]]:
    selected = [row for row in rows if int(row["min_inliers"]) == min_inliers]
    rot_values = sorted({float(row["tau_R_deg"]) for row in selected})
    trans_values = sorted({float(row["tau_t_mm"]) for row in selected})
    by_key = {(float(row["tau_R_deg"]), float(row["tau_t_mm"])): row for row in selected}
    out = []
    for rot in rot_values:
        line: dict[str, object] = {"tau_R_deg": rot}
        for trans in trans_values:
            value = by_key[(rot, trans)][metric]
            line[f"tau_t_{trans:g}_mm"] = value
        out.append(line)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rot_thresholds = parse_list(args.rot_thresholds, float)
    trans_thresholds = parse_list(args.trans_thresholds, float)
    min_inlier_values = parse_list(args.min_inliers, int)

    query_by_key, groups = load_candidate_groups(args.input_dir)
    if not groups:
        raise RuntimeError(f"No pairwise candidate rows found under {args.input_dir}")
    if not 0.0 <= args.corruption_fraction < 1.0:
        raise ValueError("--corruption-fraction must be in [0, 1)")

    summary_rows = []
    prediction_rows = []
    query_gt_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    candidate_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    rng = np.random.default_rng(args.corruption_seed)
    for item, candidate_rows in sorted(groups.items()):
        clean_rotations = np.stack([matrix_from_row(row, "pred") for row in candidate_rows])
        clean_translations = np.stack(
            [
                np.asarray(
                    [
                        float(row["pred_tx_mm"]),
                        float(row["pred_ty_mm"]),
                        float(row["pred_tz_mm"]),
                    ],
                    dtype=np.float64,
                )
                for row in candidate_rows
            ]
        )
        candidate_cache[item] = corrupt_candidates(
            clean_rotations,
            clean_translations,
            args.corruption_fraction,
            args.corruption_rot_deg,
            args.corruption_trans_mm,
            rng,
        )

    for tau_r in rot_thresholds:
        for tau_t in trans_thresholds:
            for min_inliers in min_inlier_values:
                per_query = []
                for item, candidate_rows in sorted(groups.items()):
                    query_row = query_by_key[item]
                    if item not in query_gt_cache:
                        gt_rotation, gt_translation_m = load_h2c(Path(query_row["query_path"]))
                        query_gt_cache[item] = (gt_rotation, gt_translation_m * 1000.0)
                    gt_rotation, gt_translation = query_gt_cache[item]

                    rotations, translations, corrupted_ids = candidate_cache[item]
                    fused_rotation, fused_translation, inlier_ids, success, consensus_size = ransac_fuse(
                        rotations,
                        translations,
                        tau_r,
                        tau_t,
                        min_inliers,
                    )
                    angle, trans_abs, trans_l2 = biwi_errors(
                        fused_rotation,
                        fused_translation,
                        gt_rotation,
                        gt_translation,
                    )
                    row = {
                        "tau_R_deg": tau_r,
                        "tau_t_mm": tau_t,
                        "min_inliers": min_inliers,
                        "subject": item[0],
                        "frame_num": item[1],
                        "identity": query_row["identity"],
                        "query_path": query_row["query_path"],
                        "anchors": len(candidate_rows),
                        "ransac_inliers": len(inlier_ids),
                        "ransac_consensus_size": consensus_size,
                        "ransac_success": int(success),
                        "inlier_ids": " ".join(str(int(index)) for index in inlier_ids),
                        "corruption_fraction": args.corruption_fraction,
                        "corrupted_anchor_ids": " ".join(str(int(index)) for index in corrupted_ids),
                        "corrupted_anchor_count": len(corrupted_ids),
                        "yaw_error_deg": float(angle[0]),
                        "pitch_error_deg": float(angle[1]),
                        "roll_error_deg": float(angle[2]),
                        "rot_mean_error_deg": float(angle.mean()),
                        "tx_error_mm": float(trans_abs[0]),
                        "ty_error_mm": float(trans_abs[1]),
                        "tz_error_mm": float(trans_abs[2]),
                        "translation_l2_error_mm": float(trans_l2),
                    }
                    per_query.append(row)
                    prediction_rows.append(row)

                summary_rows.append(
                    {
                        "tau_R_deg": tau_r,
                        "tau_t_mm": tau_t,
                        "min_inliers": min_inliers,
                        "queries": len(per_query),
                        "success_rate": float(np.mean([row["ransac_success"] for row in per_query])),
                        "failure_rate": float(1.0 - np.mean([row["ransac_success"] for row in per_query])),
                        "mean_yaw_error_deg": summarize([row["yaw_error_deg"] for row in per_query], "mean"),
                        "mean_pitch_error_deg": summarize([row["pitch_error_deg"] for row in per_query], "mean"),
                        "mean_roll_error_deg": summarize([row["roll_error_deg"] for row in per_query], "mean"),
                        "mean_rot_mae_deg": summarize([row["rot_mean_error_deg"] for row in per_query], "mean"),
                        "median_rot_mae_deg": summarize([row["rot_mean_error_deg"] for row in per_query], "median"),
                        "p95_rot_mae_deg": summarize([row["rot_mean_error_deg"] for row in per_query], "p95"),
                        "mean_tx_error_mm": summarize([row["tx_error_mm"] for row in per_query], "mean"),
                        "mean_ty_error_mm": summarize([row["ty_error_mm"] for row in per_query], "mean"),
                        "mean_tz_error_mm": summarize([row["tz_error_mm"] for row in per_query], "mean"),
                        "mean_translation_l2_error_mm": summarize(
                            [row["translation_l2_error_mm"] for row in per_query], "mean"
                        ),
                        "median_inliers": summarize([row["ransac_inliers"] for row in per_query], "median"),
                        "mean_inliers": summarize([row["ransac_inliers"] for row in per_query], "mean"),
                        "median_consensus_size": summarize(
                            [row["ransac_consensus_size"] for row in per_query], "median"
                        ),
                        "mean_consensus_size": summarize(
                            [row["ransac_consensus_size"] for row in per_query], "mean"
                        ),
                    }
                )

    summary_rows.sort(
        key=lambda row: (
            float(row["mean_rot_mae_deg"]),
            -float(row["success_rate"]),
            abs(float(row["tau_R_deg"]) - args.selected_rot_threshold)
            + abs(float(row["tau_t_mm"]) - args.selected_trans_threshold) / 10.0,
        )
    )
    ranked_rows = []
    for rank, row in enumerate(summary_rows, start=1):
        updated = dict(row)
        updated["rank_by_mean_rot_mae"] = rank
        updated["is_selected_setting"] = int(
            abs(float(row["tau_R_deg"]) - args.selected_rot_threshold) < 1e-9
            and abs(float(row["tau_t_mm"]) - args.selected_trans_threshold) < 1e-9
            and int(row["min_inliers"]) == args.selected_min_inliers
        )
        ranked_rows.append(updated)
    summary_rows = ranked_rows

    write_csv(args.output_dir / "synthetic_ransac_threshold_sweep_summary.csv", summary_rows)
    write_csv(args.output_dir / "synthetic_ransac_threshold_sweep_predictions.csv", prediction_rows)
    write_csv(
        args.output_dir / f"heatmap_mean_rot_mae_min{args.selected_min_inliers}.csv",
        build_heatmap_rows(summary_rows, "mean_rot_mae_deg", args.selected_min_inliers),
    )
    write_csv(
        args.output_dir / f"heatmap_success_rate_min{args.selected_min_inliers}.csv",
        build_heatmap_rows(summary_rows, "success_rate", args.selected_min_inliers),
    )

    selected = [
        row
        for row in summary_rows
        if int(row["is_selected_setting"]) == 1
    ][0]
    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "rot_thresholds_deg": rot_thresholds,
        "trans_thresholds_mm": trans_thresholds,
        "min_inliers": min_inlier_values,
        "selected_setting": {
            "tau_R_deg": args.selected_rot_threshold,
            "tau_t_mm": args.selected_trans_threshold,
            "min_inliers": args.selected_min_inliers,
        },
        "queries": len(groups),
        "corruption": {
            "enabled": args.corruption_fraction > 0.0,
            "fraction": args.corruption_fraction,
            "rot_deg": args.corruption_rot_deg,
            "trans_mm": args.corruption_trans_mm,
            "seed": args.corruption_seed,
            "mode": "deterministic per sweep ",
        },
        "source_metadata": json.loads((args.input_dir / "metadata.json").read_text())
        if (args.input_dir / "metadata.json").exists()
        else {},
    }
    (args.output_dir / "synthetic_ransac_threshold_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    md = [
        "# Synthetic RANSAC Threshold Sweep",
        "",
        f"Input candidates: `{args.input_dir}`",
        f"Queries: {len(groups)}",
        (
            f"Corruption: {100.0 * args.corruption_fraction:.1f}% anchors/query, "
            f"rot={args.corruption_rot_deg:g} deg, trans sigma={args.corruption_trans_mm:g} mm, "
            f"seed={args.corruption_seed}"
        ),
        "",
        "Selected paper setting:",
        (
            f"- tau_R={args.selected_rot_threshold:g} deg, "
            f"tau_t={args.selected_trans_threshold:g} mm, "
            f"min inliers={args.selected_min_inliers}"
        ),
        (
            f"- success={100.0 * float(selected['success_rate']):.1f}%, "
            f"mean Rot MAE={float(selected['mean_rot_mae_deg']):.4f} deg, "
            f"median inliers={float(selected['median_inliers']):.1f}, "
            f"rank by mean Rot MAE={int(selected['rank_by_mean_rot_mae'])}/{len(summary_rows)}"
        ),
        "",
        "Top settings by mean rotation MAE:",
        "",
        "| rank | tau_R | tau_t | min inliers | success | Rot MAE | median inliers |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows[:12]:
        md.append(
            f"| {int(row['rank_by_mean_rot_mae'])} | "
            f"{float(row['tau_R_deg']):g} | {float(row['tau_t_mm']):g} | "
            f"{int(row['min_inliers'])} | {100.0 * float(row['success_rate']):.1f}% | "
            f"{float(row['mean_rot_mae_deg']):.4f} | {float(row['median_inliers']):.1f} |"
        )
    (args.output_dir / "synthetic_ransac_threshold_sweep_summary.md").write_text("\n".join(md) + "\n")
    print(args.output_dir / "synthetic_ransac_threshold_sweep_summary.csv")
    print(args.output_dir / "synthetic_ransac_threshold_sweep_summary.md")


if __name__ == "__main__":
    main()
