#!/usr/bin/env python3
"""Criterion-count heatmap for corrupted synthetic RANSAC thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path("outputs/synthetic_threshold_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--min-inliers", type=int, default=3)
    parser.add_argument("--rot-tolerance-deg", type=float, default=0.12)
    parser.add_argument("--trans-tolerance-mm", type=float, default=0.05)
    parser.add_argument("--inlier-plateau-ratio", type=float, default=0.99)
    parser.add_argument("--hide-rot-thresholds", default="7.5")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    rows_all = [
        row
        for row in read_csv(args.sweep_dir / "synthetic_ransac_threshold_sweep_summary.csv")
        if int(float(row["min_inliers"])) == args.min_inliers
    ]
    hidden = {float(value) for value in args.hide_rot_thresholds.split(",") if value.strip()}
    rows = [row for row in rows_all if f(row, "tau_R_deg") not in hidden]

    best_trans = min(f(row, "mean_translation_l2_error_mm") for row in rows_all)
    max_inliers = max(f(row, "mean_inliers") for row in rows_all)
    best_rot = min(f(row, "mean_rot_mae_deg") for row in rows_all)

    rot_limit = best_rot + args.rot_tolerance_deg
    trans_limit = best_trans + args.trans_tolerance_mm
    inlier_limit = args.inlier_plateau_ratio * max_inliers

    rot_values = sorted({f(row, "tau_R_deg") for row in rows})
    trans_values = sorted({f(row, "tau_t_mm") for row in rows})
    by_key = {(f(row, "tau_R_deg"), f(row, "tau_t_mm")): row for row in rows}

    table_rows: list[dict[str, object]] = []
    count_grid = np.zeros((len(rot_values), len(trans_values)), dtype=np.float64)
    for row_idx, tau_r in enumerate(rot_values):
        for col_idx, tau_t in enumerate(trans_values):
            row = by_key[(tau_r, tau_t)]
            pass_trans = f(row, "mean_translation_l2_error_mm") <= trans_limit
            pass_inliers = f(row, "mean_inliers") >= inlier_limit
            pass_rot = f(row, "mean_rot_mae_deg") <= rot_limit
            count = int(pass_trans) + int(pass_inliers) + int(pass_rot)
            count_grid[row_idx, col_idx] = count
            table_rows.append(
                {
                    "tau_R_deg": tau_r,
                    "tau_t_mm": tau_t,
                    "criteria_passed": count,
                    "pass_trans_l2": int(pass_trans),
                    "pass_mean_inliers": int(pass_inliers),
                    "pass_rot_mae": int(pass_rot),
                    "mean_rot_mae_deg": f(row, "mean_rot_mae_deg"),
                    "mean_translation_l2_error_mm": f(row, "mean_translation_l2_error_mm"),
                    "mean_inliers": f(row, "mean_inliers"),
                }
            )

    passing = [row for row in table_rows if int(row["criteria_passed"]) == 3]
    passing.sort(key=lambda row: (float(row["tau_R_deg"]), float(row["tau_t_mm"])))
    selected = passing[0] if passing else None

    fig, ax = plt.subplots(figsize=(6.3, 4.7), dpi=240)
    image = ax.imshow(count_grid, cmap="YlGn", vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(np.arange(len(trans_values)), [f"{value:g}" for value in trans_values])
    ax.set_yticks(np.arange(len(rot_values)), [f"{value:g}" for value in rot_values])
    ax.set_xlabel(r"$\tau_t$ (mm)")
    ax.set_ylabel(r"$\tau_R$ (deg)")
    ax.set_title("Corrupted Synthetic Criterion Count")

    for row_idx, tau_r in enumerate(rot_values):
        for col_idx, tau_t in enumerate(trans_values):
            row = by_key[(tau_r, tau_t)]
            count = int(count_grid[row_idx, col_idx])
            text_color = "white" if count <= 1 else "black"
            label = (
                f"{count}/3\n"
                f"R {f(row, 'mean_rot_mae_deg'):.4f}\n"
                f"T {f(row, 'mean_translation_l2_error_mm'):.4f}\n"
                f"I {f(row, 'mean_inliers'):.2f}"
            )
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontsize=5.8, linespacing=1.02)
            if selected and tau_r == float(selected["tau_R_deg"]) and tau_t == float(selected["tau_t_mm"]):
                ax.scatter([col_idx], [row_idx], marker="s", s=360, facecolors="none", edgecolors="red", linewidths=2.0)

    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    colorbar.set_label("# criteria satisfied")
    fig.tight_layout()

    out_png = args.sweep_dir / "synthetic_ransac_corrupted_criterion_count_heatmap_min3.png"
    out_csv = args.sweep_dir / "synthetic_ransac_corrupted_criterion_count_min3.csv"
    out_md = args.sweep_dir / "synthetic_ransac_corrupted_criterion_count_min3.md"
    fig.savefig(out_png)
    plt.close(fig)
    write_csv(out_csv, table_rows)

    lines = [
        "# Corrupted Synthetic Criterion Count",
        "",
        f"Displayed thresholds hide tau_R values: `{args.hide_rot_thresholds}`.",
        "",
        "Criteria:",
        f"- Trans. L2 <= best + {args.trans_tolerance_mm:g} mm = {trans_limit:.8f} mm",
        f"- Mean inliers >= {100.0 * args.inlier_plateau_ratio:.1f}% of max = {inlier_limit:.8f}",
        f"- Rot. MAE <= best + {args.rot_tolerance_deg:g} deg = {rot_limit:.8f} deg",
        "",
    ]
    if selected:
        lines.extend(
            [
                "Smallest threshold pair satisfying all criteria:",
                f"- tau_R={float(selected['tau_R_deg']):g} deg, tau_t={float(selected['tau_t_mm']):g} mm",
                (
                    f"- Rot MAE={float(selected['mean_rot_mae_deg']):.8f}, "
                    f"Trans L2={float(selected['mean_translation_l2_error_mm']):.8f}, "
                    f"Mean inliers={float(selected['mean_inliers']):.8f}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "| tau_R | tau_t | count | Rot MAE | Trans L2 | Mean inliers |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table_rows:
        lines.append(
            f"| {float(row['tau_R_deg']):g} | {float(row['tau_t_mm']):g} | "
            f"{int(row['criteria_passed'])}/3 | "
            f"{float(row['mean_rot_mae_deg']):.8f} | "
            f"{float(row['mean_translation_l2_error_mm']):.8f} | "
            f"{float(row['mean_inliers']):.8f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(out_png)
    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
