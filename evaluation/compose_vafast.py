#!/usr/bin/env python3
"""VA-fast composition (CPU, torch-free): full-set VA-fast rotation rows.

    R_vafast(q) = dR(q<-e) . R_bank(e) = R_pnprel(q) . R_pnp(e)^T . R_bank(e)

- R_pnprel(q): per-query PnP-enrollment+relative predictions (--pnprel-root,
  subject_XX/pose_predictions.csv from scripts/eval_biwi_pnp_relative_subject.sh).
- R_bank(e): SO(3)-projected mean of the inlier bank->enrollment candidates
  (--calib-root, written by scripts/eval_biwi_direct5r_subject.sh with
  --frame-selection enrollment_only).
- R_pnp(e): enrollment EPnP (300 it / 5 px / 0.999 / >=6 inliers + refine) on
  the alignment landmarks, cv2 RNG seeded 1728+subject for determinism.
- Per-axis errors use the BIWI euler + 6DRepNet wrap conventions of
  eval_biwi_crop.biwi_euler_deg_batch/angular_error, ported to numpy float32.
"""
from __future__ import annotations

import datetime as dt

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# Paths are provided via CLI arguments (see main()).
PNPREL = ALIGNMENT = BIWI_ROOT = None  # set in main()


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def row_matrix(row: dict, prefix: str) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{prefix}_r{i}{j}"]) for j in range(3)] for i in range(3)],
        dtype=np.float64,
    )


def fused_enrollment_rotation(pairwise_csv: Path) -> np.ndarray:
    """Inlier-mean, SO(3)-projected — mirrors iter_fused_pairwise for one frame."""
    by_frame: dict[str, list[np.ndarray]] = {}
    with pairwise_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["inlier"] == "1":
                by_frame.setdefault(row["frame_num"], []).append(row_matrix(row, "pred"))
    if len(by_frame) != 1:
        raise RuntimeError(f"{pairwise_csv}: expected one enrollment frame, got {sorted(by_frame)}")
    (rotations,) = by_frame.values()
    if not rotations:
        raise RuntimeError(f"{pairwise_csv}: no inlier candidates")
    return project_rotation(np.mean(rotations, axis=0))


def load_rgb_intrinsics(path: Path):
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    K = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
    R_cam = np.array([[float(x) for x in lines[i].split()] for i in range(4, 7)])
    return K, R_cam


def pnp_enrollment_rotation(subject: str, reference: str = "mean_neutral") -> np.ndarray:
    similarity = json.loads(
        (ALIGNMENT / f"subject{subject}" / "similarity/facelift_to_flame_similarity.json").read_text()
    )
    fit = json.loads(Path(similarity["source_flame_fit"]).read_text())
    object_points = np.asarray(
        fit["canonical_references"][reference]["landmarks_xyz_opencv"], dtype=np.float64
    )
    image_points = np.asarray(fit["fit"]["detected_landmarks_xy"], dtype=np.float64)
    intrinsics, rgb_rotation = load_rgb_intrinsics(BIWI_ROOT / subject / "rgb.cal")
    distortion = np.zeros((4, 1), dtype=np.float64)
    cv2.setRNGSeed(1728 + int(subject))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, intrinsics.astype(np.float64), distortion,
        iterationsCount=300, reprojectionError=5.0, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError(f"subject {subject}: enrollment PnP failed")
    ids = inliers.reshape(-1)
    ok, rvec, tvec = cv2.solvePnP(
        object_points[ids], image_points[ids], intrinsics.astype(np.float64), distortion,
        rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError(f"subject {subject}: PnP refinement failed")
    rotation_rgb = cv2.Rodrigues(rvec)[0]
    return project_rotation(rgb_rotation.T @ rotation_rgb)


def biwi_euler_deg(R: np.ndarray) -> np.ndarray:
    """Port of eval_biwi_crop.biwi_euler_deg_batch (float32, single matrix)."""
    Rt = R.astype(np.float32).T
    yaw = -np.arctan2(-Rt[2, 0], np.sqrt(Rt[2, 1] ** 2 + Rt[2, 2] ** 2)) * (180.0 / np.pi)
    pitch = np.arctan2(Rt[2, 1], Rt[2, 2]) * (180.0 / np.pi)
    roll = -np.arctan2(Rt[1, 0], Rt[0, 0]) * (180.0 / np.pi)
    return np.asarray([yaw, pitch, roll], dtype=np.float64)


def angular_error(gt_deg: np.ndarray, pred_deg: np.ndarray) -> np.ndarray:
    """Port of eval_biwi_crop.angular_error (6DRepNet 360/180 wrapping)."""
    candidates = np.stack(
        [
            np.abs(gt_deg - pred_deg),
            np.abs(pred_deg + 360.0 - gt_deg),
            np.abs(pred_deg - 360.0 - gt_deg),
            np.abs(pred_deg + 180.0 - gt_deg),
            np.abs(pred_deg - 180.0 - gt_deg),
        ]
    )
    return candidates.min(axis=0)


def main() -> int:
    global PNPREL, ALIGNMENT, BIWI_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, nargs="+", default=[20])
    ap.add_argument("--calib-root", type=Path, required=True,
                    help="directory with calib_nd{M}/subject_XX_pairwise.csv from the bank evaluator run in --frame-selection enrollment_only mode")
    ap.add_argument("--pnprel-root", type=Path, required=True,
                    help="per-subject PnP-relative outputs (subject_XX/pose_predictions.csv)")
    ap.add_argument("--alignment-root", type=Path, required=True,
                    help="enrollment alignment root (subjectXX/similarity/...)")
    ap.add_argument("--biwi-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    PNPREL, ALIGNMENT, BIWI_ROOT = args.pnprel_root, args.alignment_root, args.biwi_root
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for m in args.m:
        calib_dir = args.calib_root / f"calib_nd{m}"
        errors = []
        for sid in range(1, 25):
            subject = f"{sid:02d}"
            bank_e = fused_enrollment_rotation(calib_dir / f"subject_{subject}_pairwise.csv")
            pnp_e = pnp_enrollment_rotation(subject)
            correction = pnp_e.T @ bank_e
            with (PNPREL / f"subject_{subject}" / "pose_predictions.csv").open(newline="") as f:
                for row in csv.DictReader(f):
                    r_vafast = project_rotation(row_matrix(row, "pred") @ correction)
                    errors.append(
                        angular_error(biwi_euler_deg(row_matrix(row, "gt")), biwi_euler_deg(r_vafast))
                    )
        errs = np.stack(errors)
        yaw, pitch, roll = errs.mean(axis=0)
        maer = errs.mean()
        print(f"VA-fast M={m}: N={len(errs)} yaw={yaw:.4f} pitch={pitch:.4f} roll={roll:.4f} MAEr={maer:.6f}")
        summary_rows.append(
            {
                "protocol": "VA-fast",
                "directions": m,
                "candidates_per_query": 1,
                "calibration_predictions": m,
                "queries": len(errs),
                "yaw": f"{yaw:.6f}",
                "pitch": f"{pitch:.6f}",
                "roll": f"{roll:.6f}",
                "MAE_R": f"{maer:.6f}",
                "translation": "identical to PnP-relative branch (Table 1 ReVA row) by definition",
            }
        )

    out = args.out / "vafast_fullset_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    (args.out / "vafast_fullset_protocol.json").write_text(
        json.dumps(
            {
                "definition": "R_vafast(q) = R_pnprel(q) . R_pnp(e)^T . R_bank(e), calibration fused from enrollment-only pairwise inliers, errors via ported biwi_euler_deg_batch + angular_error",
                "pnprel_source": str(PNPREL),
                "alignment_root": str(ALIGNMENT),
                "cv2_seed": "1728 + subject (PnP RANSAC determinism)",
                "date": dt.date.today().isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
