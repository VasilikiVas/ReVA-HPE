import argparse
import csv
import logging
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


ARKIT_STD_TAN = 608.0 / 957.2577
ARKIT_CENTER_OFFSET_M = np.array([0.0, 0.01, -0.021], dtype=np.float32)
BIWI_OBJ_TO_ARKIT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)


def load_obj_vertices(path_to_file):
    vertices = []
    with open(path_to_file, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float32)


def parse_pose_txt(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    R = np.array([[float(v) for v in lines[i].split()] for i in range(3)], dtype=np.float32)
    t = np.array([float(v) for v in lines[3].split()], dtype=np.float32)
    return R, t


def parse_rgb_cal(path):
    with open(path) as f:
        rows = [[float(x) for x in l.strip().split()] for l in f if l.strip()]

    rows3 = [row for row in rows if len(row) == 3]
    if len(rows3) < 7:
        raise ValueError(f"Unexpected rgb.cal format in {path}")

    K = np.array(rows3[:3], dtype=np.float32)
    R_cal = np.array(rows3[3:6], dtype=np.float32)
    t_cal = np.array(rows3[6], dtype=np.float32)
    return K, R_cal, t_cal


def convert_pose_to_sixdof_gl(R_pose, t_pose_mm, R_cal, t_cal_mm):
    R_t_raw = np.eye(4, dtype=np.float32)
    R_t_raw[:3, :3] = R_pose.T
    R_t_raw[3, :3] = t_pose_mm

    R_t_calibration = np.eye(4, dtype=np.float32)
    R_t_calibration[:3, :3] = R_cal.T
    R_t_calibration[3, :3] = t_cal_mm

    R_t_cv = R_t_raw @ R_t_calibration
    R_t_cv[3, :3] /= 1000.0

    R_t_gl = R_t_cv.copy()
    euler = Rotation.from_matrix(R_t_gl[:3, :3].T).as_euler("yxz", degrees=True)
    euler[0] *= -1.0
    euler[2] *= -1.0
    R_t_gl[:3, :3] = Rotation.from_euler("yxz", euler, degrees=True).as_matrix().T.astype(np.float32)

    T_vec = R_t_gl[3, :3].copy()
    T_vec[1] *= -1.0
    T_vec[2] *= -1.0
    R_t_gl[3, :3] = T_vec

    return R_t_gl


def standardize_biwi_image_to_arkit_fov(img_rgb, K_img):
    img_h, img_w = img_rgb.shape[:2]
    fx = float(K_img[0, 0])
    fy = float(K_img[1, 1])
    cx = float(K_img[0, 2])
    cy = float(K_img[1, 2])

    M2 = np.array(
        [
            [fx, 0.0, 0.0, 0.0],
            [0.0, fy, 0.0, 0.0],
            [-cx, -(img_h - cy), -0.99999976, -1.0],
            [0.0, 0.0, -0.001, 0.0],
        ],
        dtype=np.float32,
    )

    M_proj = np.array(
        [
            [fx / (img_w / 2.0), 0.0, 0.0, 0.0],
            [0.0, fy / (img_h / 2.0), 0.0, 0.0],
            [M2[2, 0] / (img_w / 2.0) + 1.0, M2[2, 1] / (img_h / 2.0) + 1.0, -0.99999976, -1.0],
            [0.0, 0.0, -0.001, 0.0],
        ],
        dtype=np.float32,
    )

    pixel_expand = int(round(ARKIT_STD_TAN * fx))
    n = 0.01
    A, B, C, D = M_proj[0, 0], M_proj[1, 1], M_proj[2, 0], M_proj[2, 1]
    r = n * (C + 1.0) / A
    t = n * (D + 1.0) / B
    factor = fx / n

    r_pixel = int(round(r * factor))
    t_pixel = int(round(t * factor))

    delta_r = pixel_expand - r_pixel
    delta_l = pixel_expand * 2 - delta_r - img_w
    delta_t = pixel_expand - t_pixel
    delta_b = pixel_expand * 2 - delta_t - img_h

    temp = img_rgb.copy()
    if delta_r < 0:
        temp = temp[:, -delta_l:delta_r]
    else:
        temp = np.pad(temp, ((0, 0), (delta_l, delta_r), (0, 0)))

    if delta_t < 0:
        temp = temp[-delta_t:delta_b]
    else:
        temp = np.pad(temp, ((delta_t, delta_b), (0, 0), (0, 0)))

    standardized = cv2.resize(temp, (800, 800), interpolation=cv2.INTER_LINEAR)

    final_proj = M_proj.copy()
    final_proj[0, 0] = 1.0 / ARKIT_STD_TAN
    final_proj[1, 1] = 1.0 / ARKIT_STD_TAN
    final_proj[2, 0] = 0.0
    final_proj[2, 1] = 0.0

    return standardized, final_proj


def project_points_arkit(verts_m, R_t_gl, proj_matrix, image_size=800):
    ones = np.ones((verts_m.shape[0], 1), dtype=np.float32)
    verts_homo = np.concatenate([verts_m, ones], axis=1)
    M_img = np.array(
        [
            [image_size / 2.0, 0.0, 0.0, 0.0],
            [0.0, image_size / 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [image_size / 2.0, image_size / 2.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    verts = verts_homo @ R_t_gl @ proj_matrix @ M_img
    verts /= verts[:, [3]]
    points2d = verts[:, :2].copy()
    points2d[:, 1] = image_size - points2d[:, 1]
    return points2d.astype(np.float32)


def prepare_biwi_for_sixdof_face(
    biwi_root,
    output_root,
    subjects=None,
    overwrite=False,
    max_frames_per_subject=None,
):
    biwi_root = Path(biwi_root)
    output_root = Path(output_root)

    csv_dir = output_root / "csv"
    image_root = output_root / "image" / "BIWI_v1"
    info_root = output_root / "info" / "BIWI_v1"
    csv_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    info_root.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = sorted([p.name for p in biwi_root.iterdir() if p.is_dir() and p.name.isdigit()])
    else:
        subjects = [str(s).zfill(2) for s in subjects]

    rows = []
    total_written = 0

    for subj in subjects:
        subj_dir = biwi_root / subj
        obj_path = biwi_root / f"{subj}.obj"
        cal_path = subj_dir / "rgb.cal"
        if not subj_dir.is_dir() or not obj_path.is_file() or not cal_path.is_file():
            logging.warning(f"Skipping subject {subj}: missing directory/object/calibration.")
            continue

        verts_m = load_obj_vertices(obj_path) @ BIWI_OBJ_TO_ARKIT
        verts_m /= 1000.0
        K_img, R_cal, t_cal = parse_rgb_cal(cal_path)

        frame_paths = sorted(subj_dir.glob("frame_*_rgb.png"))
        if max_frames_per_subject is not None:
            frame_paths = frame_paths[:max_frames_per_subject]

        written_for_subject = 0
        for img_path in frame_paths:
            match = re.search(r"frame_(\d+)_rgb\.png", img_path.name)
            if match is None:
                continue
            frame_id = match.group(1)
            pose_path = img_path.with_name(img_path.name.replace("_rgb.png", "_pose.txt"))
            if not pose_path.is_file():
                continue

            out_img_dir = image_root / subj / "biwi"
            out_info_dir = info_root / subj / "biwi"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_info_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = out_img_dir / f"{frame_id}_ar.jpg"
            out_info_path = out_info_dir / f"{frame_id}_info.npz"

            if not overwrite and out_img_path.is_file() and out_info_path.is_file():
                rows.append(
                    {
                        "data_batch": "BIWI",
                        "subject_id": subj,
                        "facial_action": "biwi",
                        "img_id": frame_id,
                    }
                )
                written_for_subject += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            R_pose, t_pose = parse_pose_txt(pose_path)
            R_t_gl = convert_pose_to_sixdof_gl(R_pose, t_pose, R_cal, t_cal)

            center_mat = np.eye(4, dtype=np.float32)
            center_mat[3, :3] = ARKIT_CENTER_OFFSET_M
            verts_aligned = verts_m - ARKIT_CENTER_OFFSET_M
            R_t_aligned = center_mat @ R_t_gl

            standardized_img, proj_matrix = standardize_biwi_image_to_arkit_fov(img_rgb, K_img)
            points2d = project_points_arkit(verts_aligned, R_t_aligned, proj_matrix, image_size=800)

            cv2.imwrite(str(out_img_path), cv2.cvtColor(standardized_img, cv2.COLOR_RGB2BGR))
            np.savez_compressed(
                out_info_path,
                points2d=points2d,
                verts_gt=verts_aligned.astype(np.float32),
                R_t=R_t_aligned.astype(np.float32),
            )

            rows.append(
                {
                    "data_batch": "BIWI",
                    "subject_id": subj,
                    "facial_action": "biwi",
                    "img_id": frame_id,
                }
            )
            written_for_subject += 1
            total_written += 1

        logging.info(f"Prepared subject {subj}: {written_for_subject} frames")

    rows = sorted(rows, key=lambda r: (r["subject_id"], r["img_id"]))
    csv_path = csv_dir / "metadata_biwi.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["data_batch", "subject_id", "facial_action", "img_id"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Saved metadata CSV: {csv_path}")
    logging.info(f"Output root ready: {output_root}")
    return str(output_root)


def parse_args():
    p = argparse.ArgumentParser(description="Prepare BIWI in 6DoF Face evaluation format")
    p.add_argument("--biwi_root", required=True, help="Raw BIWI root containing subject folders, rgb.cal, pose.txt, and subject .obj files.")
    p.add_argument("--output_root", required=True, help="Output root that will contain csv/, image/, and info/.")
    p.add_argument("--subjects", type=int, nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_frames_per_subject", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_biwi_for_sixdof_face(
        biwi_root=args.biwi_root,
        output_root=args.output_root,
        subjects=args.subjects,
        overwrite=args.overwrite,
        max_frames_per_subject=args.max_frames_per_subject,
    )
