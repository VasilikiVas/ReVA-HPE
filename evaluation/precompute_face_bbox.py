import os
import glob
import pickle
import argparse

import numpy as np
import trimesh
from tqdm import tqdm


def load_h2c_opencv(view_dir):
    cam_path = os.path.join(view_dir, "output_opencv_camera.pkl")
    obj_path = os.path.join(view_dir, "output_object_transform_post_render.pkl")

    with open(cam_path, "rb") as f:
        cam = pickle.load(f)
    with open(obj_path, "rb") as f:
        obj = pickle.load(f)

    R_w2c = np.array(cam["R"], dtype=np.float64)
    t_w2c = np.array(cam["t"], dtype=np.float64).flatten()

    R_h2w = np.array(obj["R_world"], dtype=np.float64)
    t_h2w = np.array(obj["t_world"], dtype=np.float64).flatten()

    R_h2c = R_w2c @ R_h2w
    t_h2c = R_w2c @ t_h2w + t_w2c
    return R_h2c, t_h2c


def load_intrinsics_opencv(view_dir):
    cam_path = os.path.join(view_dir, "output_opencv_camera.pkl")
    with open(cam_path, "rb") as f:
        cam = pickle.load(f)
    K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
    return K


def compute_face_bbox(view_dir, mesh_filename="output_mesh.obj"):
    """
    Project mesh vertices through H2C + K to get the 2D face bounding box.

    Returns:
        np.ndarray: [xmin, xmax, ymin, ymax] in pixel coordinates.
    """
    K = load_intrinsics_opencv(view_dir)
    R_h2c, t_h2c = load_h2c_opencv(view_dir)

    mesh_path = os.path.join(view_dir, mesh_filename)
    mesh = trimesh.load(mesh_path, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)  # (N, 3) in head frame

    # head -> camera
    pts_cam = (R_h2c @ verts.T).T + t_h2c  # (N, 3)

    # keep only vertices in front of camera (z > 0)
    valid = pts_cam[:, 2] > 0
    if valid.sum() == 0:
        raise ValueError(f"No vertices in front of camera for {view_dir}")
    pts_cam = pts_cam[valid]

    # camera -> pixel:  p = K @ p_cam,  then  [x, y] = p[:2] / p[2]
    pts_2d_h = (K @ pts_cam.T).T  # (N, 3)
    pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]  # (N, 2)  columns [x, y]

    xmin, ymin = pts_2d.min(axis=0)
    xmax, ymax = pts_2d.max(axis=0)

    return np.array([xmin, xmax, ymin, ymax], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Precompute face bounding boxes")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of the dataset (contains identity_* folders)",
    )
    parser.add_argument(
        "--mesh_filename",
        type=str,
        default="output_mesh.obj",
        help="Name of the mesh file inside each view directory",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="face_bbox.npy",
        help="Name of the output .npy file saved in each view directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bbox files",
    )
    args = parser.parse_args()

    # Find all view directories
    view_dirs = sorted(
        glob.glob(os.path.join(args.root_dir, "identity_*/lighting_*/expr_*/view_*"))
    )
    print(f"Found {len(view_dirs)} view directories")

    success, skipped, failed = 0, 0, 0

    for view_dir in tqdm(view_dirs, desc="Computing bboxes"):
        out_path = os.path.join(view_dir, args.output_filename)

        # Skip if already computed
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        # Check required files exist
        required = [
            os.path.join(view_dir, "output_opencv_camera.pkl"),
            os.path.join(view_dir, "output_object_transform_post_render.pkl"),
            os.path.join(view_dir, args.mesh_filename),
        ]
        if not all(os.path.exists(f) for f in required):
            tqdm.write(f"SKIP (missing files): {view_dir}")
            skipped += 1
            continue

        try:
            bbox = compute_face_bbox(view_dir, mesh_filename=args.mesh_filename)
            np.save(out_path, bbox)
            success += 1
        except Exception as e:
            tqdm.write(f"FAIL: {view_dir} — {e}")
            failed += 1

    print(f"\nDone: {success} computed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
