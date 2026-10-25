import argparse
import csv
import gc
import logging
import os
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_biwi_for_sixdof_face import prepare_biwi_for_sixdof_face


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_SUBJECTS = [5, 6, 9, 14, 16, 17, 20, 24]
CANONICAL_VERT_COUNT = 1220

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _empty_acc():
    return {
        "yaw": [],
        "pitch": [],
        "roll": [],
        "tx": [],
        "ty": [],
        "tz": [],
        "tx_bias": [],
        "ty_bias": [],
        "tz_bias": [],
    }


def _parse_subject_from_img_path(img_path):
    path = Path(img_path)
    if len(path.parts) < 3:
        return "unknown"
    return path.parts[-3]


def _resolve_data_root(args):
    if args.sixdof_face_data_root:
        return args.sixdof_face_data_root
    if args.sixdof_face_cache_root:
        return args.sixdof_face_cache_root
    return os.path.join(args.sixdof_face_repo, "dataset", "ARKitFace")


def _data_root_is_ready(path):
    path = Path(path)
    return (
        (path / "csv" / "metadata_biwi.csv").is_file()
        and (path / "image").is_dir()
        and (path / "info").is_dir()
    )


def _prepare_data_root_if_needed(args):
    data_root = _resolve_data_root(args)
    if _data_root_is_ready(data_root):
        return data_root

    if not args.prepare_sixdof_face_if_missing:
        return data_root

    logging.info(
        "6DoF Face BIWI evaluation root is missing, preparing it from raw BIWI data at %s",
        args.biwi_dir,
    )
    subjects = [str(s).zfill(2) for s in args.test_subjects] if args.test_subjects else None
    prepare_biwi_for_sixdof_face(
        biwi_root=args.biwi_dir,
        output_root=data_root,
        subjects=subjects,
        overwrite=False,
    )
    return data_root


def _resolve_source_checkpoints_dir(args):
    return args.sixdof_face_checkpoints_dir or os.path.join(
        args.sixdof_face_repo, "checkpoint"
    )


def _prepare_runtime_checkpoints_dir(args):
    source_checkpoints_dir = os.path.abspath(_resolve_source_checkpoints_dir(args))
    source_run_dir = os.path.join(source_checkpoints_dir, args.sixdof_face_name)
    source_ckpt_path = os.path.join(
        source_run_dir,
        f"{args.sixdof_face_epoch}_net_R.pth",
    )

    runtime_root = os.path.join(SCRIPT_DIR, "cache", "sixdof_face_runtime_checkpoints")
    runtime_run_dir = os.path.join(runtime_root, args.sixdof_face_name)
    runtime_ckpt_path = os.path.join(
        runtime_run_dir,
        f"{args.sixdof_face_epoch}_net_R.pth",
    )

    os.makedirs(runtime_run_dir, exist_ok=True)

    source_stat = os.stat(source_ckpt_path)
    needs_refresh = True
    if os.path.isfile(runtime_ckpt_path):
        runtime_stat = os.stat(runtime_ckpt_path)
        needs_refresh = (
            runtime_stat.st_size != source_stat.st_size
            or runtime_stat.st_mtime < source_stat.st_mtime
        )

    if needs_refresh:
        if os.path.lexists(runtime_ckpt_path):
            os.unlink(runtime_ckpt_path)
        try:
            os.symlink(source_ckpt_path, runtime_ckpt_path)
        except OSError:
            shutil.copy2(source_ckpt_path, runtime_ckpt_path)

    return runtime_root


def _validate_paths(args):
    if not os.path.isdir(args.sixdof_face_repo):
        raise FileNotFoundError(f"6DoF Face repo not found: {args.sixdof_face_repo}")

    checkpoints_dir = _resolve_source_checkpoints_dir(args)
    ckpt_path = os.path.join(
        checkpoints_dir,
        args.sixdof_face_name,
        f"{args.sixdof_face_epoch}_net_R.pth",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"6DoF Face checkpoint not found: {ckpt_path}")

    data_root = _prepare_data_root_if_needed(args)
    if not _data_root_is_ready(data_root):
        raise FileNotFoundError(
            f"6DoF Face BIWI evaluation root not found: {data_root}\n"
            "Pass --sixdof_face_data_root to a prepared BIWI eval root containing csv/, image/, and info/, "
            "or enable --prepare_sixdof_face_if_missing."
        )


def _build_test_options(args, checkpoints_dir):
    gpu_ids = str(args.gpu if torch.cuda.is_available() else -1)
    cmd_line = " ".join(
        [
            "--dataset_mode",
            "biwi",
            "--model",
            args.sixdof_face_model,
            "--img_size",
            str(args.sixdof_face_img_size),
            "--batch_size",
            str(args.sixdof_face_batch_size),
            "--num_threads",
            str(args.sixdof_face_num_workers),
            "--checkpoints_dir",
            checkpoints_dir,
            "--name",
            args.sixdof_face_name,
            "--epoch",
            args.sixdof_face_epoch,
            "--gpu_ids",
            gpu_ids,
            "--use_gt_bbox",
        ]
    )
    return cmd_line


def _sample_or_wrap_indices(num_items, target_items):
    if num_items <= 0:
        raise ValueError("Cannot sample from an empty set of BIWI correspondences.")
    if num_items >= target_items:
        return np.linspace(0, num_items - 1, num=target_items, dtype=np.int64)
    return np.resize(np.arange(num_items, dtype=np.int64), target_items)


def _build_biwi_correspondences(points2d, verts_gt, tform, img_size, n_pts, kpt_ind):
    W, b = tform.T[:2], tform.T[2]
    local_points2d = points2d @ W + b

    finite = np.isfinite(local_points2d).all(axis=1)
    valid = (
        finite
        & (local_points2d[:, 0] >= 0.0)
        & (local_points2d[:, 0] < img_size)
        & (local_points2d[:, 1] >= 0.0)
        & (local_points2d[:, 1] < img_size)
    )

    candidate_ids = np.flatnonzero(valid)
    candidate_points = local_points2d[candidate_ids]

    if candidate_ids.size < 4:
        candidate_ids = np.flatnonzero(finite)
        candidate_points = np.clip(local_points2d[candidate_ids], 0.0, img_size - 1.001)

    if candidate_ids.size < 4:
        kpt_ids = np.asarray(kpt_ind, dtype=np.int64)
        kpt_points = np.clip(local_points2d[kpt_ids], 0.0, img_size - 1.001)
        kpt_finite = np.isfinite(kpt_points).all(axis=1)
        candidate_ids = kpt_ids[kpt_finite]
        candidate_points = kpt_points[kpt_finite]

    if candidate_ids.size == 0:
        candidate_ids = np.zeros(4, dtype=np.int64)
        center = np.array([img_size / 2.0, img_size / 2.0], dtype=np.float32)
        offsets = np.array(
            [[-12.0, -12.0], [12.0, -12.0], [-12.0, 12.0], [12.0, 12.0]],
            dtype=np.float32,
        )
        candidate_points = np.clip(center + offsets, 0.0, img_size - 1.001)

    sample_idx = _sample_or_wrap_indices(candidate_ids.size, n_pts)
    sampled_vertex_ids = candidate_ids[sample_idx]
    sampled_points = candidate_points[sample_idx]

    px = np.clip(np.floor(sampled_points[:, 0]).astype(np.int64), 0, img_size - 1)
    py = np.clip(np.floor(sampled_points[:, 1]).astype(np.int64), 0, img_size - 1)
    choose = py * img_size + px

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    if candidate_points.shape[0] >= 3:
        hull = cv2.convexHull(np.round(candidate_points).astype(np.float32))
        cv2.fillConvexPoly(mask, np.ascontiguousarray(hull.astype(np.int32)), 1)
    else:
        mask[py, px] = 1

    nocs = verts_gt[sampled_vertex_ids] * 4.5
    return choose, mask, nocs


def _build_patched_biwi_dataset_class(biwi_dataset_module):
    class PatchedBIWIDataset(biwi_dataset_module.BIWIDataset):
        def __getitem__(self, index):
            data_batch = str(self.df["data_batch"][index])
            subject_id = str(self.df["subject_id"][index])
            facial_action = str(self.df["facial_action"][index])
            img_id = str(self.df["img_id"][index])

            img_path = (
                self.data_root
                / "image"
                / data_batch
                / subject_id
                / facial_action
                / f"{img_id}_ar.jpg"
            )
            npz_path = (
                self.data_root
                / "info"
                / data_batch
                / subject_id
                / facial_action
                / f"{img_id}_info.npz"
            )
            M = np.load(npz_path)

            if biwi_dataset_module.use_jpeg4py:
                img_raw = biwi_dataset_module.jpeg.JPEG(img_path).decode()
            else:
                img_raw = cv2.imread(str(img_path))
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            points2d = M["points2d"].astype(np.float32)
            points2d_68 = points2d[self.kpt_ind]
            verts_gt = M["verts_gt"].astype(np.float32)

            x_min, y_min = points2d_68.min(axis=0)
            x_max, y_max = points2d_68.max(axis=0)
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            size = max(x_max - x_min, y_max - y_min)
            ss = np.array([0.75, 0.75, 0.75, 0.75], dtype=np.float32)

            if self.is_train:
                rnd_size = 0.15 * size
                x_center += np.random.uniform(-rnd_size, rnd_size)
                y_center += np.random.uniform(-rnd_size, rnd_size)
                ss *= np.random.uniform(0.95, 1.05)

            left = x_center - ss[0] * size
            right = x_center + ss[1] * size
            top = y_center - ss[2] * size
            bottom = y_center + ss[3] * size

            src_pts = np.float32(
                [
                    [left, top],
                    [left, bottom],
                    [right, top],
                ]
            )
            tform = cv2.getAffineTransform(src_pts, self.dst_pts)
            img = cv2.warpAffine(img_raw, tform, (self.img_size,) * 2)

            if self.is_train:
                img = self.tfm_train(biwi_dataset_module.Image.fromarray(img))
            else:
                img = self.tfm_test(biwi_dataset_module.Image.fromarray(img))

            choose, mask, nocs = _build_biwi_correspondences(
                points2d=points2d,
                verts_gt=verts_gt,
                tform=tform,
                img_size=self.img_size,
                n_pts=self.opt.n_pts,
                kpt_ind=self.kpt_ind,
            )

            placeholder_model = np.zeros((CANONICAL_VERT_COUNT, 3), dtype=np.float32)
            placeholder_corr_mat = np.zeros(
                (self.opt.n_pts, CANONICAL_VERT_COUNT), dtype=np.float32
            )

            d = {
                "img": img,
                "choose": torch.tensor(choose.astype(np.int64)),
                "model": torch.tensor(placeholder_model),
                "uvmap": torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32),
                "nocs": torch.tensor(nocs.astype(np.float32)),
                "mask": torch.tensor(mask.astype(np.int64)),
                "corr_mat": torch.tensor(placeholder_corr_mat),
            }

            if hasattr(self.opt, "eval"):
                d["tform_inv"] = cv2.getAffineTransform(self.dst_pts, src_pts)
                d["R_t"] = M["R_t"]
                d["img_path"] = str(img_path)

            return d

    return PatchedBIWIDataset


def _build_safe_solvepnp_ransac(original_solvepnp_ransac):
    def _fail_result():
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)
        inliers = np.empty((0, 1), dtype=np.int32)
        return False, rvec, tvec, inliers

    def _safe_solvepnp_ransac(object_points, image_points, camera_matrix, dist_coeffs=None, *args, **kwargs):
        object_arr = np.asarray(object_points)
        image_arr = np.asarray(image_points)

        if object_arr.ndim < 2 or image_arr.ndim < 2:
            return _fail_result()
        if object_arr.shape[0] < 4 or image_arr.shape[0] < 4:
            return _fail_result()
        if not np.isfinite(object_arr).all() or not np.isfinite(image_arr).all():
            return _fail_result()

        try:
            retval, rvec, tvec, inliers = original_solvepnp_ransac(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                *args,
                **kwargs,
            )
        except cv2.error:
            return _fail_result()

        if rvec is None or tvec is None:
            return _fail_result()

        return retval, rvec, tvec, inliers

    return _safe_solvepnp_ransac


def _decode_batch_translation_bias_mm(model):
    inst_shape = model.verts3d_pred / 9.0
    assign_mat = F.softmax(model.assign_mat, dim=2)
    nocs_coords = torch.bmm(assign_mat, inst_shape)
    nocs_coords = nocs_coords.detach().cpu().numpy().reshape(-1, model.opt.n_pts, 3)
    choose_arr = model.choose.detach().cpu().numpy()

    img_size = 800
    f = 1.574437 * img_size / 2
    K_img = np.array(
        [
            [f, 0.0, img_size / 2.0],
            [0.0, f, img_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    batch_bias_mm = []
    bs = choose_arr.shape[0]
    for i in range(bs):
        choose = choose_arr[i]
        choose, choose_idx = np.unique(choose, return_index=True)
        nocs_coord = nocs_coords[i, choose_idx, :].astype(np.float64)
        col_idx = choose % model.opt.img_size
        row_idx = choose // model.opt.img_size
        local_pts2d = np.concatenate(
            (col_idx.reshape(-1, 1), row_idx.reshape(-1, 1)),
            axis=1,
        ).astype(np.float64)

        tfm = np.asarray(model.tform_inv_list[i], dtype=np.float64)
        W, b = tfm.T[:2], tfm.T[2]
        global_pts_pred = local_pts2d @ W + b

        _, rvecs, tvecs, _ = cv2.solvePnPRansac(
            nocs_coord,
            global_pts_pred,
            K_img,
            None,
        )

        rotM = cv2.Rodrigues(rvecs)[0].T
        tvecs = tvecs.squeeze(axis=1)

        R_temp = np.identity(4, dtype=np.float64)
        R_temp[:3, :3] = rotM
        R_temp[3, :3] = tvecs
        R_t_pred = R_temp @ T

        if R_t_pred[3, 2] > 0 or R_t_pred[3, 2] < -100:
            R_t_pred = np.identity(4, dtype=np.float64)

        R_t_gt = np.asarray(model.R_t_list[i], dtype=np.float64)
        bias_mm = (R_t_pred[3, :3] - R_t_gt[3, :3]) * 1000.0
        batch_bias_mm.append(bias_mm.astype(np.float64))

    return batch_bias_mm


def _summarize_metrics(
    yaw_arr,
    pitch_arr,
    roll_arr,
    tx_arr_mm,
    ty_arr_mm,
    tz_arr_mm,
    tx_bias_arr_mm,
    ty_bias_arr_mm,
    tz_bias_arr_mm,
):
    yaw_arr = np.asarray(yaw_arr, dtype=np.float64)
    pitch_arr = np.asarray(pitch_arr, dtype=np.float64)
    roll_arr = np.asarray(roll_arr, dtype=np.float64)
    tx_arr_mm = np.asarray(tx_arr_mm, dtype=np.float64)
    ty_arr_mm = np.asarray(ty_arr_mm, dtype=np.float64)
    tz_arr_mm = np.asarray(tz_arr_mm, dtype=np.float64)
    tx_bias_arr_mm = np.asarray(tx_bias_arr_mm, dtype=np.float64)
    ty_bias_arr_mm = np.asarray(ty_bias_arr_mm, dtype=np.float64)
    tz_bias_arr_mm = np.asarray(tz_bias_arr_mm, dtype=np.float64)

    if yaw_arr.size == 0:
        return None

    frame_angle_mae = (yaw_arr + pitch_arr + roll_arr) / 3.0
    l2_arr_mm = np.sqrt(tx_arr_mm ** 2 + ty_arr_mm ** 2 + tz_arr_mm ** 2)

    return {
        "yaw": float(yaw_arr.mean()),
        "pitch": float(pitch_arr.mean()),
        "roll": float(roll_arr.mean()),
        "mean": float(frame_angle_mae.mean()),
        "yaw_median": float(np.median(yaw_arr)),
        "pitch_median": float(np.median(pitch_arr)),
        "roll_median": float(np.median(roll_arr)),
        "median": float(np.median(frame_angle_mae)),
        "tx": float(tx_arr_mm.mean()),
        "ty": float(ty_arr_mm.mean()),
        "tz": float(tz_arr_mm.mean()),
        "l2": float(l2_arr_mm.mean()),
        "tx_median": float(np.median(tx_arr_mm)),
        "ty_median": float(np.median(ty_arr_mm)),
        "tz_median": float(np.median(tz_arr_mm)),
        "l2_median": float(np.median(l2_arr_mm)),
        "tx_bias": float(tx_bias_arr_mm.mean()),
        "ty_bias": float(ty_bias_arr_mm.mean()),
        "tz_bias": float(tz_bias_arr_mm.mean()),
        "tx_bias_median": float(np.median(tx_bias_arr_mm)),
        "ty_bias_median": float(np.median(ty_bias_arr_mm)),
        "tz_bias_median": float(np.median(tz_bias_arr_mm)),
        "count": int(yaw_arr.size),
    }


def _save_results_csv(output_csv, overall, per_subject):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "subject",
            "model",
            "yaw_mae_deg",
            "pitch_mae_deg",
            "roll_mae_deg",
            "mean_mae_deg",
            "yaw_median_deg",
            "pitch_median_deg",
            "roll_median_deg",
            "median_frame_mae_deg",
            "tx_mae_mm",
            "ty_mae_mm",
            "tz_mae_mm",
            "translation_l2_mae_mm",
            "tx_median_mm",
            "ty_median_mm",
            "tz_median_mm",
            "translation_l2_median_mm",
            "tx_signed_bias_mean_mm",
            "ty_signed_bias_mean_mm",
            "tz_signed_bias_mean_mm",
            "tx_signed_bias_median_mm",
            "ty_signed_bias_median_mm",
            "tz_signed_bias_median_mm",
        ]
        writer.writerow(header)

        for subj, metrics in sorted(per_subject.items(), key=lambda item: item[0]):
            writer.writerow(
                [
                    subj,
                    "sixdof_face",
                    f"{metrics['yaw']:.4f}",
                    f"{metrics['pitch']:.4f}",
                    f"{metrics['roll']:.4f}",
                    f"{metrics['mean']:.4f}",
                    f"{metrics['yaw_median']:.4f}",
                    f"{metrics['pitch_median']:.4f}",
                    f"{metrics['roll_median']:.4f}",
                    f"{metrics['median']:.4f}",
                    f"{metrics['tx']:.4f}",
                    f"{metrics['ty']:.4f}",
                    f"{metrics['tz']:.4f}",
                    f"{metrics['l2']:.4f}",
                    f"{metrics['tx_median']:.4f}",
                    f"{metrics['ty_median']:.4f}",
                    f"{metrics['tz_median']:.4f}",
                    f"{metrics['l2_median']:.4f}",
                    f"{metrics['tx_bias']:.4f}",
                    f"{metrics['ty_bias']:.4f}",
                    f"{metrics['tz_bias']:.4f}",
                    f"{metrics['tx_bias_median']:.4f}",
                    f"{metrics['ty_bias_median']:.4f}",
                    f"{metrics['tz_bias_median']:.4f}",
                ]
            )

        if overall is not None:
            writer.writerow([])
            writer.writerow(["# Overall results"])
            writer.writerow(header)
            writer.writerow(
                [
                    "overall",
                    "sixdof_face",
                    f"{overall['yaw']:.4f}",
                    f"{overall['pitch']:.4f}",
                    f"{overall['roll']:.4f}",
                    f"{overall['mean']:.4f}",
                    f"{overall['yaw_median']:.4f}",
                    f"{overall['pitch_median']:.4f}",
                    f"{overall['roll_median']:.4f}",
                    f"{overall['median']:.4f}",
                    f"{overall['tx']:.4f}",
                    f"{overall['ty']:.4f}",
                    f"{overall['tz']:.4f}",
                    f"{overall['l2']:.4f}",
                    f"{overall['tx_median']:.4f}",
                    f"{overall['ty_median']:.4f}",
                    f"{overall['tz_median']:.4f}",
                    f"{overall['l2_median']:.4f}",
                    f"{overall['tx_bias']:.4f}",
                    f"{overall['ty_bias']:.4f}",
                    f"{overall['tz_bias']:.4f}",
                    f"{overall['tx_bias_median']:.4f}",
                    f"{overall['ty_bias_median']:.4f}",
                    f"{overall['tz_bias_median']:.4f}",
                ]
            )


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "6DoF Face BIWI evaluation requires CUDA. "
            "The upstream implementation hardcodes CUDA tensors, so run this script on a GPU node."
        )

    _validate_paths(args)

    repo_root = os.path.abspath(args.sixdof_face_repo)
    data_root = os.path.abspath(_prepare_data_root_if_needed(args))
    checkpoints_dir = os.path.abspath(_prepare_runtime_checkpoints_dir(args))

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)
    orig_resnet18 = None
    orig_solvepnp_ransac = None
    model_repository_module = None

    try:
        os.chdir(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from options.test_options import TestOptions
        from models import create_model

        if "data.augmentation" not in sys.modules:
            augmentation_stub = types.ModuleType("data.augmentation")

            class _UnusedAugmentor:
                def __init__(self, *args, **kwargs):
                    pass

            augmentation_stub.EulerAugmentor = _UnusedAugmentor
            augmentation_stub.HorizontalFlipAugmentor = _UnusedAugmentor
            sys.modules["data.augmentation"] = augmentation_stub

        import models.model_repository as model_repository_module

        orig_resnet18 = model_repository_module.resnet18

        def _offline_resnet18(*args, **kwargs):
            kwargs["pretrained"] = False
            return orig_resnet18(*args, **kwargs)

        model_repository_module.resnet18 = _offline_resnet18

        import data.biwi_dataset as biwi_dataset_module

        biwi_dataset_module.data_root = Path(data_root)
        BIWIDataset = _build_patched_biwi_dataset_class(biwi_dataset_module)

        orig_solvepnp_ransac = cv2.solvePnPRansac
        cv2.solvePnPRansac = _build_safe_solvepnp_ransac(orig_solvepnp_ransac)

        opt = TestOptions(
            cmd_line=_build_test_options(args, checkpoints_dir)
        ).parse()
        dataset = BIWIDataset(opt)

        if args.test_subjects:
            subject_ids = {int(s) for s in args.test_subjects}
            dataset.df = dataset.df[
                dataset.df["subject_id"].astype(int).isin(subject_ids)
            ].reset_index(drop=True)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
            drop_last=False,
        )

        logging.info("Evaluating %d BIWI frames with 6DoF Face", len(dataset))

        model = create_model(opt)
        per_subject_acc = defaultdict(_empty_acc)
        did_init = False
        global_tx_bias_mm = []
        global_ty_bias_mm = []
        global_tz_bias_mm = []

        for batch_idx, data in enumerate(dataloader):
            if batch_idx % 20 == 0:
                logging.info("[6DoF Face] batch %d/%d", batch_idx + 1, len(dataloader))

            if not did_init:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                model.eval()
                model.init_evaluation()
                did_init = True

            model.set_input(data)
            start_idx = len(model.inference_data["yaw_mae"])
            model.inference_curr_batch()
            batch_bias_mm = _decode_batch_translation_bias_mm(model)
            end_idx = len(model.inference_data["yaw_mae"])
            new_count = end_idx - start_idx

            batch_paths = list(model.img_path_list)[:new_count]
            for local_idx in range(new_count):
                subj = _parse_subject_from_img_path(batch_paths[local_idx])
                yaw = float(model.inference_data["yaw_mae"][start_idx + local_idx])
                pitch = float(model.inference_data["pitch_mae"][start_idx + local_idx])
                roll = float(model.inference_data["roll_mae"][start_idx + local_idx])
                tx_m = float(model.inference_data["tx_mae"][start_idx + local_idx])
                ty_m = float(model.inference_data["ty_mae"][start_idx + local_idx])
                tz_m = float(model.inference_data["tz_mae"][start_idx + local_idx])
                tx_bias_mm = float(batch_bias_mm[local_idx][0])
                ty_bias_mm = float(batch_bias_mm[local_idx][1])
                tz_bias_mm = float(batch_bias_mm[local_idx][2])

                acc = per_subject_acc[subj]
                acc["yaw"].append(yaw)
                acc["pitch"].append(pitch)
                acc["roll"].append(roll)
                acc["tx"].append(tx_m * 1000.0)
                acc["ty"].append(ty_m * 1000.0)
                acc["tz"].append(tz_m * 1000.0)
                acc["tx_bias"].append(tx_bias_mm)
                acc["ty_bias"].append(ty_bias_mm)
                acc["tz_bias"].append(tz_bias_mm)
                global_tx_bias_mm.append(tx_bias_mm)
                global_ty_bias_mm.append(ty_bias_mm)
                global_tz_bias_mm.append(tz_bias_mm)

        yaw_arr = np.asarray(model.inference_data["yaw_mae"], dtype=np.float64)
        pitch_arr = np.asarray(model.inference_data["pitch_mae"], dtype=np.float64)
        roll_arr = np.asarray(model.inference_data["roll_mae"], dtype=np.float64)
        tx_arr_m = np.asarray(model.inference_data["tx_mae"], dtype=np.float64)
        ty_arr_m = np.asarray(model.inference_data["ty_mae"], dtype=np.float64)
        tz_arr_m = np.asarray(model.inference_data["tz_mae"], dtype=np.float64)

        overall = _summarize_metrics(
            yaw_arr=yaw_arr,
            pitch_arr=pitch_arr,
            roll_arr=roll_arr,
            tx_arr_mm=tx_arr_m * 1000.0,
            ty_arr_mm=ty_arr_m * 1000.0,
            tz_arr_mm=tz_arr_m * 1000.0,
            tx_bias_arr_mm=global_tx_bias_mm,
            ty_bias_arr_mm=global_ty_bias_mm,
            tz_bias_arr_mm=global_tz_bias_mm,
        )

        per_subject = {}
        for subj, acc in sorted(per_subject_acc.items(), key=lambda item: item[0]):
            metrics = _summarize_metrics(
                yaw_arr=acc["yaw"],
                pitch_arr=acc["pitch"],
                roll_arr=acc["roll"],
                tx_arr_mm=acc["tx"],
                ty_arr_mm=acc["ty"],
                tz_arr_mm=acc["tz"],
                tx_bias_arr_mm=acc["tx_bias"],
                ty_bias_arr_mm=acc["ty_bias"],
                tz_bias_arr_mm=acc["tz_bias"],
            )
            if metrics is None:
                continue

            per_subject[subj] = metrics

            logging.info(
                "Subject %s (%d frames):\n"
                "  Mean      — Yaw: %.2f  Pitch: %.2f  Roll: %.2f  MAE: %.2f\n"
                "  Median    — Yaw: %.2f  Pitch: %.2f  Roll: %.2f  MAE: %.2f\n"
                "  TransMean — Tx: %.2f  Ty: %.2f  Tz: %.2f  L2: %.2f\n"
                "  TransMed  — Tx: %.2f  Ty: %.2f  Tz: %.2f  L2: %.2f\n"
                "  BiasMean  — Tx: %.2f  Ty: %.2f  Tz: %.2f\n"
                "  BiasMed   — Tx: %.2f  Ty: %.2f  Tz: %.2f",
                subj,
                metrics["count"],
                metrics["yaw"],
                metrics["pitch"],
                metrics["roll"],
                metrics["mean"],
                metrics["yaw_median"],
                metrics["pitch_median"],
                metrics["roll_median"],
                metrics["median"],
                metrics["tx"],
                metrics["ty"],
                metrics["tz"],
                metrics["l2"],
                metrics["tx_median"],
                metrics["ty_median"],
                metrics["tz_median"],
                metrics["l2_median"],
                metrics["tx_bias"],
                metrics["ty_bias"],
                metrics["tz_bias"],
                metrics["tx_bias_median"],
                metrics["ty_bias_median"],
                metrics["tz_bias_median"],
            )

        print("\n" + "=" * 60)
        print("6DoF FACE BIWI RESULTS")
        print("=" * 60)
        if overall is None:
            print("No BIWI frames evaluated.")
        else:
            print(f"Frames evaluated: {overall['count']}")
            print(
                f"\nMean      — Yaw: {overall['yaw']:.4f}  Pitch: {overall['pitch']:.4f}"
                f"  Roll: {overall['roll']:.4f}  MAE: {overall['mean']:.4f}"
            )
            print(
                f"Median    — Yaw: {overall['yaw_median']:.4f}  Pitch: {overall['pitch_median']:.4f}"
                f"  Roll: {overall['roll_median']:.4f}  MAE: {overall['median']:.4f}"
            )
            print(
                f"TransMean — Tx: {overall['tx']:.4f}  Ty: {overall['ty']:.4f}"
                f"  Tz: {overall['tz']:.4f}  L2: {overall['l2']:.4f}"
            )
            print(
                f"TransMed  — Tx: {overall['tx_median']:.4f}  Ty: {overall['ty_median']:.4f}"
                f"  Tz: {overall['tz_median']:.4f}  L2: {overall['l2_median']:.4f}"
            )
            print(
                f"BiasMean  — Tx: {overall['tx_bias']:.4f}  Ty: {overall['ty_bias']:.4f}"
                f"  Tz: {overall['tz_bias']:.4f}"
            )
            print(
                f"BiasMed   — Tx: {overall['tx_bias_median']:.4f}  Ty: {overall['ty_bias_median']:.4f}"
                f"  Tz: {overall['tz_bias_median']:.4f}"
            )
        print("=" * 60)

        _save_results_csv(args.output_csv, overall, per_subject)
        logging.info("Results saved -> %s", args.output_csv)

    finally:
        if model_repository_module is not None and orig_resnet18 is not None:
            model_repository_module.resnet18 = orig_resnet18
        if orig_solvepnp_ransac is not None:
            cv2.solvePnPRansac = orig_solvepnp_ransac
        os.chdir(old_cwd)
        sys.path = old_sys_path
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone BIWI benchmark for the native 6DoF Face method"
    )

    parser.add_argument(
        "--biwi_dir",
        default="data/biwi/faces_0",
        help="Raw BIWI root containing subject folders, pose.txt files, rgb.cal, and subject .obj meshes.",
    )
    parser.add_argument(
        "--test_subjects",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_SUBJECTS,
        help="BIWI subject IDs to evaluate.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to expose to 6DoF Face. The upstream model requires CUDA.",
    )

    parser.add_argument(
        "--sixdof_face_repo",
        default="third_party/6dof_face",
        help="Path to the 6DoF Face repository.",
    )
    parser.add_argument(
        "--sixdof_face_data_root",
        default="",
        help="Prepared 6DoF Face BIWI eval root containing csv/, image/, and info/.",
    )
    parser.add_argument(
        "--sixdof_face_checkpoints_dir",
        default="",
        help="Checkpoint root passed to 6DoF Face. Defaults to <repo>/checkpoint.",
    )
    parser.add_argument(
        "--sixdof_face_cache_root",
        default=os.path.join(SCRIPT_DIR, "cache", "sixdof_face_biwi_eval"),
        help="Where to auto-build the prepared BIWI cache if it is missing.",
    )
    parser.add_argument(
        "--prepare_sixdof_face_if_missing",
        action="store_true",
        help="Auto-build the 6DoF Face BIWI eval cache from --biwi_dir when needed.",
    )
    parser.add_argument(
        "--sixdof_face_name",
        default="run1",
        help="Experiment name inside the 6DoF Face checkpoint directory.",
    )
    parser.add_argument(
        "--sixdof_face_epoch",
        default="latest",
        help="Checkpoint epoch identifier, e.g. latest.",
    )
    parser.add_argument(
        "--sixdof_face_model",
        default="perspnet",
        help="6DoF Face model name.",
    )
    parser.add_argument(
        "--sixdof_face_img_size",
        type=int,
        default=192,
        help="Input image size expected by 6DoF Face.",
    )
    parser.add_argument(
        "--sixdof_face_batch_size",
        type=int,
        default=64,
        help="Batch size passed to the native 6DoF Face test loader.",
    )
    parser.add_argument(
        "--sixdof_face_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers used by the native 6DoF Face test loader.",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(SCRIPT_DIR, "sixdof_face_biwi_results.csv"),
        help="Where to save the per-subject and overall benchmark CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
