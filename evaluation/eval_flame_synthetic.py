"""Evaluate VGGT on the synthetic FLAME H2C validation split.

Usage examples:
    # Evaluate with LoRA checkpoint (relative H2C, 2-view pairs)
    python eval_flame_h2c.py \\
        --base_checkpoint checkpoints/VGGT-1B/model.pt \\
        --lora_checkpoint training/logs/flame_h2c_lora/ckpts/checkpoint.pt \\
        --pose_mode relative_h2c

    # Evaluate base model, absolute single-view
    python eval_flame_h2c.py \\
        --base_checkpoint checkpoints/VGGT-1B/model.pt --no_lora \\
        --pose_mode absolute_single --num_views 1

    # Custom dataset roots
    python eval_flame_h2c.py \\
        --root_dirs '[{"path":"/path/to/hair_v2","name":"hair"}]' \\
        --n_val_identities 10
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── LoRA target patterns (must match training) ───────────────────────────────
LORA_TARGET_PATTERNS = [
    "frame_blocks.*.attn.qkv",  "frame_blocks.*.attn.proj",
    "frame_blocks.*.mlp.fc1",   "frame_blocks.*.mlp.fc2",
    "global_blocks.*.attn.qkv", "global_blocks.*.attn.proj",
    "global_blocks.*.mlp.fc1",  "global_blocks.*.mlp.fc2",
]

DEFAULT_POSE_ENCODING_TYPE = "absT_quaR_FoV"


# ═════════════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_vggt_model(base_ckpt, lora_ckpt, device, enable_track=False, pose_encoding_type=DEFAULT_POSE_ENCODING_TYPE):
    from vggt.models.vggt import VGGT

    def load_compatible_state(model, state, label):
        model_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for key, value in state.items():
            target_value = model_state.get(key)
            if target_value is not None and target_value.shape != value.shape:
                skipped.append((key, tuple(value.shape), tuple(target_value.shape)))
                continue
            compatible_state[key] = value

        miss, unex = model.load_state_dict(compatible_state, strict=False)
        logging.info(f"  {label} -- missing: {len(miss)}, unexpected: {len(unex)}")
        if skipped:
            skipped_keys = ", ".join(key for key, _, _ in skipped[:4])
            logging.info(
                f"  {label} -- skipped {len(skipped)} shape-mismatched keys: {skipped_keys}"
            )
        return miss, unex

    lora_cfg = None
    if lora_ckpt is not None:
        lora_cfg = dict(
            enabled=True, rank=8, alpha=16, dropout=0.0,
            freeze_base=True, target_patterns=LORA_TARGET_PATTERNS,
        )

    model = VGGT(
        enable_camera=True,
        enable_depth=False,
        enable_point=False,
        enable_track=enable_track,
        pose_encoding_type=pose_encoding_type,
        lora=lora_cfg,
    )

    logging.info(f"Loading VGGT base: {base_ckpt}")
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    load_compatible_state(model, state, "Base")

    if lora_ckpt is not None:
        logging.info(f"Loading LoRA: {lora_ckpt}")
        lckpt = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
        lstate = lckpt.get("model", lckpt)
        load_compatible_state(model, lstate, "LoRA")
        if "epoch" in lckpt:
            logging.info(f"  Checkpoint epoch: {lckpt['epoch']}")

    model.eval().to(device)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 2. POSE DECODING
# ═════════════════════════════════════════════════════════════════════════════

def decode_pose_enc(pose_enc, image_size_hw=(518, 518), pose_encoding_type=DEFAULT_POSE_ENCODING_TYPE):
    """Decode VGGT pose_enc -> extrinsics (B, S, 3, 4) and intrinsics."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc.float(),
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=True,
    )
    return extrinsics, intrinsics


# ═════════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ═════════════════════════════════════════════════════════════════════════════

def rotation_matrix_to_euler_deg(R):
    """
    (N, 3, 3) rotation matrices -> (N, 3) euler angles in degrees.
    Convention: R = Rz @ Ry @ Rx  ->  [pitch (x), yaw (y), roll (z)].
    Uses the same decomposition as the BIWI evaluation for consistency.
    """
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = (sy < 1e-6).float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)

    pitch = (x * (1 - singular) + xs * singular) * (180.0 / np.pi)
    yaw = (y * (1 - singular) + ys * singular) * (180.0 / np.pi)
    roll = (z * (1 - singular)) * (180.0 / np.pi)

    return torch.stack([yaw, pitch, roll], dim=-1)  # (N, 3)


def angular_error(gt_deg, pred_deg):
    """Wrapped angular error (handles ±360/±180 ambiguity)."""
    candidates = torch.stack([
        torch.abs(gt_deg - pred_deg),
        torch.abs(pred_deg + 360 - gt_deg),
        torch.abs(pred_deg - 360 - gt_deg),
        torch.abs(pred_deg + 180 - gt_deg),
        torch.abs(pred_deg - 180 - gt_deg),
    ])
    return candidates.min(dim=0)[0]


def geodesic_rotation_error(R_pred, R_gt):
    """Geodesic rotation error in degrees: arccos((tr(R_pred^T @ R_gt) - 1) / 2)."""
    R_diff = R_pred.transpose(-1, -2) @ R_gt
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    return torch.acos(cos_angle) * (180.0 / np.pi)


def _empty_acc():
    return {
        "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "geodesic": 0.0,
        "tx": 0.0, "ty": 0.0, "tz": 0.0, "l2": 0.0,
        "count": 0,
    }


def accumulate_errors(acc, euler_err, geo_err, t_err, t_l2):
    """Add batch errors to accumulator."""
    n = euler_err.shape[0]
    acc["yaw"] += euler_err[:, 0].sum().item()
    acc["pitch"] += euler_err[:, 1].sum().item()
    acc["roll"] += euler_err[:, 2].sum().item()
    acc["geodesic"] += geo_err.sum().item()
    acc["tx"] += t_err[:, 0].sum().item()
    acc["ty"] += t_err[:, 1].sum().item()
    acc["tz"] += t_err[:, 2].sum().item()
    acc["l2"] += t_l2.sum().item()
    acc["count"] += n


def metrics_from_acc(acc):
    """Compute mean metrics from accumulator."""
    c = max(acc["count"], 1)
    return {
        "yaw": acc["yaw"] / c,
        "pitch": acc["pitch"] / c,
        "roll": acc["roll"] / c,
        "mean_euler": (acc["yaw"] + acc["pitch"] + acc["roll"]) / (3 * c),
        "geodesic": acc["geodesic"] / c,
        "tx": acc["tx"] / c,
        "ty": acc["ty"] / c,
        "tz": acc["tz"] / c,
        "l2": acc["l2"] / c,
        "count": acc["count"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. BATCH PROCESSING 
# ═════════════════════════════════════════════════════════════════════════════

def normalise_batch(batch, scale_by_points=True):
    """
    Apply the same extrinsic normalisation used during training.
    This makes the first frame the identity and optionally scales by point cloud extent.
    """
    training_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from train_utils.normalization import normalize_camera_extrinsics_and_points_batch

    norm_ext, norm_cam, norm_world, norm_depth = normalize_camera_extrinsics_and_points_batch(
        extrinsics=batch["extrinsics"],
        cam_points=batch["cam_points"],
        world_points=batch["world_points"],
        depths=batch["depths"],
        point_masks=batch["point_masks"],
        scale_by_points=scale_by_points,
    )
    batch["extrinsics"] = norm_ext
    batch["cam_points"] = norm_cam
    batch["world_points"] = norm_world
    batch["depths"] = norm_depth
    return batch


def should_apply_normalisation(args):
    """
    Match the synthetic eval default to the training setup:
      - relative_h2c      -> normalise to the first frame
      - absolute_* modes  -> keep absolute poses
    """
    if getattr(args, "force_normalise", False):
        return True
    if args.no_normalise:
        return False
    return args.pose_mode == "relative_h2c"


def should_scale_by_points(args, apply_normalisation):
    """
    FLAME H2C training configs disable point-scale normalisation by default.
    Keep this opt-in for eval so translation can be reported in metric units.
    """
    if not apply_normalisation:
        return False
    if args.no_scale_by_points:
        return False
    return args.scale_by_points


def _parse_root_dirs_arg(args):
    if args.root_dirs:
        return json.loads(args.root_dirs) if isinstance(args.root_dirs, str) else args.root_dirs
    return None


def _make_common_conf(args):
    class _Conf:
        pass

    common_conf = _Conf()
    common_conf.debug = False
    common_conf.training = False
    common_conf.img_size = args.img_size
    common_conf.patch_size = 14
    _augs = _Conf()
    _augs.scales = [1.0, 1.0]
    common_conf.augs = _augs
    common_conf.rescale = True
    common_conf.rescale_aug = False
    common_conf.landscape_check = True
    return common_conf


def _load_explicit_view_sample(ds, view_infos, target_image_shape):
    from data.dataset_util import read_image_cv2

    images, depths, cam_points, world_points = [], [], [], []
    point_masks, extrinsics, intrinsics, original_sizes = [], [], [], []

    for view_info in view_infos:
        view_dir = view_info["path"]
        image_path = os.path.join(view_dir, "output.png")
        image = read_image_cv2(image_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        R_h2c, t_h2c = ds._get_cached_h2c(view_info)
        intri = ds._load_intrinsics(view_dir)
        depth_map = np.ones(image.shape[:2], dtype=np.float32)

        if ds.face_crop:
            xmin, xmax, ymin, ymax = ds._load_face_bbox(view_dir)
            image, depth_map, intri = ds._face_crop(
                image, depth_map, intri, xmin, xmax, ymin, ymax
            )

        original_size = np.array(image.shape[:2])
        extri_h2c = np.zeros((3, 4), dtype=np.float64)
        extri_h2c[:3, :3] = R_h2c
        extri_h2c[:3, 3] = t_h2c

        (
            image,
            depth_map,
            extri_h2c,
            intri,
            world_coords_points,
            cam_coords_points,
            point_mask,
            _,
        ) = ds.process_one_image(
            image,
            depth_map,
            extri_h2c,
            intri,
            original_size,
            target_image_shape,
            filepath=image_path,
        )

        images.append(image)
        depths.append(depth_map)
        extrinsics.append(extri_h2c)
        intrinsics.append(intri)
        cam_points.append(cam_coords_points)
        world_points.append(world_coords_points)
        point_masks.append(point_mask)
        original_sizes.append(original_size)

    seq_root = view_infos[0]["root_name"]
    seq_identity = view_infos[0]["identity"][1]
    seq_lighting = view_infos[0]["lighting"]
    seq_name = f"flame_h2c_{seq_root}_{seq_identity}_{seq_lighting}"

    return {
        "seq_name": seq_name,
        "ids": np.arange(len(view_infos)),
        "frame_num": len(view_infos),
        "images": images,
        "depths": depths,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "cam_points": cam_points,
        "world_points": world_points,
        "point_masks": point_masks,
        "original_sizes": original_sizes,
        "tracks": None,
        "track_masks": None,
        "view_paths": [v["path"] for v in view_infos],
        "mesh_paths": [os.path.join(v["path"], "output_mesh.obj") for v in view_infos],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATASET CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def build_dataset(args):
    training_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)
    from data.datasets.flame_h2c import FlameH2CDataset

    root_dirs = _parse_root_dirs_arg(args)
    common_conf = _make_common_conf(args)

    inner_dataset = FlameH2CDataset(
        common_conf=common_conf,
        split="test",
        root_dirs=root_dirs,
        len_test=args.num_samples,
        num_views=args.num_views,
        same_identity=True,
        same_lighting=True,
        same_expression=False,
        n_val_identities_per_root=args.n_val_identities,
        face_crop=not args.no_face_crop,
        ad_min=args.ad_min,
        ad_max=args.ad_max,
        color_aug=False,
    )

    if args.pair_csv:
        with open(args.pair_csv, newline="") as f:
            pair_rows = list(csv.DictReader(f))
        target_image_shape = inner_dataset.get_target_shape(1.0)

        class _PairWrapper(torch.utils.data.Dataset):
            def __init__(self, ds, rows, pose_mode):
                self.ds = ds
                self.rows = rows
                self.pose_mode = pose_mode

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                row = self.rows[idx]

                def _view_info(prefix):
                    return {
                        "path": row[f"{prefix}_path"],
                        "root_name": row["root_name"],
                        "identity": (row["root_name"], row["identity_name"]),
                        "lighting": row["lighting"],
                        "expression": row[f"{prefix}_expression"],
                        "view": row[f"{prefix}_view"],
                    }

                anchor_view = _view_info("anchor")
                query_view = _view_info("query")
                view_infos = [query_view] if self.pose_mode == "absolute_single" else [anchor_view, query_view]
                batch = _load_explicit_view_sample(self.ds, view_infos, target_image_shape)

                images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
                images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)

                return {
                    "seq_name": batch["seq_name"],
                    "images": images,
                    "depths": torch.from_numpy(np.stack(batch["depths"]).astype(np.float32)),
                    "extrinsics": torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32)),
                    "intrinsics": torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32)),
                    "cam_points": torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32)),
                    "world_points": torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32)),
                    "point_masks": torch.from_numpy(np.stack(batch["point_masks"])),
                }

            def __getattr__(self, name):
                return getattr(self.ds, name)

        dataset = _PairWrapper(inner_dataset, pair_rows, args.pose_mode)
        return dataset, common_conf

    class _EvalWrapper(torch.utils.data.Dataset):
        def __init__(self, ds, num_views):
            self.ds = ds
            self.num_views = num_views

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            # BaseDataset.__getitem__ expects (seq_index, img_per_seq, aspect_ratio)
            batch = self.ds[(idx, self.num_views, 1.0)]

            # ── Replicate ComposedDataset tensor conversion ──
            images = torch.from_numpy(
                np.stack(batch["images"]).astype(np.float32)
            ).contiguous()
            # (S, H, W, C) -> (S, C, H, W), normalise [0,255] -> [0,1]
            images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)

            depths = torch.from_numpy(np.stack(batch["depths"]).astype(np.float32))
            extrinsics = torch.from_numpy(np.stack(batch["extrinsics"]).astype(np.float32))
            intrinsics = torch.from_numpy(np.stack(batch["intrinsics"]).astype(np.float32))
            cam_points = torch.from_numpy(np.stack(batch["cam_points"]).astype(np.float32))
            world_points = torch.from_numpy(np.stack(batch["world_points"]).astype(np.float32))
            point_masks = torch.from_numpy(np.stack(batch["point_masks"]))

            sample = {
                "seq_name": batch["seq_name"],
                "images": images,
                "depths": depths,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "cam_points": cam_points,
                "world_points": world_points,
                "point_masks": point_masks,
            }
            return sample

        # Forward attribute access for identity_list, all_views, etc.
        def __getattr__(self, name):
            return getattr(self.ds, name)

    dataset = _EvalWrapper(inner_dataset, args.num_views)
    return dataset, common_conf


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    apply_normalisation = should_apply_normalisation(args)
    scale_by_points = should_scale_by_points(args, apply_normalisation)

    if scale_by_points and args.translation_scale != 1.0:
        logging.warning(
            "Point-scale normalisation is enabled, so translation is not metric. "
            "Ignoring --translation_scale=%s for reporting.",
            args.translation_scale,
        )
    translation_scale = args.translation_scale if not scale_by_points else 1.0
    translation_unit = args.translation_unit if not scale_by_points else "scaled_units"

    logging.info(f"Device: {device}  |  AMP: {amp_dtype}")
    logging.info(f"Pose mode: {args.pose_mode}")
    logging.info(f"Pose encoding: {args.pose_encoding_type}")
    logging.info(f"Num views: {args.num_views}")
    logging.info(f"Normalise: {apply_normalisation}")
    logging.info(f"Scale by points: {scale_by_points}")
    logging.info(
        "Translation reporting: x%.4f -> %s",
        translation_scale,
        translation_unit,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    lora_path = None if args.no_lora else args.lora_checkpoint
    model = load_vggt_model(
        args.base_checkpoint,
        lora_path,
        device,
        enable_track=args.enable_track,
        pose_encoding_type=args.pose_encoding_type,
    )

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset, common_conf = build_dataset(args)
    logging.info(f"Val dataset: {len(dataset)} samples, {len(dataset.identity_list)} identities")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    # ── Determine which frames to evaluate ────────────────────────────────────
    if args.pose_mode == "relative_h2c":
        eval_frame_idx = 1  # target frame in a 2-view pair
    elif args.pose_mode == "absolute_second":
        eval_frame_idx = 1
    elif args.pose_mode == "absolute_single":
        eval_frame_idx = 0
    else:
        raise ValueError(f"Unknown pose_mode: {args.pose_mode}")

    # ── Accumulators ──────────────────────────────────────────────────────────
    overall_acc = _empty_acc()
    per_root_acc = defaultdict(_empty_acc)
    n_batches = len(loader)

    logging.info(f"Running evaluation: {n_batches} batches\n")

    for b_idx, batch in enumerate(loader):
        if b_idx % 20 == 0:
            logging.info(f"  batch {b_idx + 1}/{n_batches}")

        # ── Normalise ──────────────────────────────────────
        if apply_normalisation:
            with torch.no_grad():
                batch = normalise_batch(batch, scale_by_points=scale_by_points)

        # ── GT extrinsics ─────────────────────────────────────────────────────
        gt_ext = batch["extrinsics"]  # (B, S, 3, 4)
        R_gt = gt_ext[:, eval_frame_idx, :3, :3]  # (B, 3, 3)
        t_gt = gt_ext[:, eval_frame_idx, :3, 3]   # (B, 3)

        # ── Forward pass ──────────────────────────────────────────────────────
        images = batch["images"].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
            out = model(images=images)

        pose_enc = out["pose_enc"].cpu().float()
        pred_ext, _ = decode_pose_enc(
            pose_enc,
            image_size_hw=(args.img_size, args.img_size),
            pose_encoding_type=args.pose_encoding_type,
        )

        R_pred = pred_ext[:, eval_frame_idx, :3, :3]  # (B, 3, 3)
        t_pred = pred_ext[:, eval_frame_idx, :3, 3]   # (B, 3)

        # ── Rotation errors ───────────────────────────────────────────────────
        euler_gt = rotation_matrix_to_euler_deg(R_gt)
        euler_pred = rotation_matrix_to_euler_deg(R_pred)
        euler_err = angular_error(euler_gt, euler_pred)  # (B, 3)
        geo_err = geodesic_rotation_error(R_pred, R_gt)  # (B,)

        # ── Translation errors ────────────────────────────────────────────────
        t_err = torch.abs(t_pred - t_gt) * translation_scale  # (B, 3)
        t_l2 = torch.norm(t_pred - t_gt, dim=-1) * translation_scale  # (B,)

        # ── Accumulate ────────────────────────────────────────────────────────
        accumulate_errors(overall_acc, euler_err, geo_err, t_err, t_l2)

        # Per-root accumulation (if view_paths are available)
        if "seq_name" in batch:
            seq_names = batch["seq_name"]
            if isinstance(seq_names, (list, tuple)):
                for i, sn in enumerate(seq_names):
                    # seq_name format: "flame_h2c_{root_name}_{identity}_{lighting}"
                    parts = sn.split("_")
                    root_name = parts[2] if len(parts) > 2 else "unknown"
                    acc = per_root_acc[root_name]
                    accumulate_errors(
                        acc,
                        euler_err[i:i+1], geo_err[i:i+1],
                        t_err[i:i+1], t_l2[i:i+1],
                    )

    # ── Results ───────────────────────────────────────────────────────────────
    overall = metrics_from_acc(overall_acc)

    print("\n" + "=" * 70)
    print("FLAME H2C SYNTHETIC EVALUATION RESULTS")
    print("=" * 70)
    print(f"Samples evaluated: {overall['count']}")
    print(f"Pose mode: {args.pose_mode}  |  Eval frame: {eval_frame_idx}")
    print(f"Normalisation: {'ON' if apply_normalisation else 'OFF'}")
    print(f"Point-scale normalisation: {'ON' if scale_by_points else 'OFF'}")
    print()
    print("Rotation:")
    print(f"  Yaw:  {overall['yaw']:.4f}°    Pitch: {overall['pitch']:.4f}°    Roll: {overall['roll']:.4f}°")
    print(f"  Mean Euler MAE: {overall['mean_euler']:.4f}°")
    print(f"  Geodesic MAE:   {overall['geodesic']:.4f}°")
    print()
    print(f"Translation ({translation_unit}):")
    print(f"  Tx: {overall['tx']:.6f}    Ty: {overall['ty']:.6f}    Tz: {overall['tz']:.6f}")
    print(f"  L2: {overall['l2']:.6f}")

    if per_root_acc:
        print(f"\n{'─' * 70}")
        print("Per-root breakdown:")
        print(f"{'─' * 70}")
        for root_name, acc in sorted(per_root_acc.items()):
            m = metrics_from_acc(acc)
            print(
                f"  {root_name:15s} ({m['count']:5d} samples) — "
                f"Euler: {m['mean_euler']:.4f}°  Geodesic: {m['geodesic']:.4f}°  "
                f"Trans L2: {m['l2']:.6f} {translation_unit}"
            )

    print("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "flame_h2c_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "scope", "count",
            "yaw_mae", "pitch_mae", "roll_mae", "mean_euler_mae", "geodesic_mae",
            f"tx_mae_{translation_unit}", f"ty_mae_{translation_unit}",
            f"tz_mae_{translation_unit}", f"l2_mae_{translation_unit}",
        ]
        w.writerow(header)

        def _row(scope, m):
            return [
                scope, m["count"],
                f"{m['yaw']:.6f}", f"{m['pitch']:.6f}", f"{m['roll']:.6f}",
                f"{m['mean_euler']:.6f}", f"{m['geodesic']:.6f}",
                f"{m['tx']:.6f}", f"{m['ty']:.6f}", f"{m['tz']:.6f}",
                f"{m['l2']:.6f}",
            ]

        w.writerow(_row("overall", overall))
        for root_name, acc in sorted(per_root_acc.items()):
            w.writerow(_row(f"root_{root_name}", metrics_from_acc(acc)))

    logging.info(f"Results saved -> {csv_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()
    return overall


# ═════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VGGT on synthetic FLAME H2C validation data")

    # Model
    p.add_argument("--base_checkpoint", default=os.path.join(SCRIPT_DIR, "checkpoints", "VGGT-1B", "model.pt"))
    p.add_argument(
        "--lora_checkpoint",
        default=os.path.join(SCRIPT_DIR, "training", "logs", "flame_h2c_lora", "ckpts", "checkpoint.pt"),
    )
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--enable_track", action="store_true")
    p.add_argument(
        "--pose_mode",
        choices=["relative_h2c", "absolute_second", "absolute_single"],
        default="relative_h2c",
    )
    p.add_argument(
        "--pose_encoding_type",
        default=DEFAULT_POSE_ENCODING_TYPE,
        help="VGGT pose encoding used by the checkpoint (for example absT_quaR_FoV or absT_quaR).",
    )

    # Dataset
    p.add_argument(
        "--root_dirs", type=str, default=None,
        help='JSON list of {"path": ..., "name": ...} dicts. Uses dataset defaults if omitted.',
    )
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=5000, help="Number of val samples to evaluate")
    p.add_argument("--n_val_identities", type=int, default=5, help="Val identities per root (must match training)")
    p.add_argument(
        "--pair_csv",
        type=str,
        default=None,
        help="Optional explicit FLAME anchor/query pair CSV. If set, evaluate exactly these pairs instead of random val sampling.",
    )
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--no_face_crop", action="store_true")
    p.add_argument("--ad_min", type=float, default=1.0, help="Face crop margin min (val uses midpoint)")
    p.add_argument("--ad_max", type=float, default=1.0, help="Face crop margin max")
    p.add_argument(
        "--no_normalise",
        action="store_true",
        help="Skip training-style extrinsic normalisation. By default, only relative_h2c is normalised.",
    )
    p.add_argument(
        "--force_normalise",
        action="store_true",
        help="Force first-frame normalisation even for absolute pose modes.",
    )
    p.add_argument(
        "--scale_by_points",
        action="store_true",
        help="Enable point-cloud scale normalisation. OFF by default to match FLAME H2C training configs.",
    )
    p.add_argument(
        "--no_scale_by_points",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--translation_scale",
        type=float,
        default=1000.0,
        help="Multiplier applied to translation errors when point-scale normalisation is OFF. "
             "Use 1000 to report mm if raw FLAME poses are in meters.",
    )
    p.add_argument(
        "--translation_unit",
        type=str,
        default="mm",
        help="Label used when printing translation metrics.",
    )

    # Runtime
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default=os.path.join(SCRIPT_DIR, "eval_flame_h2c_output"))

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
