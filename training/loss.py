# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from functools import lru_cache
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from vggt.utils.rotation import euler_to_mat
from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss
    """
    def __init__(self, camera=None, depth=None, point=None, track=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.track = track

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)

        # Tracking loss - optional auxiliary task driven by first-frame query points
        if self.track is not None and ("track_list" in predictions or "track" in predictions):
            track_loss_dict = compute_track_loss(predictions, batch, **self.track)
            track_loss = track_loss_dict["loss_track"] * self.track["weight"]
            total_loss = total_loss + track_loss
            loss_dict.update(track_loss_dict)
        
        loss_dict["objective"] = total_loss
        loss_dict["loss_objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_trans_norm=0.0,  # weight for normalized translation-vector loss
    weight_trans_dir=0.0,   # weight for translation direction loss
    weight_trans_log_norm=0.0,  # weight for translation log-norm loss
    weight_reproj=0.0,      # weight for reprojection loss on sparse mesh points
    weight_decoded_norm=0.0,  # weight for decoded absolute normalized query translation loss
    weight_decoded_projective=0.0,  # weight for decoded relative projective translation loss
    weight_decoded_head_reproj=0.0,  # weight for decoded head-center reprojection loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    use_focal_ratio=False,  # if True, supervise on focal ratio (frame1/frame0) instead of absolute FoV
    supervise_frame_idxs=None,  # if set, supervise only these frame indices
    pose_supervise_frame_idxs=None,  # if set, supervise translation/rotation only on these frame indices
    translation_target_type="relative_se3",  # "relative_se3", "delta_t", "depth_norm_delta_t", or "projective_delta_t"
    absolute_anchor_supervision=False,  # if True, frame 0 is supervised as absolute H2C anchor pose
    absolute_anchor_frame_idx=0,
    absolute_anchor_translation_target_type="norm_log_z",  # "norm_log_z" or "raw"
    absolute_pose_supervision=False,  # if True, supervise pred_dict['pose_enc_abs_list'] as absolute H2C pose
    absolute_supervise_frame_idxs=None,
    absolute_translation_target_type="norm_log_z",  # "norm_log_z" or "raw"
    weight_abs_trans=0.0,
    weight_abs_rot=0.0,
    weight_abs_focal=0.0,
    trans_aux_loss_type="smooth_l1",
    trans_aux_beta=0.1,
    trans_aux_min_norm=1e-4,
    trans_aux_eps=1e-6,
    decoded_aux_loss_type="smooth_l1",
    decoded_aux_beta=0.1,
    decoded_aux_frame_idxs=None,
    decoded_head_reproj_loss_type="smooth_l1",
    decoded_head_reproj_beta=5.0,
    decoded_head_reproj_min_depth=1e-4,
    rotation_loss_type="encoding",
    rotation_geodesic_eps=1e-7,
    reproj_loss_type="smooth_l1",
    reproj_beta=5.0,
    reproj_min_depth=1e-4,
    reproj_num_points=256,
    reproj_frame_idxs=None,
    reproj_use_gt_intrinsics=True,
    min_valid_points=100,
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points.
    valid_frame_mask = point_masks.sum(dim=[-1, -2]) > min_valid_points
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )
    gt_pose_encoding = apply_translation_target_type(
        gt_pose_encoding,
        batch_data,
        pose_encoding_type=pose_encoding_type,
        translation_target_type=translation_target_type,
    )
    gt_pose_encoding = apply_absolute_anchor_target(
        gt_pose_encoding,
        batch_data,
        image_hw=image_hw,
        pose_encoding_type=pose_encoding_type,
        enabled=absolute_anchor_supervision,
        frame_idx=absolute_anchor_frame_idx,
        translation_target_type=absolute_anchor_translation_target_type,
    )
    gt_pose_encoding_visual = None
    if pose_encoding_type in {"quatR", "absT_quaR"}:
        gt_pose_encoding_visual = extri_intri_to_pose_encoding(
            gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type="absT_quaR_FoV"
        )

    pose_frame_idxs = pose_supervise_frame_idxs if pose_supervise_frame_idxs is not None else supervise_frame_idxs

    gt_pose_for_loss = select_supervised_frames(gt_pose_encoding, pose_frame_idxs)
    camera_valid_frame_mask = select_supervised_frames(valid_frame_mask, pose_frame_idxs)
    gt_pose_for_focal = select_supervised_frames(gt_pose_encoding, supervise_frame_idxs)
    focal_valid_frame_mask = select_supervised_frames(valid_frame_mask, supervise_frame_idxs)

    absolute_pose_encodings = pred_dict.get("pose_enc_abs_list")
    use_absolute_pose_loss = (
        absolute_pose_supervision
        or weight_abs_trans > 0
        or weight_abs_rot > 0
        or weight_abs_focal > 0
    )
    if use_absolute_pose_loss:
        if absolute_pose_encodings is None:
            raise ValueError(
                "absolute_pose_supervision/weight_abs_* requires model output 'pose_enc_abs_list'. "
                "Enable model.enable_absolute_camera in the config."
            )
        gt_abs_pose_encoding = build_absolute_pose_target(
            batch_data,
            image_hw=image_hw,
            pose_encoding_type=pose_encoding_type,
            translation_target_type=absolute_translation_target_type,
            device=gt_pose_encoding.device,
            dtype=gt_pose_encoding.dtype,
        )
        gt_abs_pose_for_loss = select_supervised_frames(gt_abs_pose_encoding, absolute_supervise_frame_idxs)
        abs_valid_frame_mask = select_supervised_frames(valid_frame_mask, absolute_supervise_frame_idxs)
    else:
        gt_abs_pose_for_loss = None
        abs_valid_frame_mask = None

    # Initialize loss accumulators for translation, rotation, focal length
    zero_loss = pred_pose_encodings[-1].sum() * 0
    total_loss_T = total_loss_T_norm = total_loss_R = total_loss_FL = 0
    total_loss_abs_T = zero_loss.clone()
    total_loss_abs_R = zero_loss.clone()
    total_loss_abs_FL = zero_loss.clone()
    total_loss_T_dir = total_loss_T_log_norm = 0
    total_loss_reproj = 0
    total_loss_T_decoded_norm = 0
    total_loss_T_decoded_projective = 0
    total_loss_T_decoded_head_reproj = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]
        pred_pose_for_loss = select_supervised_frames(pred_pose_stage, pose_frame_idxs)
        pred_pose_valid, gt_pose_valid = gather_valid_camera_pose_pairs(
            pred_pose_for_loss,
            gt_pose_for_loss,
            camera_valid_frame_mask,
            use_focal_ratio=False,
        )
        pred_pose_for_focal = select_supervised_frames(pred_pose_stage, supervise_frame_idxs)
        pred_pose_focal_valid, gt_pose_focal_valid = gather_valid_camera_pose_pairs(
            pred_pose_for_focal,
            gt_pose_for_focal,
            focal_valid_frame_mask,
            use_focal_ratio=use_focal_ratio,
        )

        if pred_pose_valid is None and pred_pose_focal_valid is None:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
            loss_T_norm_stage = (pred_pose_stage * 0).mean()
            loss_T_dir_stage = (pred_pose_stage * 0).mean()
            loss_T_log_norm_stage = (pred_pose_stage * 0).mean()
            loss_reproj_stage = (pred_pose_stage * 0).mean()
        else:
            if pred_pose_valid is None:
                zero_ref = pred_pose_stage * 0
                loss_T_stage = zero_ref.mean()
                loss_R_stage = zero_ref.mean()
                loss_T_norm_stage = zero_ref.mean()
                loss_T_dir_stage = zero_ref.mean()
                loss_T_log_norm_stage = zero_ref.mean()
            else:
                loss_T_stage, loss_R_stage = camera_TR_loss_single(
                    pred_pose_valid.clone(),
                    gt_pose_valid.clone(),
                    loss_type=loss_type,
                    pose_encoding_type=pose_encoding_type,
                    rotation_loss_type=rotation_loss_type,
                    rotation_geodesic_eps=rotation_geodesic_eps,
                )
                if weight_trans_norm > 0:
                    loss_T_norm_stage = translation_normalized_loss_single(
                        pred_pose_valid[..., :3],
                        gt_pose_valid[..., :3],
                        loss_type=trans_aux_loss_type,
                        beta=trans_aux_beta,
                        min_norm=trans_aux_min_norm,
                    )
                else:
                    loss_T_norm_stage = loss_T_stage * 0
                if weight_trans_dir > 0 or weight_trans_log_norm > 0:
                    loss_T_dir_stage, loss_T_log_norm_stage = translation_aux_loss_single(
                        pred_pose_valid[..., :3],
                        gt_pose_valid[..., :3],
                        loss_type=trans_aux_loss_type,
                        beta=trans_aux_beta,
                        min_norm=trans_aux_min_norm,
                        eps=trans_aux_eps,
                    )
                else:
                    loss_T_dir_stage = loss_T_stage * 0
                    loss_T_log_norm_stage = loss_T_stage * 0

            if pred_pose_focal_valid is None:
                loss_FL_stage = (pred_pose_stage * 0).mean()
            else:
                loss_FL_stage = camera_focal_loss_single(
                    pred_pose_focal_valid.clone(),
                    gt_pose_focal_valid.clone(),
                    loss_type=loss_type,
                    use_focal_ratio=use_focal_ratio,
                    pose_encoding_type=pose_encoding_type,
                )
            if weight_reproj > 0 and "mesh_paths" in batch_data:
                loss_reproj_stage = camera_reprojection_loss(
                    pred_pose_stage=pred_pose_stage,
                    gt_extrinsics=gt_extrinsics,
                    gt_intrinsics=gt_intrinsics,
                    valid_frame_mask=valid_frame_mask,
                    mesh_paths=batch_data["mesh_paths"],
                    image_hw=image_hw,
                    pose_encoding_type=pose_encoding_type,
                    frame_idxs=reproj_frame_idxs if reproj_frame_idxs is not None else pose_frame_idxs,
                    loss_type=reproj_loss_type,
                    beta=reproj_beta,
                    min_depth=reproj_min_depth,
                    num_points=reproj_num_points,
                    use_gt_intrinsics=reproj_use_gt_intrinsics,
                )
            else:
                loss_reproj_stage = loss_T_stage * 0

        loss_T_decoded_norm_stage = loss_T_stage * 0
        loss_T_decoded_projective_stage = loss_T_stage * 0
        loss_T_decoded_head_reproj_stage = loss_T_stage * 0
        if weight_decoded_norm > 0 or weight_decoded_projective > 0 or weight_decoded_head_reproj > 0:
            (
                loss_T_decoded_norm_stage,
                loss_T_decoded_projective_stage,
                loss_T_decoded_head_reproj_stage,
            ) = decoded_translation_aux_losses(
                pred_pose_stage=pred_pose_stage,
                absolute_extrinsics=batch_data.get("absolute_extrinsics"),
                gt_intrinsics=gt_intrinsics,
                valid_frame_mask=valid_frame_mask,
                translation_target_type=translation_target_type,
                frame_idxs=decoded_aux_frame_idxs if decoded_aux_frame_idxs is not None else pose_frame_idxs,
                loss_type=decoded_aux_loss_type,
                beta=decoded_aux_beta,
                head_reproj_loss_type=decoded_head_reproj_loss_type,
                head_reproj_beta=decoded_head_reproj_beta,
                head_reproj_min_depth=decoded_head_reproj_min_depth,
                compute_norm=weight_decoded_norm > 0,
                compute_projective=weight_decoded_projective > 0,
                compute_head_reproj=weight_decoded_head_reproj > 0,
            )

        loss_abs_T_stage = loss_T_stage * 0
        loss_abs_R_stage = loss_R_stage * 0
        loss_abs_FL_stage = loss_FL_stage * 0
        if use_absolute_pose_loss:
            pred_abs_stage = absolute_pose_encodings[stage_idx]
            pred_abs_for_loss = select_supervised_frames(pred_abs_stage, absolute_supervise_frame_idxs)
            pred_abs_valid, gt_abs_valid = gather_valid_camera_pose_pairs(
                pred_abs_for_loss,
                gt_abs_pose_for_loss,
                abs_valid_frame_mask,
                use_focal_ratio=False,
            )
            if pred_abs_valid is not None:
                loss_abs_T_stage, loss_abs_R_stage = camera_TR_loss_single(
                    pred_abs_valid.clone(),
                    gt_abs_valid.clone(),
                    loss_type=loss_type,
                    pose_encoding_type=pose_encoding_type,
                    rotation_loss_type=rotation_loss_type,
                    rotation_geodesic_eps=rotation_geodesic_eps,
                )
                loss_abs_FL_stage = camera_focal_loss_single(
                    pred_abs_valid.clone(),
                    gt_abs_valid.clone(),
                    loss_type=loss_type,
                    use_focal_ratio=False,
                    pose_encoding_type=pose_encoding_type,
                )

        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_T_norm += loss_T_norm_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight
        total_loss_abs_T += loss_abs_T_stage * stage_weight
        total_loss_abs_R += loss_abs_R_stage * stage_weight
        total_loss_abs_FL += loss_abs_FL_stage * stage_weight
        total_loss_T_dir += loss_T_dir_stage * stage_weight
        total_loss_T_log_norm += loss_T_log_norm_stage * stage_weight
        total_loss_reproj += loss_reproj_stage * stage_weight
        total_loss_T_decoded_norm += loss_T_decoded_norm_stage * stage_weight
        total_loss_T_decoded_projective += loss_T_decoded_projective_stage * stage_weight
        total_loss_T_decoded_head_reproj += loss_T_decoded_head_reproj_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_T_norm = total_loss_T_norm / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages
    avg_loss_abs_T = total_loss_abs_T / n_stages
    avg_loss_abs_R = total_loss_abs_R / n_stages
    avg_loss_abs_FL = total_loss_abs_FL / n_stages
    avg_loss_T_dir = total_loss_T_dir / n_stages
    avg_loss_T_log_norm = total_loss_T_log_norm / n_stages
    avg_loss_reproj = total_loss_reproj / n_stages
    avg_loss_T_decoded_norm = total_loss_T_decoded_norm / n_stages
    avg_loss_T_decoded_projective = total_loss_T_decoded_projective / n_stages
    avg_loss_T_decoded_head_reproj = total_loss_T_decoded_head_reproj / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_T_norm * weight_trans_norm +
        avg_loss_T_dir * weight_trans_dir +
        avg_loss_T_log_norm * weight_trans_log_norm +
        avg_loss_reproj * weight_reproj +
        avg_loss_T_decoded_norm * weight_decoded_norm +
        avg_loss_T_decoded_projective * weight_decoded_projective +
        avg_loss_T_decoded_head_reproj * weight_decoded_head_reproj +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal +
        avg_loss_abs_T * weight_abs_trans +
        avg_loss_abs_R * weight_abs_rot +
        avg_loss_abs_FL * weight_abs_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_T_norm": avg_loss_T_norm,
        "loss_T_dir": avg_loss_T_dir,
        "loss_T_log_norm": avg_loss_T_log_norm,
        "loss_reproj": avg_loss_reproj,
        "loss_T_decoded_norm": avg_loss_T_decoded_norm,
        "loss_T_decoded_projective": avg_loss_T_decoded_projective,
        "loss_T_decoded_head_reproj": avg_loss_T_decoded_head_reproj,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL,
        "loss_abs_T": avg_loss_abs_T,
        "loss_abs_R": avg_loss_abs_R,
        "loss_abs_FL": avg_loss_abs_FL,
        "pose_encoding": gt_pose_encoding,  # Loss-space encoding
        "pose_encoding_visual": gt_pose_encoding_visual,
    }


def apply_translation_target_type(
    gt_pose_encoding,
    batch_data,
    pose_encoding_type="absT_quaR_FoV",
    translation_target_type="relative_se3",
):
    """Optionally replace the pose-encoding translation slot with a decoupled target."""
    if translation_target_type == "relative_se3":
        return gt_pose_encoding

    if translation_target_type not in {"delta_t", "depth_norm_delta_t", "projective_delta_t"}:
        raise ValueError(
            "translation_target_type must be 'relative_se3', 'delta_t', 'depth_norm_delta_t', "
            "or 'projective_delta_t', "
            f"got {translation_target_type!r}"
        )

    if pose_encoding_type == "quatR":
        raise ValueError(f"translation_target_type={translation_target_type!r} requires a pose encoding with translation slots")

    if "absolute_extrinsics" not in batch_data:
        raise ValueError(
            f"translation_target_type={translation_target_type!r} requires batch_data['absolute_extrinsics']; "
            "the trainer must save pre-normalized extrinsics before first-frame normalization."
        )

    absolute_extrinsics = batch_data["absolute_extrinsics"].to(
        device=gt_pose_encoding.device,
        dtype=gt_pose_encoding.dtype,
    )
    if absolute_extrinsics.ndim != 4 or absolute_extrinsics.shape[-2:] != (3, 4):
        raise ValueError(
            "absolute_extrinsics must have shape [B, S, 3, 4], "
            f"got {tuple(absolute_extrinsics.shape)}"
        )
    if absolute_extrinsics.shape[:2] != gt_pose_encoding.shape[:2]:
        raise ValueError(
            "absolute_extrinsics and gt_pose_encoding must agree on [B, S], "
            f"got {tuple(absolute_extrinsics.shape[:2])} vs {tuple(gt_pose_encoding.shape[:2])}"
        )

    translations = absolute_extrinsics[:, :, :3, 3]
    anchor_t = translations[:, :1]
    delta_t = translations - anchor_t
    if translation_target_type in {"depth_norm_delta_t", "projective_delta_t"}:
        eps = torch.finfo(gt_pose_encoding.dtype).eps
        z0 = anchor_t[..., 2].clamp(min=eps)
        z = translations[..., 2].clamp(min=eps)
        if translation_target_type == "depth_norm_delta_t":
            delta_t = torch.stack(
                [
                    delta_t[..., 0] / z0,
                    delta_t[..., 1] / z0,
                    torch.log(z / z0),
                ],
                dim=-1,
            )
        else:
            delta_t = torch.stack(
                [
                    translations[..., 0] / z - anchor_t[..., 0] / z0,
                    translations[..., 1] / z - anchor_t[..., 1] / z0,
                    torch.log(z / z0),
                ],
                dim=-1,
            )

    gt_pose_encoding = gt_pose_encoding.clone()
    gt_pose_encoding[..., :3] = delta_t
    return gt_pose_encoding


def build_absolute_pose_target(
    batch_data,
    image_hw,
    pose_encoding_type="absT_quaR_FoV",
    translation_target_type="norm_log_z",
    device=None,
    dtype=None,
):
    """Build absolute H2C pose targets for the auxiliary absolute camera head."""
    if pose_encoding_type == "quatR":
        raise ValueError("Absolute pose supervision requires a pose encoding with translation slots")

    if "absolute_extrinsics" not in batch_data:
        raise ValueError(
            "Absolute pose supervision requires batch_data['absolute_extrinsics']; "
            "the trainer must save pre-normalized extrinsics before first-frame normalization."
        )

    absolute_extrinsics = batch_data["absolute_extrinsics"]
    gt_intrinsics = batch_data["intrinsics"]
    if device is None:
        device = gt_intrinsics.device
    if dtype is None:
        dtype = gt_intrinsics.dtype
    absolute_extrinsics = absolute_extrinsics.to(device=device, dtype=dtype)
    gt_intrinsics = gt_intrinsics.to(device=device, dtype=dtype)

    gt_abs_pose_encoding = extri_intri_to_pose_encoding(
        absolute_extrinsics,
        gt_intrinsics,
        image_hw,
        pose_encoding_type=pose_encoding_type,
    )

    if translation_target_type == "norm_log_z":
        eps = torch.finfo(dtype).eps
        translations = absolute_extrinsics[:, :, :3, 3]
        z = translations[..., 2].clamp(min=eps)
        gt_abs_pose_encoding = gt_abs_pose_encoding.clone()
        gt_abs_pose_encoding[..., :3] = torch.stack(
            [
                translations[..., 0] / z,
                translations[..., 1] / z,
                torch.log(z),
            ],
            dim=-1,
        )
    elif translation_target_type == "raw":
        pass
    else:
        raise ValueError(
            "absolute_translation_target_type must be 'norm_log_z' or 'raw', "
            f"got {translation_target_type!r}"
        )

    return gt_abs_pose_encoding


def apply_absolute_anchor_target(
    gt_pose_encoding,
    batch_data,
    image_hw,
    pose_encoding_type="absT_quaR_FoV",
    enabled=False,
    frame_idx=0,
    translation_target_type="norm_log_z",
):
    """Replace one frame target with an absolute H2C anchor target.

    This supports mixed-output training: frame 0 can predict an absolute anchor
    pose, while frame 1 keeps the relative/query target created above.
    """
    if not enabled:
        return gt_pose_encoding

    if pose_encoding_type == "quatR":
        raise ValueError("absolute_anchor_supervision requires a pose encoding with translation slots")

    if "absolute_extrinsics" not in batch_data:
        raise ValueError(
            "absolute_anchor_supervision requires batch_data['absolute_extrinsics']; "
            "the trainer must save pre-normalized extrinsics before first-frame normalization."
        )

    absolute_extrinsics = batch_data["absolute_extrinsics"].to(
        device=gt_pose_encoding.device,
        dtype=gt_pose_encoding.dtype,
    )
    gt_intrinsics = batch_data["intrinsics"].to(
        device=gt_pose_encoding.device,
        dtype=gt_pose_encoding.dtype,
    )
    if absolute_extrinsics.shape[:2] != gt_pose_encoding.shape[:2]:
        raise ValueError(
            "absolute_extrinsics and gt_pose_encoding must agree on [B, S], "
            f"got {tuple(absolute_extrinsics.shape[:2])} vs {tuple(gt_pose_encoding.shape[:2])}"
        )

    frame_idx = int(frame_idx)
    if frame_idx < 0:
        frame_idx += gt_pose_encoding.shape[1]
    if frame_idx < 0 or frame_idx >= gt_pose_encoding.shape[1]:
        raise ValueError(
            f"absolute_anchor_frame_idx={frame_idx} is outside sequence length {gt_pose_encoding.shape[1]}"
        )

    absolute_pose_encoding = extri_intri_to_pose_encoding(
        absolute_extrinsics,
        gt_intrinsics,
        image_hw,
        pose_encoding_type=pose_encoding_type,
    )

    if translation_target_type == "norm_log_z":
        eps = torch.finfo(gt_pose_encoding.dtype).eps
        translations = absolute_extrinsics[:, :, :3, 3]
        z = translations[..., 2].clamp(min=eps)
        absolute_t = torch.stack(
            [
                translations[..., 0] / z,
                translations[..., 1] / z,
                torch.log(z),
            ],
            dim=-1,
        )
        absolute_pose_encoding = absolute_pose_encoding.clone()
        absolute_pose_encoding[..., :3] = absolute_t
    elif translation_target_type == "raw":
        pass
    else:
        raise ValueError(
            "absolute_anchor_translation_target_type must be 'norm_log_z' or 'raw', "
            f"got {translation_target_type!r}"
        )

    gt_pose_encoding = gt_pose_encoding.clone()
    gt_pose_encoding[:, frame_idx] = absolute_pose_encoding[:, frame_idx]
    return gt_pose_encoding


def decode_translation_slot_to_absolute(
    pred_translation,
    absolute_extrinsics,
    translation_target_type="relative_se3",
    eps=1e-6,
):
    """Decode a decoupled translation prediction back to absolute H2C translation."""
    if absolute_extrinsics is None:
        raise ValueError(
            "Decoded translation auxiliary losses require batch_data['absolute_extrinsics']; "
            "enable them only for decoupled translation targets."
        )

    if translation_target_type == "relative_se3":
        raise ValueError(
            "Decoded translation auxiliary losses are only defined for decoupled targets "
            "('delta_t', 'depth_norm_delta_t', or 'projective_delta_t'), not relative_se3."
        )

    if translation_target_type not in {"delta_t", "depth_norm_delta_t", "projective_delta_t"}:
        raise ValueError(
            "translation_target_type must be 'delta_t', 'depth_norm_delta_t', or "
            f"'projective_delta_t', got {translation_target_type!r}"
        )

    absolute_extrinsics = absolute_extrinsics.to(device=pred_translation.device, dtype=pred_translation.dtype)
    gt_translation = absolute_extrinsics[:, :, :3, 3]
    anchor_t = gt_translation[:, :1]

    if translation_target_type == "delta_t":
        return anchor_t + pred_translation

    z0 = anchor_t[..., 2].clamp(min=eps)
    log_z_ratio = pred_translation[..., 2].clamp(min=-20.0, max=20.0)
    z = z0 * torch.exp(log_z_ratio)

    if translation_target_type == "depth_norm_delta_t":
        x = anchor_t[..., 0] + pred_translation[..., 0] * z0
        y = anchor_t[..., 1] + pred_translation[..., 1] * z0
    else:
        u0 = anchor_t[..., 0] / z0
        v0 = anchor_t[..., 1] / z0
        x = (u0 + pred_translation[..., 0]) * z
        y = (v0 + pred_translation[..., 1]) * z

    return torch.stack([x, y, z], dim=-1)


def absolute_normalized_translation(translation, eps=1e-6):
    z = translation[..., 2].clamp(min=eps)
    return torch.stack(
        [
            translation[..., 0] / z,
            translation[..., 1] / z,
            torch.log(z),
        ],
        dim=-1,
    )


def relative_projective_translation(translation, anchor_t, eps=1e-6):
    z = translation[..., 2].clamp(min=eps)
    z0 = anchor_t[..., 2].clamp(min=eps)
    return torch.stack(
        [
            translation[..., 0] / z - anchor_t[..., 0] / z0,
            translation[..., 1] / z - anchor_t[..., 1] / z0,
            torch.log(z / z0),
        ],
        dim=-1,
    )


def translation_representation_loss(
    pred_values,
    gt_values,
    valid_mask,
    loss_type="smooth_l1",
    beta=0.1,
    loss_name="loss_T_decoded",
):
    if valid_mask.sum() == 0:
        return (pred_values * 0).mean()

    pred_valid = pred_values[valid_mask]
    gt_valid = gt_values[valid_mask]

    if loss_type == "l1":
        loss = (pred_valid - gt_valid).abs().mean(dim=-1)
    elif loss_type == "l2":
        loss = (pred_valid - gt_valid).norm(dim=-1)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(
            pred_valid,
            gt_valid,
            beta=beta,
            reduction="none",
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unknown decoded translation auxiliary loss type: {loss_type}")

    loss = check_and_fix_inf_nan(loss, loss_name)
    return loss.mean()


def project_head_center(translation, intrinsics, min_depth=1e-4):
    z = translation[..., 2].clamp(min=min_depth)
    u = intrinsics[..., 0, 0] * (translation[..., 0] / z) + intrinsics[..., 0, 2]
    v = intrinsics[..., 1, 1] * (translation[..., 1] / z) + intrinsics[..., 1, 2]
    return torch.stack([u, v], dim=-1)


def decoded_translation_aux_losses(
    pred_pose_stage,
    absolute_extrinsics,
    gt_intrinsics,
    valid_frame_mask,
    translation_target_type="relative_se3",
    frame_idxs=None,
    loss_type="smooth_l1",
    beta=0.1,
    head_reproj_loss_type="smooth_l1",
    head_reproj_beta=5.0,
    head_reproj_min_depth=1e-4,
    compute_norm=False,
    compute_projective=False,
    compute_head_reproj=False,
):
    """Auxiliary losses after decoding a decoupled translation slot to absolute H2C."""
    pred_abs_t = decode_translation_slot_to_absolute(
        pred_pose_stage[..., :3],
        absolute_extrinsics,
        translation_target_type=translation_target_type,
    )
    absolute_extrinsics = absolute_extrinsics.to(device=pred_pose_stage.device, dtype=pred_pose_stage.dtype)
    gt_abs_t = absolute_extrinsics[:, :, :3, 3]

    pred_abs_t = select_supervised_frames(pred_abs_t, frame_idxs)
    gt_abs_t = select_supervised_frames(gt_abs_t, frame_idxs)
    valid_mask = select_supervised_frames(valid_frame_mask, frame_idxs)
    valid_mask = valid_mask & (gt_abs_t[..., 2] > head_reproj_min_depth)

    zero_loss = (pred_pose_stage * 0).mean()
    loss_decoded_norm = zero_loss
    loss_decoded_projective = zero_loss
    loss_decoded_head_reproj = zero_loss

    if compute_norm:
        loss_decoded_norm = translation_representation_loss(
            absolute_normalized_translation(pred_abs_t),
            absolute_normalized_translation(gt_abs_t),
            valid_mask,
            loss_type=loss_type,
            beta=beta,
            loss_name="loss_T_decoded_norm",
        )

    if compute_projective:
        anchor_t = absolute_extrinsics[:, :1, :3, 3]
        anchor_t = anchor_t.expand(-1, absolute_extrinsics.shape[1], -1)
        anchor_t = select_supervised_frames(anchor_t, frame_idxs)
        loss_decoded_projective = translation_representation_loss(
            relative_projective_translation(pred_abs_t, anchor_t),
            relative_projective_translation(gt_abs_t, anchor_t),
            valid_mask,
            loss_type=loss_type,
            beta=beta,
            loss_name="loss_T_decoded_projective",
        )

    if compute_head_reproj:
        gt_intrinsics = gt_intrinsics.to(device=pred_pose_stage.device, dtype=pred_pose_stage.dtype)
        intrinsics = select_supervised_frames(gt_intrinsics, frame_idxs)
        loss_decoded_head_reproj = translation_representation_loss(
            project_head_center(pred_abs_t, intrinsics, min_depth=head_reproj_min_depth),
            project_head_center(gt_abs_t, intrinsics, min_depth=head_reproj_min_depth),
            valid_mask,
            loss_type=head_reproj_loss_type,
            beta=head_reproj_beta,
            loss_name="loss_T_decoded_head_reproj",
        )

    return loss_decoded_norm, loss_decoded_projective, loss_decoded_head_reproj


def select_supervised_frames(data, supervise_frame_idxs=None):
    """Optionally keep only a subset of frames from a [B, S, ...] tensor."""
    if supervise_frame_idxs is None:
        return data

    if len(supervise_frame_idxs) == 0:
        raise ValueError("supervise_frame_idxs must not be empty")

    return data[:, supervise_frame_idxs]


def gather_valid_camera_pose_pairs(pred_pose_enc, gt_pose_enc, valid_frame_mask, use_focal_ratio=False):
    """
    Gather the valid pose entries used by the camera loss.

    For focal-ratio supervision we must keep the sequence dimension intact.
    For absolute supervision we can flatten valid frames independently.
    """
    if pred_pose_enc.shape[:2] != gt_pose_enc.shape[:2]:
        raise ValueError(
            f"Pred/GT pose shapes must agree on [B, S], got {pred_pose_enc.shape} vs {gt_pose_enc.shape}"
        )
    if pred_pose_enc.shape[:2] != valid_frame_mask.shape:
        raise ValueError(
            f"valid_frame_mask shape must match [B, S], got {valid_frame_mask.shape} vs {pred_pose_enc.shape[:2]}"
        )

    if use_focal_ratio and pred_pose_enc.shape[1] > 1:
        valid_sequence_mask = valid_frame_mask.all(dim=1)
        if valid_sequence_mask.sum() == 0:
            return None, None
        return pred_pose_enc[valid_sequence_mask], gt_pose_enc[valid_sequence_mask]

    if valid_frame_mask.sum() == 0:
        return None, None

    return pred_pose_enc[valid_frame_mask], gt_pose_enc[valid_frame_mask]


def select_frame_paths(paths, frame_idxs, batch_size, seq_len):
    """
    Keep selected frame indices from a nested path structure and return batch-major paths.
    """
    if paths is None:
        return None

    if isinstance(paths, tuple):
        paths = list(paths)

    if len(paths) == 0:
        return [[] for _ in range(batch_size)]

    if frame_idxs is None:
        frame_idxs = list(range(seq_len))
    else:
        frame_idxs = [int(idx) for idx in frame_idxs]

    if len(paths) == batch_size and all(isinstance(row, (list, tuple)) for row in paths):
        batch_major = [list(row) for row in paths]
    elif len(paths) == seq_len and all(isinstance(row, (list, tuple)) for row in paths):
        batch_major = [[paths[s][b] for s in range(seq_len)] for b in range(batch_size)]
    else:
        raise ValueError(
            f"Unsupported path layout for reprojection loss: batch_size={batch_size}, seq_len={seq_len}"
        )

    return [[str(batch_major[b][idx]) for idx in frame_idxs] for b in range(batch_size)]


@lru_cache(maxsize=8192)
def load_obj_vertices_subset(mesh_path: str, num_points: int):
    """
    Load a deterministic sparse subset of OBJ vertices for reprojection supervision.
    """
    vertices = []
    with open(mesh_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if len(vertices) == 0:
        raise ValueError(f"No vertices found in OBJ mesh: {mesh_path}")

    verts = np.asarray(vertices, dtype=np.float32)
    if num_points is not None and num_points > 0 and len(verts) > num_points:
        indices = np.linspace(0, len(verts) - 1, num_points, dtype=np.int64)
        verts = verts[indices]
    return verts


def project_points_with_extrinsics(points_obj, extrinsics, intrinsics, min_depth=1e-4):
    """
    Project object-space points with OpenCV-style camera-from-object extrinsics.
    """
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    cam_points = points_obj @ R.transpose(0, 1) + T.unsqueeze(0)

    z = cam_points[:, 2]
    z_safe = z.clamp(min=min_depth)
    u = intrinsics[0, 0] * (cam_points[:, 0] / z_safe) + intrinsics[0, 2]
    v = intrinsics[1, 1] * (cam_points[:, 1] / z_safe) + intrinsics[1, 2]
    return torch.stack([u, v], dim=-1), z > min_depth


def reprojection_loss_single(
    pred_extrinsics,
    gt_extrinsics,
    intrinsics,
    points_obj,
    loss_type="smooth_l1",
    beta=5.0,
    min_depth=1e-4,
):
    pred_uv, _ = project_points_with_extrinsics(
        points_obj, pred_extrinsics, intrinsics, min_depth=min_depth
    )
    gt_uv, gt_valid = project_points_with_extrinsics(
        points_obj, gt_extrinsics, intrinsics, min_depth=min_depth
    )

    if gt_valid.sum() == 0:
        return (pred_uv * 0).mean()

    pred_uv = pred_uv[gt_valid]
    gt_uv = gt_uv[gt_valid]

    if loss_type == "l1":
        reproj_loss = (pred_uv - gt_uv).abs().mean(dim=-1)
    elif loss_type == "l2":
        reproj_loss = (pred_uv - gt_uv).norm(dim=-1)
    elif loss_type == "smooth_l1":
        reproj_loss = F.smooth_l1_loss(
            pred_uv,
            gt_uv,
            beta=beta,
            reduction="none",
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unknown reprojection loss type: {loss_type}")

    reproj_loss = check_and_fix_inf_nan(reproj_loss, "loss_reproj", hard_max=1000)
    return reproj_loss.mean()


def camera_reprojection_loss(
    pred_pose_stage,
    gt_extrinsics,
    gt_intrinsics,
    valid_frame_mask,
    mesh_paths,
    image_hw,
    pose_encoding_type="absT_quaR_FoV",
    frame_idxs=None,
    loss_type="smooth_l1",
    beta=5.0,
    min_depth=1e-4,
    num_points=256,
    use_gt_intrinsics=True,
):
    """
    Reproject sparse mesh vertices with predicted and GT poses into the processed crop.
    """
    pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
        pred_pose_stage.float(),
        image_size_hw=image_hw,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=not use_gt_intrinsics,
    )

    pred_extrinsics = select_supervised_frames(pred_extrinsics, frame_idxs)
    gt_extrinsics = select_supervised_frames(gt_extrinsics, frame_idxs)
    valid_mask = select_supervised_frames(valid_frame_mask, frame_idxs)
    intrinsics = select_supervised_frames(gt_intrinsics, frame_idxs) if use_gt_intrinsics else select_supervised_frames(pred_intrinsics, frame_idxs)

    batch_size = pred_extrinsics.shape[0]
    seq_len = pred_pose_stage.shape[1]
    mesh_paths = select_frame_paths(mesh_paths, frame_idxs, batch_size, seq_len)

    losses = []
    device = pred_extrinsics.device
    dtype = pred_extrinsics.dtype

    for batch_idx in range(pred_extrinsics.shape[0]):
        for frame_idx in range(pred_extrinsics.shape[1]):
            if not bool(valid_mask[batch_idx, frame_idx].item()):
                continue

            mesh_path = mesh_paths[batch_idx][frame_idx]
            if not mesh_path:
                continue

            points_obj = torch.from_numpy(
                load_obj_vertices_subset(mesh_path, int(num_points))
            ).to(device=device, dtype=dtype)

            losses.append(
                reprojection_loss_single(
                    pred_extrinsics[batch_idx, frame_idx].float(),
                    gt_extrinsics[batch_idx, frame_idx].float(),
                    intrinsics[batch_idx, frame_idx].float(),
                    points_obj.float(),
                    loss_type=loss_type,
                    beta=beta,
                    min_depth=min_depth,
                )
            )

    if len(losses) == 0:
        return (pred_pose_stage * 0).mean()

    return torch.stack(losses).mean()


def translation_normalized_loss_single(
    pred_translation,
    gt_translation,
    loss_type="smooth_l1",
    beta=0.1,
    min_norm=1e-4,
    eps=1e-6,
):
    """
    Compare translation vectors after unit-normalization.

    This keeps the raw translation L1 as the main magnitude supervision while adding
    a separate normalized-vector penalty that is insensitive to translation scale.
    """
    gt_norm = gt_translation.norm(dim=-1)
    valid_mask = gt_norm > min_norm

    if valid_mask.sum() == 0:
        return (pred_translation * 0).mean()

    pred_unit = F.normalize(pred_translation[valid_mask], dim=-1, eps=eps)
    gt_unit = F.normalize(gt_translation[valid_mask], dim=-1, eps=eps)

    if loss_type == "l1":
        loss = (pred_unit - gt_unit).abs().mean(dim=-1)
    elif loss_type == "l2":
        loss = (pred_unit - gt_unit).square().mean(dim=-1)
    elif loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(
            pred_unit,
            gt_unit,
            beta=beta,
            reduction="none",
        ).mean(dim=-1)
    else:
        raise ValueError(f"Unknown normalized translation loss type: {loss_type}")

    loss = check_and_fix_inf_nan(loss, "loss_T_norm")
    return loss.mean()


def translation_aux_loss_single(
    pred_translation,
    gt_translation,
    loss_type="smooth_l1",
    beta=0.1,
    min_norm=1e-4,
    eps=1e-6,
):
    """
    Auxiliary translation losses that focus on vector direction and magnitude.

    The magnitude term is supervised in log space so scale mismatches are penalized
    relatively instead of purely in raw coordinate space. Tiny GT translations are
    ignored because direction and log-norm are ill-defined near zero, which is
    expected for the anchor frame in relative-H2C training.
    """
    gt_norm = gt_translation.norm(dim=-1)
    valid_mask = gt_norm > min_norm

    if valid_mask.sum() == 0:
        dummy_loss = (pred_translation * 0).mean()
        return dummy_loss, dummy_loss

    pred_valid = pred_translation[valid_mask]
    gt_valid = gt_translation[valid_mask]

    loss_dir = 1.0 - F.cosine_similarity(pred_valid, gt_valid, dim=-1, eps=eps)

    pred_log_norm = torch.log(pred_valid.norm(dim=-1).clamp(min=eps))
    gt_log_norm = torch.log(gt_valid.norm(dim=-1).clamp(min=eps))

    if loss_type == "l1":
        loss_log_norm = (pred_log_norm - gt_log_norm).abs()
    elif loss_type == "l2":
        loss_log_norm = (pred_log_norm - gt_log_norm).square()
    elif loss_type == "smooth_l1":
        loss_log_norm = F.smooth_l1_loss(
            pred_log_norm,
            gt_log_norm,
            beta=beta,
            reduction="none",
        )
    else:
        raise ValueError(f"Unknown translation aux loss type: {loss_type}")

    loss_dir = check_and_fix_inf_nan(loss_dir, "loss_T_dir")
    loss_log_norm = check_and_fix_inf_nan(loss_log_norm, "loss_T_log_norm")

    return loss_dir.mean(), loss_log_norm.mean()


def rotation_geodesic_loss_single(
    pred_rot,
    gt_rot,
    pose_encoding_type="absT_quaR_FoV",
    eps=1e-7,
):
    """Compute geodesic angular error in radians for quaternion or Euler rotation encodings."""
    if pose_encoding_type in {"absT_quaR_FoV", "absT_quaR", "quatR"}:
        pred_quat = F.normalize(pred_rot, dim=-1, eps=eps)
        gt_quat = F.normalize(gt_rot, dim=-1, eps=eps)
        cos_half_theta = (pred_quat * gt_quat).sum(dim=-1).abs()
        cos_half_theta = cos_half_theta.clamp(min=0.0, max=1.0 - eps)
        return 2.0 * torch.acos(cos_half_theta)

    if pose_encoding_type == "absT_eulerR_FoV":
        pred_R = euler_to_mat(pred_rot)
        gt_R = euler_to_mat(gt_rot)
        relative_R = pred_R @ gt_R.transpose(-1, -2)
        trace = relative_R[..., 0, 0] + relative_R[..., 1, 1] + relative_R[..., 2, 2]
        cos_theta = ((trace - 1.0) / 2.0).clamp(min=-1.0 + eps, max=1.0 - eps)
        return torch.acos(cos_theta)

    raise ValueError(f"Unknown pose_encoding_type for geodesic loss: {pose_encoding_type}")


def camera_focal_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1", use_focal_ratio=False,
                             pose_encoding_type="absT_quaR_FoV"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.

    Args:
        pred_pose_enc: (B, S, D) or (N, D) predicted pose encoding
        gt_pose_enc: (B, S, D) or (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
        use_focal_ratio: if True, compute focal ratio (frame1/frame0) loss instead of absolute FoV
        pose_encoding_type: Type of pose encoding:
            - "absT_quaR_FoV": T(3) + Quaternion(4) + FoV(2) = 9D
            - "absT_eulerR_FoV": T(3) + Euler(3) + FoV(2) = 8D
    Returns:
        loss_FL: focal length/intrinsics loss (mean)

    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if pose_encoding_type == "absT_quaR_FoV":
        fov_start = 7
    elif pose_encoding_type == "absT_quaR":
        fov_start = None
    elif pose_encoding_type == "absT_eulerR_FoV":
        fov_start = 6
    elif pose_encoding_type == "quatR":
        fov_start = None
    else:
        raise ValueError(f"Unknown pose_encoding_type: {pose_encoding_type}")

    # Handle focal ratio computation for 2-view case
    if fov_start is None:
        loss_FL = pred_pose_enc[..., :1] * 0
    elif use_focal_ratio and pred_pose_enc.dim() >= 2 and pred_pose_enc.shape[-2] >= 2:
        # pred_pose_enc shape: (B, S, D) where S >= 2
        # Compute focal ratio: FoV_1 / FoV_0 (using tan for proper ratio)
        # FoV is in radians, tan(FoV/2) is proportional to 1/focal_length
        pred_fov = pred_pose_enc[..., fov_start:]  # (B, S, 2)
        gt_fov = gt_pose_enc[..., fov_start:]      # (B, S, 2)

        # Compute ratio using log space for numerical stability
        # log(tan(fov1/2)) - log(tan(fov0/2)) = log(f0/f1)
        eps = 1e-6
        pred_log_tan_fov = torch.log(torch.tan(pred_fov / 2).clamp(min=eps))
        gt_log_tan_fov = torch.log(torch.tan(gt_fov / 2).clamp(min=eps))

        # Compute the difference (ratio in log space) between frame 1 and frame 0
        pred_focal_ratio = pred_log_tan_fov[..., 1:, :] - pred_log_tan_fov[..., :1, :]  # (B, S-1, 2)
        gt_focal_ratio = gt_log_tan_fov[..., 1:, :] - gt_log_tan_fov[..., :1, :]        # (B, S-1, 2)

        if loss_type == "l1":
            loss_FL = (pred_focal_ratio - gt_focal_ratio).abs()
        elif loss_type == "l2":
            loss_FL = (pred_focal_ratio - gt_focal_ratio).norm(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    else:
        # Standard absolute FoV loss
        if loss_type == "l1":
            loss_FL = (pred_pose_enc[..., fov_start:] - gt_pose_enc[..., fov_start:]).abs()
        elif loss_type == "l2":
            loss_FL = (pred_pose_enc[..., fov_start:] - gt_pose_enc[..., fov_start:]).norm(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")
    loss_FL = loss_FL.mean()
    return loss_FL


def camera_TR_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1",
                          pose_encoding_type="absT_quaR_FoV", rotation_loss_type="encoding",
                          rotation_geodesic_eps=1e-7):
    """Compute translation and rotation losses without focal supervision."""
    if pose_encoding_type == "absT_quaR_FoV":
        rot_start, rot_end = 3, 7
    elif pose_encoding_type == "absT_quaR":
        rot_start, rot_end = 3, 7
    elif pose_encoding_type == "absT_eulerR_FoV":
        rot_start, rot_end = 3, 6
    elif pose_encoding_type == "quatR":
        rot_start, rot_end = 0, 4
    else:
        raise ValueError(f"Unknown pose_encoding_type: {pose_encoding_type}")

    pred_rot = pred_pose_enc[..., rot_start:rot_end]
    gt_rot = gt_pose_enc[..., rot_start:rot_end]

    if pose_encoding_type == "quatR":
        loss_T = pred_rot[..., :1] * 0
    elif loss_type == "l1":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
    elif loss_type == "l2":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    if rotation_loss_type == "encoding":
        if loss_type == "l1":
            loss_R = (pred_rot - gt_rot).abs()
        elif loss_type == "l2":
            loss_R = (pred_rot - gt_rot).norm(dim=-1)
    elif rotation_loss_type == "geodesic":
        loss_R = rotation_geodesic_loss_single(
            pred_rot,
            gt_rot,
            pose_encoding_type=pose_encoding_type,
            eps=rotation_geodesic_eps,
        )
    else:
        raise ValueError(f"Unknown rotation_loss_type: {rotation_loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    return loss_T, loss_R


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1", use_focal_ratio=False,
                       pose_encoding_type="absT_quaR_FoV", rotation_loss_type="encoding",
                       rotation_geodesic_eps=1e-7):
    """Backward-compatible wrapper that computes translation, rotation, and focal losses together."""
    loss_T, loss_R = camera_TR_loss_single(
        pred_pose_enc,
        gt_pose_enc,
        loss_type=loss_type,
        pose_encoding_type=pose_encoding_type,
        rotation_loss_type=rotation_loss_type,
        rotation_geodesic_eps=rotation_geodesic_eps,
    )
    loss_FL = camera_focal_loss_single(
        pred_pose_enc,
        gt_pose_enc,
        loss_type=loss_type,
        use_focal_ratio=use_focal_ratio,
        pose_encoding_type=pose_encoding_type,
    )
    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def compute_track_loss(
    predictions,
    batch,
    gamma=0.8,
    weight_coord=1.0,
    weight_vis=0.1,
    weight_conf=0.1,
    conf_threshold=3.0,
    use_positive_mask=True,
    vis_aware=False,
    huber=False,
    delta=10,
    vis_aware_w=0.1,
    prob_eps=1e-6,
    **kwargs,
):
    """
    Compute auxiliary tracking losses from first-frame query points.

    The track head predicts trajectories for points queried in frame 0. We supervise:
    - coordinate trajectories on visible frames
    - per-frame visibility classification
    - confidence on visible frames based on a pixel-threshold target
    """
    coord_preds = predictions.get("track_list")
    if coord_preds is None:
        if "track" not in predictions:
            raise KeyError("Tracking loss requested but predictions contain no track outputs.")
        coord_preds = [predictions["track"]]

    vis_scores = predictions["vis"]
    conf_scores = predictions.get("conf")

    gt_tracks = batch["tracks"]
    gt_track_vis_mask = batch["track_vis_mask"].bool()
    train_query_points = coord_preds[-1].shape[2]

    gt_tracks = check_and_fix_inf_nan(
        gt_tracks[:, :, :train_query_points],
        "gt_tracks",
        hard_max=None,
    )
    gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]

    positive_mask = batch.get("track_positive_mask")
    if positive_mask is not None:
        if positive_mask.dim() == 1:
            positive_mask = positive_mask.unsqueeze(0)
        positive_mask = positive_mask[:, :train_query_points].bool()

    valids = torch.ones_like(gt_track_vis_mask, dtype=torch.bool)
    valids = valids & gt_track_vis_mask[:, :1, :]
    if use_positive_mask and positive_mask is not None:
        valids = valids & positive_mask.unsqueeze(1)

    zero_ref = coord_preds[0].mean() * 0
    if not valids.any():
        return {
            "loss_track": zero_ref,
            "loss_track_coord": zero_ref,
            "loss_track_vis": zero_ref,
            "loss_track_conf": zero_ref,
        }

    track_loss = sequence_loss(
        flow_preds=coord_preds,
        flow_gt=gt_tracks,
        vis=gt_track_vis_mask,
        valids=valids,
        gamma=gamma,
        vis_aware=vis_aware,
        huber=huber,
        delta=delta,
        vis_aware_w=vis_aware_w,
    )

    vis_loss = binary_prediction_loss(
        vis_scores[valids],
        gt_track_vis_mask[valids].float(),
        eps=prob_eps,
        loss_name="track_vis_loss",
    )

    if conf_scores is not None:
        conf_valids = valids & gt_track_vis_mask
        if conf_valids.any():
            gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < conf_threshold
            conf_loss = binary_prediction_loss(
                conf_scores[conf_valids],
                gt_conf_mask[conf_valids].float(),
                eps=prob_eps,
                loss_name="track_conf_loss",
            )
        else:
            conf_loss = zero_ref
    else:
        conf_loss = zero_ref

    total_track_loss = (
        track_loss * weight_coord
        + vis_loss * weight_vis
        + conf_loss * weight_conf
    )

    return {
        "loss_track": total_track_loss,
        "loss_track_coord": track_loss,
        "loss_track_vis": vis_loss,
        "loss_track_conf": conf_loss,
    }


def binary_prediction_loss(prediction, target, eps=1e-6, loss_name="binary_pred_loss"):
    """
    BCE helper that accepts either logits or probabilities.
    """
    prediction = check_and_fix_inf_nan(prediction, loss_name, hard_max=None)
    target = target.to(dtype=prediction.dtype)
    pred_detached = prediction.detach()

    if pred_detached.min() < 0 or pred_detached.max() > 1:
        logits = prediction
    else:
        probs = prediction.clamp(min=eps, max=1.0 - eps)
        logits = torch.logit(probs)

    loss = F.binary_cross_entropy_with_logits(logits, target)

    return check_and_fix_inf_nan(loss, loss_name, hard_max=None)


def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    for a, b in zip(x.size(), mask.size()):
        assert a == b
    prod = x * mask

    if dim is None:
        numer = torch.sum(prod)
        denom = torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom.clamp(min=1)
    mean = torch.where(denom > 0,
                       mean,
                       torch.zeros_like(mean))
    return mean


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        if huber:
            i_loss = F.huber_loss(flow_pred, flow_gt, reduction="none", delta=delta)
        else:
            i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_{i}", hard_max=None)

        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)

        num_valid_points = combined_mask.sum()

        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if num_valid_points > 2:
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                i_loss = check_and_fix_inf_nan(i_loss, f"i_loss_iter_safe_check_{i}", hard_max=None)
                flow_loss += 0 * i_loss.mean()

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss
