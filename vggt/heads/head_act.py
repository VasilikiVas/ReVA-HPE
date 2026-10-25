# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear",
                  pose_encoding_type="absT_quaR_FoV"):
    """
    Activate pose parameters with specified activation functions.

    Args:
        pred_pose_enc: Tensor containing encoded pose parameters
            For quaternion (absT_quaR_FoV): [translation(3), quaternion(4), fov(2)] = 9D
            For quaternion without fov (absT_quaR): [translation(3), quaternion(4)] = 7D
            For Euler (absT_eulerR_FoV): [translation(3), euler(3), fov(2)] = 8D
            For rotation-only (quatR): [quaternion(4)] = 4D
        trans_act: Activation type for translation component
        quat_act: Activation type for rotation component (quaternion or Euler)
        fl_act: Activation type for focal length component
        pose_encoding_type: Type of pose encoding ("absT_quaR_FoV", "absT_quaR", "absT_eulerR_FoV", or "quatR")

    Returns:
        Activated pose parameters tensor
    """
    if pose_encoding_type == "absT_quaR_FoV":
        T = pred_pose_enc[..., :3]
        rot = pred_pose_enc[..., 3:7]  # quaternion (4D)
        fl = pred_pose_enc[..., 7:]    # fov (2D)
    elif pose_encoding_type == "absT_quaR":
        T = pred_pose_enc[..., :3]
        rot = pred_pose_enc[..., 3:7]  # quaternion (4D)
        T = base_pose_act(T, trans_act)
        rot = base_pose_act(rot, quat_act)
        return torch.cat([T, rot], dim=-1)
    elif pose_encoding_type == "absT_eulerR_FoV":
        T = pred_pose_enc[..., :3]
        rot = pred_pose_enc[..., 3:6]  # euler angles (3D)
        fl = pred_pose_enc[..., 6:]    # fov (2D)
    elif pose_encoding_type == "quatR":
        rot = pred_pose_enc[..., :4]
        rot = base_pose_act(rot, quat_act)
        return rot
    else:
        raise ValueError(f"Unknown pose_encoding_type: {pose_encoding_type}")

    T = base_pose_act(T, trans_act)
    rot = base_pose_act(rot, quat_act)  # works for both quaternion and euler
    fl = base_pose_act(fl, fl_act)

    pred_pose_enc = torch.cat([T, rot, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    Apply basic activation function to pose parameters.

    Args:
        pose_enc: Tensor containing encoded pose parameters
        act_type: Activation type ("linear", "inv_log", "exp", "relu")

    Returns:
        Activated pose parameters
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    Process network output to extract 3D points and confidence values.

    Args:
        out: Network output tensor (B, C, H, W)
        activation: Activation type for 3D points
        conf_activation: Activation type for confidence values

    Returns:
        Tuple of (3D points tensor, confidence tensor)
    """
    # Move channels from last dim to the 4th dimension => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C expected

    # Split into xyz (first C-1 channels) and confidence (last channel)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    Apply inverse log transform: sign(y) * (exp(|y|) - 1)

    Args:
        y: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))
