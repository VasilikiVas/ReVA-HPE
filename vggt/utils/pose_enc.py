# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotation import quat_to_mat, mat_to_quat, euler_to_mat, mat_to_euler


def get_pose_encoding_dim(pose_encoding_type: str) -> int:
    """Get the dimensionality of a pose encoding type.

    Args:
        pose_encoding_type: The type of pose encoding.

    Returns:
        int: The number of dimensions for the encoding.
    """
    if pose_encoding_type == "absT_quaR_FoV":
        return 9  # T(3) + Q(4) + FoV(2)
    elif pose_encoding_type == "absT_quaR":
        return 7  # T(3) + Q(4)
    elif pose_encoding_type == "absT_eulerR_FoV":
        return 8  # T(3) + Euler(3) + FoV(2)
    elif pose_encoding_type == "quatR":
        return 4  # Quaternion (4)
    else:
        raise NotImplementedError(f"Unknown pose_encoding_type: {pose_encoding_type}")


def extri_intri_to_pose_encoding(
    extrinsics, intrinsics, image_size_hw=None, pose_encoding_type="absT_quaR_FoV"  # e.g., (256, 512)
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use:
            - "absT_quaR_FoV": absolute translation, quaternion rotation, field of view (9D)
            - "absT_quaR": absolute translation, quaternion rotation (7D)
            - "absT_eulerR_FoV": absolute translation, Euler angles (roll/pitch/yaw), field of view (8D)
            - "quatR": quaternion rotation only (4D)

    Returns:
        torch.Tensor: Encoded camera pose parameters.
            For "absT_quaR_FoV" type, shape is BxSx9:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
            For "absT_eulerR_FoV" type, shape is BxSx8:
            - [:3] = absolute translation vector T (3D)
            - [3:6] = rotation as Euler angles (roll, pitch, yaw) (3D)
            - [6:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    R = extrinsics[:, :, :3, :3]  # BxSx3x3
    T = extrinsics[:, :, :3, 3]  # BxSx3

    # Compute FoV (common to both encoding types)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])

    if pose_encoding_type == "absT_quaR_FoV":
        quat = mat_to_quat(R)
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    elif pose_encoding_type == "absT_quaR":
        quat = mat_to_quat(R)
        pose_encoding = torch.cat([T, quat], dim=-1).float()
    elif pose_encoding_type == "absT_eulerR_FoV":
        euler = mat_to_euler(R)  # (B, S, 3) - roll, pitch, yaw
        pose_encoding = torch.cat([T, euler, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    elif pose_encoding_type == "quatR":
        pose_encoding = mat_to_quat(R).float()
    else:
        raise NotImplementedError(f"Unknown pose_encoding_type: {pose_encoding_type}")

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters.
            For "absT_quaR_FoV" type, shape is BxSx9:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
            For "absT_eulerR_FoV" type, shape is BxSx8:
            - [:3] = absolute translation vector T (3D)
            - [3:6] = rotation as Euler angles (roll, pitch, yaw) (3D)
            - [6:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used:
            - "absT_quaR_FoV": quaternion rotation (9D)
            - "absT_quaR": translation + quaternion rotation (7D)
            - "absT_eulerR_FoV": Euler angles rotation (8D)
            - "quatR": quaternion rotation only (4D)
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None
    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]
        R = quat_to_mat(quat)
    elif pose_encoding_type == "absT_quaR":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        R = quat_to_mat(quat)
        fov_h = None
        fov_w = None
    elif pose_encoding_type == "absT_eulerR_FoV":
        T = pose_encoding[..., :3]
        euler = pose_encoding[..., 3:6]
        fov_h = pose_encoding[..., 6]
        fov_w = pose_encoding[..., 7]
        R = euler_to_mat(euler)
    elif pose_encoding_type == "quatR":
        T = torch.zeros(pose_encoding.shape[:-1] + (3,), device=pose_encoding.device, dtype=pose_encoding.dtype)
        quat = pose_encoding[..., :4]
        R = quat_to_mat(quat)
        fov_h = None
        fov_w = None
    else:
        raise NotImplementedError(f"Unknown pose_encoding_type: {pose_encoding_type}")

    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    if build_intrinsics and pose_encoding_type not in {"quatR", "absT_quaR"}:
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)
        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1

    return extrinsics, intrinsics
