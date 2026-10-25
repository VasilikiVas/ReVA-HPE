# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from PyTorch3D, https://github.com/facebookresearch/pytorch3d

import torch
import numpy as np
import torch.nn.functional as F


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_euler(matrix: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles (roll, pitch, yaw).

    Uses the XYZ convention by default:
    - Roll (X): rotation around the X-axis (left-right tilt)
    - Pitch (Y): rotation around the Y-axis (up-down nod)
    - Yaw (Z): rotation around the Z-axis (left-right turn)

    The rotation is applied as R = Rz @ Ry @ Rx (intrinsic rotations).

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Euler angle convention. Currently supports "XYZ".

    Returns:
        Euler angles (roll, pitch, yaw) as tensor of shape (..., 3) in radians.
        Range: roll [-pi, pi], pitch [-pi/2, pi/2], yaw [-pi, pi]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    if convention != "XYZ":
        raise NotImplementedError(f"Convention {convention} not implemented. Use 'XYZ'.")

    # Extract rotation matrix elements
    # R = Rz @ Ry @ Rx for XYZ intrinsic (or equivalently X'Y'Z' extrinsic)
    #
    # R = | cy*cz  -cy*sz   sy  |
    #     | cx*sz+sx*sy*cz  cx*cz-sx*sy*sz  -sx*cy |
    #     | sx*sz-cx*sy*cz  sx*cz+cx*sy*sz   cx*cy |
    #
    # where cx=cos(roll), sx=sin(roll), cy=cos(pitch), sy=sin(pitch), cz=cos(yaw), sz=sin(yaw)

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    # Compute pitch (Y rotation) - handle gimbal lock
    # sy = m02, so pitch = arcsin(m02)
    # Clamp to avoid numerical issues at ±1
    sy = m02.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    pitch = torch.asin(sy)

    # Compute roll (X rotation) and yaw (Z rotation)
    # For the non-gimbal-lock case:
    # roll = atan2(-m12, m22) = atan2(sx*cy, cx*cy) = atan2(sx, cx)
    # yaw = atan2(-m01, m00) = atan2(cy*sz, cy*cz) = atan2(sz, cz)
    cy = torch.cos(pitch)

    # Handle gimbal lock (when pitch is close to ±90°)
    gimbal_lock = cy.abs() < 1e-6

    # Normal case
    roll = torch.atan2(-m12, m22)
    yaw = torch.atan2(-m01, m00)

    # Gimbal lock case: set roll=0 and compute yaw from m10, m11
    # When pitch = +90°: m10 = sin(yaw-roll), m11 = cos(yaw-roll), set roll=0
    # When pitch = -90°: m10 = -sin(yaw+roll), m11 = cos(yaw+roll), set roll=0
    roll_gimbal = torch.zeros_like(roll)
    yaw_gimbal = torch.atan2(m10, m11)

    roll = torch.where(gimbal_lock, roll_gimbal, roll)
    yaw = torch.where(gimbal_lock, yaw_gimbal, yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_mat(euler: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrices.

    Uses the XYZ convention by default:
    - Roll (X): rotation around the X-axis
    - Pitch (Y): rotation around the Y-axis
    - Yaw (Z): rotation around the Z-axis

    The rotation is applied as R = Rz @ Ry @ Rx (intrinsic rotations).

    Args:
        euler: Euler angles (roll, pitch, yaw) as tensor of shape (..., 3) in radians.
        convention: Euler angle convention. Currently supports "XYZ".

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler.size(-1) != 3:
        raise ValueError(f"Invalid euler angles shape {euler.shape}. Expected (..., 3).")

    if convention != "XYZ":
        raise NotImplementedError(f"Convention {convention} not implemented. Use 'XYZ'.")

    roll = euler[..., 0]   # X rotation
    pitch = euler[..., 1]  # Y rotation
    yaw = euler[..., 2]    # Z rotation

    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)

    # R = Rz @ Ry @ Rx
    # Build rotation matrix elements
    m00 = cy * cz
    m01 = -cy * sz
    m02 = sy
    m10 = cx * sz + sx * sy * cz
    m11 = cx * cz - sx * sy * sz
    m12 = -sx * cy
    m20 = sx * sz - cx * sy * cz
    m21 = sx * cz + cx * sy * sz
    m22 = cx * cy

    matrix = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)

    return matrix
