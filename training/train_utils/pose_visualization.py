# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualization utilities for H2C (Head-to-Camera) pose predictions.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def denormalize_image(img: torch.Tensor, use_imagenet_norm: bool = False) -> np.ndarray:
    """
    Denormalize an image tensor and convert to numpy array (H, W, C) in [0, 255].

    Args:
        img: Image tensor of shape (C, H, W) or (H, W, C), in [0, 1] range
        use_imagenet_norm: If True, undo ImageNet normalization. If False, just scale to [0,255]

    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255]
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    # Ensure (H, W, C) format
    if img.shape[0] in [1, 3]:  # CHW format
        img = np.transpose(img, (1, 2, 0))

    if use_imagenet_norm:
        # Undo ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean

    # Scale to [0, 255]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def euler_to_degrees(euler_rad: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles from radians to degrees."""
    return euler_rad * 180 / math.pi


def format_pose_text(
    translation: np.ndarray,
    rotation: np.ndarray,
    pose_encoding_type: str = "absT_eulerR_FoV",
    prefix: str = "",
) -> List[str]:
    """
    Format pose parameters as text lines for visualization.

    Args:
        translation: Translation vector (3,)
        rotation: Rotation (4 for quaternion, 3 for Euler in radians)
        pose_encoding_type: Type of rotation encoding
        prefix: Prefix for each line (e.g., "Pred: " or "GT: ")

    Returns:
        List of text lines
    """
    lines = []
    lines.append(f"{prefix}T: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")

    if pose_encoding_type == "absT_eulerR_FoV":
        # Euler angles in degrees
        rot_deg = rotation * 180 / np.pi
        lines.append(f"{prefix}R: [roll={rot_deg[0]:.1f}°, pitch={rot_deg[1]:.1f}°, yaw={rot_deg[2]:.1f}°]")
    else:
        # Quaternion
        lines.append(f"{prefix}Q: [{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}, {rotation[3]:.3f}]")

    return lines


def create_pose_comparison_image(
    images: torch.Tensor,
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    pose_encoding_type: str = "absT_eulerR_FoV",
    sample_idx: int = 0,
    supervise_frame_idxs: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create a visualization image comparing predicted and GT poses.

    Args:
        images: Image tensor (B, S, C, H, W) or (B, S, H, W, C)
        pred_pose: Predicted pose encoding (B, S, D)
        gt_pose: Ground truth pose encoding (B, S, D)
        pose_encoding_type: Type of pose encoding
        sample_idx: Which sample in the batch to visualize
        supervise_frame_idxs: Optional frame indices that receive direct supervision

    Returns:
        Visualization image as numpy array (H, W, 3)
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for visualization. Install with: pip install Pillow")

    # Extract sample - convert to float32 for numpy compatibility (handles bfloat16)
    imgs = images[sample_idx]  # (S, C, H, W) or (S, H, W, C)
    pred = pred_pose[sample_idx].float().cpu().numpy()  # (S, D)
    gt = gt_pose[sample_idx].float().cpu().numpy()  # (S, D)

    S = imgs.shape[0]
    supervised_frame_set = (
        None if supervise_frame_idxs is None else {int(idx) for idx in supervise_frame_idxs}
    )

    # Determine rotation indices based on encoding type
    if pose_encoding_type == "absT_eulerR_FoV":
        rot_start, rot_end = 3, 6
        fov_start = 6
    elif pose_encoding_type == "absT_quaR":
        rot_start, rot_end = 3, 7
        fov_start = None
    else:
        rot_start, rot_end = 3, 7
        fov_start = 7

    # Denormalize images - convert to float32 for numpy
    img_list = []
    for i in range(S):
        img = denormalize_image(imgs[i].float())
        img_list.append(img)

    # Create side-by-side image
    H, W = img_list[0].shape[:2]

    # Layout: images on top, pose info below
    text_height = 120
    combined_width = W * S + (S - 1) * 10  # 10px gap between images
    combined_height = H + text_height

    canvas = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

    # Place images
    x_offset = 0
    for i, img in enumerate(img_list):
        canvas[:H, x_offset:x_offset + W] = img
        x_offset += W + 10

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Add pose text for each frame
    y_text = H + 5
    x_offset = 5

    for i in range(S):
        # Extract pose components
        pred_T = pred[i, :3]
        pred_R = pred[i, rot_start:rot_end]
        gt_T = gt[i, :3]
        gt_R = gt[i, rot_start:rot_end]

        # Calculate errors
        T_error = np.linalg.norm(pred_T - gt_T)
        R_error = np.linalg.norm(pred_R - gt_R)
        if pose_encoding_type == "absT_eulerR_FoV":
            R_error_deg = R_error * 180 / np.pi
        else:
            R_error_deg = R_error  # Quaternion error (not directly degrees)

        if supervised_frame_set is None:
            frame_header = f"Frame {i}:"
        elif i in supervised_frame_set:
            frame_header = f"Frame {i} (supervised):"
        else:
            frame_header = f"Frame {i} (context only):"

        # Format text
        text_lines = [
            frame_header,
            f"  Pred T: [{pred_T[0]:.2f}, {pred_T[1]:.2f}, {pred_T[2]:.2f}]",
            f"  GT T:   [{gt_T[0]:.2f}, {gt_T[1]:.2f}, {gt_T[2]:.2f}]",
            f"  T err: {T_error:.3f}",
        ]

        if pose_encoding_type == "absT_eulerR_FoV":
            pred_deg = pred_R * 180 / np.pi
            gt_deg = gt_R * 180 / np.pi
            text_lines.extend([
                f"  Pred R: [r={pred_deg[0]:.1f}°, p={pred_deg[1]:.1f}°, y={pred_deg[2]:.1f}°]",
                f"  GT R:   [r={gt_deg[0]:.1f}°, p={gt_deg[1]:.1f}°, y={gt_deg[2]:.1f}°]",
                f"  R err: {R_error_deg:.2f}°",
            ])
        else:
            text_lines.extend([
                f"  Pred Q: [{pred_R[0]:.3f}, {pred_R[1]:.3f}, {pred_R[2]:.3f}, {pred_R[3]:.3f}]",
                f"  GT Q:   [{gt_R[0]:.3f}, {gt_R[1]:.3f}, {gt_R[2]:.3f}, {gt_R[3]:.3f}]",
                f"  Q err: {R_error_deg:.3f}",
            ])

        # Draw text
        for j, line in enumerate(text_lines):
            color = (0, 0, 0) if "err" not in line else (200, 0, 0)
            draw.text((x_offset, y_text + j * 14), line, fill=color, font=font)

        x_offset += W + 10

    return np.array(pil_img)


def create_batch_visualization(
    images: torch.Tensor,
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    pose_encoding_type: str = "absT_eulerR_FoV",
    max_samples: int = 4,
    supervise_frame_idxs: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Create a grid visualization for multiple samples in a batch.

    Args:
        images: Image tensor (B, S, C, H, W)
        pred_pose: Predicted pose encoding (B, S, D)
        gt_pose: Ground truth pose encoding (B, S, D)
        pose_encoding_type: Type of pose encoding
        max_samples: Maximum number of samples to visualize
        supervise_frame_idxs: Optional frame indices that receive direct supervision

    Returns:
        Grid visualization as numpy array
    """
    B = min(images.shape[0], max_samples)

    vis_list = []
    for i in range(B):
        vis = create_pose_comparison_image(
            images, pred_pose, gt_pose,
            pose_encoding_type=pose_encoding_type,
            sample_idx=i,
            supervise_frame_idxs=supervise_frame_idxs,
        )
        vis_list.append(vis)

    # Stack vertically
    return np.vstack(vis_list)


def compute_pose_metrics(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    pose_encoding_type: str = "absT_eulerR_FoV",
) -> Dict[str, float]:
    """
    Compute pose prediction metrics.

    Args:
        pred_pose: Predicted pose encoding (B, S, D)
        gt_pose: Ground truth pose encoding (B, S, D)
        pose_encoding_type: Type of pose encoding

    Returns:
        Dictionary of metrics
    """
    # Convert to float32 for consistent computation (handles bfloat16)
    pred_pose = pred_pose.float()
    gt_pose = gt_pose.float()

    # Determine indices
    if pose_encoding_type == "absT_eulerR_FoV":
        rot_start, rot_end = 3, 6
        fov_start = 6
    elif pose_encoding_type == "absT_quaR":
        rot_start, rot_end = 3, 7
        fov_start = None
    else:
        rot_start, rot_end = 3, 7
        fov_start = 7

    # Extract components
    pred_T = pred_pose[..., :3]
    pred_R = pred_pose[..., rot_start:rot_end]
    gt_T = gt_pose[..., :3]
    gt_R = gt_pose[..., rot_start:rot_end]

    # Compute errors
    T_error = (pred_T - gt_T).norm(dim=-1).mean().item()
    R_error = (pred_R - gt_R).norm(dim=-1).mean().item()

    metrics = {
        "translation_error": T_error,
        "rotation_error": R_error,
    }
    if fov_start is not None:
        pred_fov = pred_pose[..., fov_start:]
        gt_fov = gt_pose[..., fov_start:]
        metrics["fov_error"] = (pred_fov - gt_fov).abs().mean().item()

    # Add rotation error in degrees for Euler
    if pose_encoding_type == "absT_eulerR_FoV":
        R_error_deg = R_error * 180 / math.pi
        metrics["rotation_error_deg"] = R_error_deg

    return metrics


def log_pose_visualization(
    logger,
    images: torch.Tensor,
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    pose_encoding_type: str,
    step: int,
    phase: str = "train",
    max_samples: int = 2,
    supervise_frame_idxs: Optional[List[int]] = None,
) -> None:
    """
    Log pose visualization to wandb or tensorboard.

    Args:
        logger: WandB or TensorBoard logger instance
        images: Image tensor (B, S, C, H, W)
        pred_pose: Predicted pose encoding (B, S, D)
        gt_pose: Ground truth pose encoding (B, S, D)
        pose_encoding_type: Type of pose encoding
        step: Current training step
        phase: "train" or "val"
        max_samples: Maximum samples to visualize
        supervise_frame_idxs: Optional frame indices that receive direct supervision
    """
    try:
        vis = create_batch_visualization(
            images, pred_pose, gt_pose,
            pose_encoding_type=pose_encoding_type,
            max_samples=max_samples,
            supervise_frame_idxs=supervise_frame_idxs,
        )

        # Convert to (C, H, W) for logging
        vis = np.transpose(vis, (2, 0, 1))

        logger.log_visuals(f"{phase}/pose_comparison", vis, step)

        # Also log metrics
        metrics = compute_pose_metrics(pred_pose, gt_pose, pose_encoding_type)
        for name, value in metrics.items():
            logger.log(f"{phase}/{name}", value, step)

    except Exception as e:
        import logging
        logging.warning(f"Failed to log pose visualization: {e}")


# ============================================================================
# Mesh Rendering Visualization (requires PyTorch3D)
# ============================================================================

try:
    from pytorch3d.renderer import (
        PerspectiveCameras, MeshRenderer, MeshRasterizer,
        RasterizationSettings, HardFlatShader, DirectionalLights, BlendParams
    )
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.textures import TexturesVertex
    PYTORCH3D_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"PyTorch3D import failed: {e}")
    PYTORCH3D_AVAILABLE = False
except Exception as e:
    import logging
    logging.warning(f"PyTorch3D import failed with unexpected error: {e}")
    PYTORCH3D_AVAILABLE = False


def pose_encoding_to_RT(
    pose_enc: np.ndarray,
    pose_encoding_type: str = "absT_quaR_FoV",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pose encoding to rotation matrix R and translation vector T.

    Args:
        pose_enc: Pose encoding array (D,)
        pose_encoding_type: Type of pose encoding

    Returns:
        R: 3x3 rotation matrix
        T: 3D translation vector
    """
    if pose_encoding_type == "quatR":
        T = np.zeros(3, dtype=np.float32)
        from scipy.spatial.transform import Rotation
        quat = pose_enc[:4]
        R = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()
    elif pose_encoding_type == "absT_eulerR_FoV":
        T = pose_enc[:3]
        # Euler angles (roll, pitch, yaw) in radians
        from scipy.spatial.transform import Rotation
        euler = pose_enc[3:6]
        R = Rotation.from_euler('xyz', euler).as_matrix()
    else:
        T = pose_enc[:3]
        # Quaternion stored as (x, y, z, w) in VGGT (scalar-last)
        from scipy.spatial.transform import Rotation
        quat = pose_enc[3:7]
        R = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()

    return R.astype(np.float32), T.astype(np.float32)


def load_h2c_opencv(view_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load head-to-camera (H2C) transformation in OpenCV convention.

    Uses output_opencv_camera.pkl (pre-converted) and the object transform.
    Both are in Blender world coordinates so they compose directly.

    Args:
        view_dir: Path to view directory containing pkl files.

    Returns:
        R_h2c: 3x3 rotation matrix (head to camera, OpenCV convention)
        t_h2c: 3D translation vector (head to camera, OpenCV convention)
    """
    import pickle
    import os

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

    return R_h2c.astype(np.float32), t_h2c.astype(np.float32)


def render_mesh_with_pose(
    mesh_path: str,
    R: np.ndarray,
    T: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    device: str = "cuda",
    color: Tuple[float, float, float] = (0.9, 0.2, 0.2),
) -> np.ndarray:
    """
    Render a mesh with given camera pose using PyTorch3D.

    Args:
        mesh_path: Path to .obj mesh file
        R: 3x3 rotation matrix (head-to-camera, OpenCV convention)
        T: 3D translation vector (head-to-camera, OpenCV convention)
        intrinsics: 3x3 camera intrinsics matrix (OpenCV convention)
        image_size: (H, W) output image size
        device: Device to use for rendering
        color: RGB color tuple for mesh

    Returns:
        Rendered image as numpy array (H, W, 3) in [0, 255]
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("PyTorch3D is required for mesh rendering")

    H, W = image_size

    # Convert OpenCV R, T to PyTorch3D convention for PerspectiveCameras.
    # OpenCV:    p_cam = R_cv @ p_world + t_cv  (column vectors)
    # PyTorch3D: p_cam = p_world @ R_p3d + T_p3d (row vectors)
    # Relationship: R_p3d = R_cv.T @ flip,  T_p3d = flip @ t_cv
    # where flip = diag(-1, -1, 1)  (flips X and Y axes between conventions)
    flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
    R_p3d = (R.T @ flip).astype(np.float32)
    T_p3d = (flip @ T.flatten()).astype(np.float32)

    # Disable autocast to ensure float32 operations for PyTorch3D
    with torch.amp.autocast(device_type='cuda', enabled=False):
        # Load mesh
        verts, faces, _ = load_obj(mesh_path)
        faces_idx = faces.verts_idx

        # Explicitly use float32 for mesh vertices
        verts = verts.to(device=device, dtype=torch.float32)[None]  # (1, N, 3)
        faces_idx = faces_idx.to(device)[None]

        # Create colored texture
        colors = torch.ones_like(verts, dtype=torch.float32)
        colors[..., 0] = color[0]  # R
        colors[..., 1] = color[1]  # G
        colors[..., 2] = color[2]  # B
        textures = TexturesVertex(verts_features=colors)
        mesh = Meshes(verts=verts, faces=faces_idx, textures=textures)

        # Extract intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        R_t = torch.tensor(R_p3d, dtype=torch.float32, device=device)[None]
        T_t = torch.tensor(T_p3d, dtype=torch.float32, device=device)[None]

        cameras = PerspectiveCameras(
            R=R_t,
            T=T_t,
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
            in_ndc=False,
            device=device,
        )

        # Setup renderer
        raster_settings = RasterizationSettings(
            image_size=(H, W),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
        )

        lights = DirectionalLights(device=device, direction=((0, 0, 1),))
        blend_params = BlendParams(background_color=(0, 0, 0))

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardFlatShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params),
        )

        # Render
        with torch.no_grad():
            img = renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy()

        return (img * 255).astype(np.uint8)


def create_mesh_overlay_visualization(
    image: np.ndarray,
    mesh_path: str,
    pred_pose: np.ndarray,
    gt_pose: np.ndarray,
    intrinsics: np.ndarray,
    pose_encoding_type: str = "absT_quaR_FoV",
    device: str = "cuda",
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create visualization with mesh overlay showing predicted vs GT pose.

    Args:
        image: Original image (H, W, 3) in [0, 255]
        mesh_path: Path to .obj mesh file
        pred_pose: Predicted pose encoding (D,)
        gt_pose: Ground truth pose encoding (D,)
        intrinsics: 3x3 camera intrinsics matrix
        pose_encoding_type: Type of pose encoding
        device: Device for rendering
        alpha: Blend factor for overlay

    Returns:
        Visualization image (H, W*3, 3) showing [original | pred_overlay | gt_overlay]
    """
    if not PYTORCH3D_AVAILABLE:
        # Return placeholder if PyTorch3D not available
        H, W = image.shape[:2]
        placeholder = np.ones((H, W * 3, 3), dtype=np.uint8) * 128
        return placeholder

    H, W = image.shape[:2]

    # Convert pose encodings to R, T
    pred_R, pred_T = pose_encoding_to_RT(pred_pose, pose_encoding_type)
    gt_R, gt_T = pose_encoding_to_RT(gt_pose, pose_encoding_type)

    # Render mesh with predicted pose
    pred_render = render_mesh_with_pose(
        mesh_path, pred_R, pred_T, intrinsics, (H, W), device
    )

    # Render mesh with GT pose
    gt_render = render_mesh_with_pose(
        mesh_path, gt_R, gt_T, intrinsics, (H, W), device
    )

    # Create overlays
    pred_overlay = cv2.addWeighted(image, 1 - alpha, pred_render, alpha, 0)
    gt_overlay = cv2.addWeighted(image, 1 - alpha, gt_render, alpha, 0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(pred_overlay, "Predicted", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(gt_overlay, "GT", (10, 30), font, 0.8, (0, 0, 255), 2)

    # Concatenate horizontally
    return np.hstack([image, pred_overlay, gt_overlay])


def create_mesh_batch_visualization(
    images: torch.Tensor,
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
    mesh_paths: List[List[str]],
    intrinsics: torch.Tensor,
    pose_encoding_type: str = "absT_quaR_FoV",
    pose_mode: str = "relative_h2c",
    device: str = "cuda",
    max_samples: int = 2,
    view_paths: List[List[str]] = None,
    translation_target_type: str = "relative_se3",
    absolute_extrinsics: torch.Tensor = None,
) -> np.ndarray:
    """
    Create a pose visualization for relative-H2C, absolute-second, or absolute-single training.

    For relative-H2C, this renders frame 1's mesh at frame 2's viewpoint.
    For absolute-second, this renders frame 2's mesh with the predicted absolute frame-2 pose.
    For absolute-single, this renders frame 1's mesh with the predicted absolute frame-1 pose.

    Args:
        images: Image tensor (B, S, C, H, W)
        pred_pose: Predicted pose encoding (B, S, D)
        gt_pose: Ground truth pose encoding (B, S, D)
        mesh_paths: List of mesh paths per sample [[path1, path2], ...]
        intrinsics: Intrinsics tensor (B, S, 3, 3)
        pose_encoding_type: Type of pose encoding
        pose_mode: "relative_h2c", "absolute_second", or "absolute_single"
        device: Device for rendering
        max_samples: Maximum samples to visualize
        view_paths: List of view directory paths [[view1_path, view2_path], ...]
        translation_target_type: Translation target stored in pred_pose[..., :3]
        absolute_extrinsics: Original absolute H2C extrinsics, before first-frame normalization

    Returns:
        Grid visualization as numpy array
    """
    import pickle
    import os

    if not PYTORCH3D_AVAILABLE:
        import logging
        logging.warning("PyTorch3D not available, skipping mesh visualization")
        return None

    B_full = images.shape[0]
    S = images.shape[1]

    if pose_mode not in {"relative_h2c", "absolute_second", "absolute_single"}:
        raise ValueError(f"Unknown pose_mode: {pose_mode}")

    def _is_frame_major(paths):
        return (
            isinstance(paths, (list, tuple))
            and len(paths) == S
            and all(isinstance(p, (list, tuple)) and len(p) == B_full for p in paths)
        )

    def _transpose_paths(paths):
        return [[paths[s][b] for s in range(S)] for b in range(B_full)]

    if _is_frame_major(mesh_paths):
        mesh_paths = _transpose_paths(mesh_paths)

    if view_paths is not None and _is_frame_major(view_paths):
        view_paths = _transpose_paths(view_paths)

    B = min(B_full, max_samples)

    vis_rows = []

    for b in range(B):
        if b >= len(mesh_paths):
            continue

        min_required_meshes = 1 if pose_mode == "absolute_single" else 2
        if not isinstance(mesh_paths[b], (list, tuple)) or len(mesh_paths[b]) < min_required_meshes:
            continue

        pose_frame_idx = 0 if pose_mode == "absolute_single" else 1
        target_frame_idx = pose_frame_idx

        source_img = denormalize_image(images[b, 0].float())
        target_img = denormalize_image(images[b, target_frame_idx].float())
        H_img, W_img = target_img.shape[:2]

        if pose_mode == "relative_h2c":
            mesh_path = mesh_paths[b][0]
            source_label = "Frame 1 (Mesh Src)"
            target_label = "Frame 2 (Target)"
            pred_label = "Pred rel->abs (Green)"
            gt_label = "GT abs (Red)"
        elif pose_mode == "absolute_second":
            mesh_path = mesh_paths[b][1]
            source_label = "Frame 1 (Context)"
            target_label = "Frame 2 (Target/Mesh Src)"
            pred_label = "Pred abs f2 (Green)"
            gt_label = "GT abs f2 (Red)"
        else:
            mesh_path = mesh_paths[b][0]
            source_label = "Frame 1 (Target/Mesh Src)"
            target_label = None
            pred_label = "Pred abs f1 (Green)"
            gt_label = "GT abs f1 (Red)"

        # Default to processed intrinsics
        intri = intrinsics[b, target_frame_idx].float().cpu().numpy()
        H_orig, W_orig = H_img, W_img

        # Load ORIGINAL intrinsics and images from view paths (matching notebook approach)
        if view_paths is not None and CV2_AVAILABLE:
            if (
                b >= len(view_paths)
                or not isinstance(view_paths[b], (list, tuple))
                or len(view_paths[b]) <= target_frame_idx
            ):
                continue

            view1_dir = view_paths[b][0]
            target_view_dir = view_paths[b][target_frame_idx]
            try:
                cam_path = os.path.join(target_view_dir, "output_opencv_camera.pkl")
                with open(cam_path, "rb") as f:
                    cam = pickle.load(f)
                intri = np.array(cam["K"]).reshape(3, 3).astype(np.float64)
                H_orig = int(cam["image_height"])
                W_orig = int(cam["image_width"])

                src_bgr = cv2.imread(os.path.join(view1_dir, "output.png"))
                tgt_bgr = cv2.imread(os.path.join(target_view_dir, "output.png"))
                if tgt_bgr is not None:
                    if src_bgr is not None:
                        source_img = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
                        if source_img.shape[:2] != (H_orig, W_orig):
                            source_img = cv2.resize(source_img, (W_orig, H_orig))
                    target_img = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
                    if target_img.shape[:2] != (H_orig, W_orig):
                        target_img = cv2.resize(target_img, (W_orig, H_orig))
                    H_img, W_img = H_orig, W_orig
                    if pose_mode == "absolute_single":
                        source_img = target_img.copy()
            except Exception:
                # Fall back to processed images/intrinsics
                intri = intrinsics[b, target_frame_idx].float().cpu().numpy()
                H_orig, W_orig = H_img, W_img

        pred_R_f2, pred_T_f2 = pose_encoding_to_RT(
            pred_pose[b, pose_frame_idx].float().cpu().numpy(),
            pose_encoding_type,
        )

        if pose_mode == "relative_h2c":
            # Training normalization: E_norm = E_abs @ E_0^{-1}
            # Undo with frame-1 absolute H2C: E_abs = E_norm @ E_0.
            pred_R, pred_T = pred_R_f2, pred_T_f2
            if view_paths is not None:
                try:
                    R_h2c_0, t_h2c_0 = load_h2c_opencv(view_paths[b][0])
                    pred_R = pred_R_f2 @ R_h2c_0
                    pred_T = pred_R_f2 @ t_h2c_0 + pred_T_f2
                except Exception:
                    pass
            if (
                translation_target_type in {"delta_t", "depth_norm_delta_t", "projective_delta_t"}
                and absolute_extrinsics is not None
            ):
                try:
                    anchor_t = absolute_extrinsics[b, 0, :3, 3].float().cpu().numpy()
                    pred_slot = pred_T_f2.astype(np.float64)
                    if translation_target_type == "delta_t":
                        pred_T = anchor_t + pred_slot
                    else:
                        eps = np.finfo(np.float32).eps
                        z0 = max(float(anchor_t[2]), float(eps))
                        log_z_ratio = float(np.clip(pred_slot[2], -20.0, 20.0))
                        z = z0 * np.exp(log_z_ratio)
                        if translation_target_type == "depth_norm_delta_t":
                            x = anchor_t[0] + pred_slot[0] * z0
                            y = anchor_t[1] + pred_slot[1] * z0
                        else:
                            u0 = anchor_t[0] / z0
                            v0 = anchor_t[1] / z0
                            x = (u0 + pred_slot[0]) * z
                            y = (v0 + pred_slot[1]) * z
                        pred_T = np.array([x, y, z], dtype=np.float32)
                except Exception:
                    pass
        else:
            pred_R, pred_T = pred_R_f2, pred_T_f2

        # For GT: load H2C directly from OpenCV pkl files
        gt_R, gt_T = None, None
        if view_paths is not None:
            try:
                gt_R, gt_T = load_h2c_opencv(view_paths[b][target_frame_idx])
            except Exception as e:
                import logging
                logging.warning(f"Failed to load H2C from {view_paths[b][target_frame_idx]}: {e}")

        # Fallback to pose encoding if loading failed
        if gt_R is None or gt_T is None:
            gt_R, gt_T = pose_encoding_to_RT(gt_pose[b, pose_frame_idx].float().cpu().numpy(), pose_encoding_type)

        # Render mesh overlays (OpenCV convention R, T — render_mesh_with_pose handles conversion)
        pred_render = render_mesh_with_pose(
            mesh_path, pred_R, pred_T, intri, (H_orig, W_orig), device,
            color=(0.2, 0.9, 0.2)
        )
        gt_render = render_mesh_with_pose(
            mesh_path, gt_R, gt_T, intri, (H_orig, W_orig), device,
            color=(0.9, 0.2, 0.2)
        )

        # Resize renders to match training image size
        if H_orig != H_img or W_orig != W_img:
            pred_render = cv2.resize(pred_render, (W_img, H_img))
            gt_render = cv2.resize(gt_render, (W_img, H_img))

        # Create overlays (0.3 original, 0.7 render)
        pred_overlay = cv2.addWeighted(target_img, 0.3, pred_render, 0.7, 0)
        gt_overlay = cv2.addWeighted(target_img, 0.3, gt_render, 0.7, 0)

        # Add labels (RGB format)
        font = cv2.FONT_HERSHEY_SIMPLEX
        source_img_labeled = source_img.copy()
        target_img_labeled = target_img.copy() if target_label is not None else None
        cv2.putText(source_img_labeled, source_label, (10, 30), font, 0.7, (255, 255, 255), 2)
        if target_img_labeled is not None:
            cv2.putText(target_img_labeled, target_label, (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(pred_overlay, pred_label, (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(gt_overlay, gt_label, (10, 30), font, 0.7, (255, 50, 50), 2)

        if target_img_labeled is None:
            vis = np.hstack([source_img_labeled, pred_overlay, gt_overlay])
        else:
            vis = np.hstack([source_img_labeled, target_img_labeled, pred_overlay, gt_overlay])
        vis_rows.append(vis)

    if not vis_rows:
        return None

    # Stack samples vertically
    return np.vstack(vis_rows)
