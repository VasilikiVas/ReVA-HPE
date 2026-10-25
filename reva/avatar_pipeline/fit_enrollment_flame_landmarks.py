#!/usr/bin/env python3
"""Fit FLAME to one enrollment RGB and export canonical metric landmarks."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--blended-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rigid-iters", type=int, default=250)
    parser.add_argument("--full-iters", type=int, default=400)
    return parser.parse_args()


def load_trial_module(blended_root):
    script = blended_root / "trial_photo_to_5views.py"
    spec = importlib.util.spec_from_file_location("trial_photo_to_5views", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_list(value):
    return value.detach().cpu().numpy().tolist()


def canonical_landmarks(flame, shape, expression):
    pose = torch.zeros((1, 6), dtype=shape.dtype, device=shape.device)
    with torch.no_grad():
        vertices, landmarks, _, _ = flame(
            shape_params=shape,
            expression_params=expression,
            pose_params=pose,
        )
    # FLAME: +X right, +Y up, +Z back. Convert to OpenCV-style canonical axes.
    flame_to_opencv = torch.diag(
        torch.tensor([1.0, -1.0, -1.0], dtype=shape.dtype, device=shape.device)
    )
    landmarks_opencv = landmarks[0] @ flame_to_opencv.T
    vertices_opencv = vertices[0] @ flame_to_opencv.T
    return landmarks[0], landmarks_opencv, vertices_opencv


def plot_result(image, detected, projected, references, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].scatter(detected[:, 0], detected[:, 1], s=10, c="lime", label="detected")
    axes[0].scatter(projected[:, 0], projected[:, 1], s=9, c="red", label="FLAME fit")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("Enrollment landmark fit")
    axes[0].axis("off")

    colors = {
        "person_neutral": "tab:blue",
        "person_expression": "tab:orange",
        "mean_neutral": "tab:green",
    }
    for name, points in references.items():
        axes[1].scatter(points[:, 0], points[:, 1], s=12, alpha=0.75,
                        color=colors[name], label=name)
        axes[2].scatter(points[:, 2], points[:, 1], s=12, alpha=0.75,
                        color=colors[name], label=name)
    axes[1].set_xlabel("X right (m)")
    axes[1].set_ylabel("Y down (m)")
    axes[1].set_title("Canonical front")
    axes[2].set_xlabel("Z forward (m)")
    axes[2].set_ylabel("Y down (m)")
    axes[2].set_title("Canonical side")
    for axis in axes[1:]:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    trial = load_trial_module(args.blended_root)
    image = trial.load_image(args.image)
    landmarks = trial.detect_landmarks_68(image, args.landmarks)
    device = torch.device(args.device)
    flame, config = trial.build_flame(device, args.blended_root / "assets")
    fit = trial.fit_flame(
        flame,
        config,
        landmarks,
        device,
        iters=(args.rigid_iters, args.full_iters),
    )

    zero_shape = torch.zeros_like(fit["shape"])
    zero_expression = torch.zeros_like(fit["expr"])
    reference_tensors = {}
    for name, shape, expression in (
        ("person_neutral", fit["shape"], zero_expression),
        ("person_expression", fit["shape"], fit["expr"]),
        ("mean_neutral", zero_shape, zero_expression),
    ):
        landmarks_flame, landmarks_opencv, vertices_opencv = canonical_landmarks(
            flame, shape, expression
        )
        reference_tensors[name] = {
            "landmarks_flame": landmarks_flame,
            "landmarks_opencv": landmarks_opencv,
            "vertices_opencv": vertices_opencv,
        }

    projected = fit["lmk_proj"].cpu().numpy()
    residuals = np.linalg.norm(projected - landmarks, axis=1)
    references = {
        name: tensors["landmarks_opencv"].cpu().numpy()
        for name, tensors in reference_tensors.items()
    }
    output = {
        "source_image": str(args.image),
        "uses_dataset_pose_or_depth_ground_truth": False,
        "coordinate_conventions": {
            "flame": "x-right, y-up, z-back",
            "opencv_canonical": "x-right, y-down, z-forward",
            "flame_to_opencv": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            "length_unit": "metres (FLAME anthropometric model prior)",
        },
        "fit": {
            "detected_landmarks_xy": landmarks.tolist(),
            "projected_landmarks_xy": projected.tolist(),
            "residual_px_by_landmark": residuals.tolist(),
            "median_residual_px": float(np.median(residuals)),
            "p95_residual_px": float(np.percentile(residuals, 95)),
            "weak_perspective_scale": float(fit["scale"].cpu().item()),
            "weak_perspective_translation_xy": tensor_list(fit["trans"]),
            "estimated_global_pose_axis_angle": tensor_list(fit["global_pose"]),
            "estimated_jaw_pose_axis_angle": tensor_list(fit["jaw_pose"]),
            "shape_coefficients": tensor_list(fit["shape"]),
            "expression_coefficients": tensor_list(fit["expr"]),
        },
        "canonical_references": {},
    }
    for name, tensors in reference_tensors.items():
        points = tensors["landmarks_opencv"].cpu().numpy()
        vertices = tensors["vertices_opencv"].cpu().numpy()
        output["canonical_references"][name] = {
            "landmarks_xyz_flame": tensor_list(tensors["landmarks_flame"]),
            "landmarks_xyz_opencv": points.tolist(),
            "landmark_minimum_xyz_opencv": points.min(axis=0).tolist(),
            "landmark_maximum_xyz_opencv": points.max(axis=0).tolist(),
            "vertex_minimum_xyz_opencv": vertices.min(axis=0).tolist(),
            "vertex_maximum_xyz_opencv": vertices.max(axis=0).tolist(),
        }

    json_path = args.output_dir / "canonical_flame_landmarks.json"
    json_path.write_text(json.dumps(output, indent=2))
    plot_path = args.output_dir / "canonical_flame_landmarks.png"
    plot_result(image, landmarks, projected, references, plot_path)
    Image.fromarray(image).save(args.output_dir / "enrollment_rgb.png")
    print(json_path)
    print(plot_path)
    print(json.dumps({
        "median_residual_px": output["fit"]["median_residual_px"],
        "p95_residual_px": output["fit"]["p95_residual_px"],
        "person_neutral_landmark_bounds_m": {
            "minimum": output["canonical_references"]["person_neutral"]["landmark_minimum_xyz_opencv"],
            "maximum": output["canonical_references"]["person_neutral"]["landmark_maximum_xyz_opencv"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
