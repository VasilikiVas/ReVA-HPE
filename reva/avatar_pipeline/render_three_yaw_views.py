#!/usr/bin/env python3
"""Render three controlled FaceLift views and record their camera matrices."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facelift-root", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--radius", type=float, default=2.7)
    parser.add_argument("--axis", choices=("yaw", "pitch", "roll"), default="yaw")
    return parser.parse_args()


def make_c2w(camera_azimuth_deg, radius, camera_elevation_deg=0.0):
    azimuth = np.deg2rad(270.0 + camera_azimuth_deg)
    elevation = np.deg2rad(camera_elevation_deg)
    base_radius = radius * np.cos(elevation)
    camera_position = np.array(
        [
            base_radius * np.cos(azimuth),
            base_radius * np.sin(azimuth),
            radius * np.sin(elevation),
        ],
        dtype=np.float64,
    )
    forward = -camera_position / np.linalg.norm(camera_position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # FaceLift uses OpenCV camera axes: +x right, +y down, +z forward.
    rotation_c2w = np.stack((right, -up, forward), axis=1)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation_c2w
    c2w[:3, 3] = camera_position
    return c2w


def render_image(model, render_opencv_cam, c2w, intrinsics, resolution):
    device = model._xyz.device
    result = render_opencv_cam(
        model,
        resolution,
        resolution,
        torch.from_numpy(c2w.astype(np.float32)).to(device),
        torch.from_numpy(intrinsics.astype(np.float32)).to(device),
    )["render"]
    array = (
        result.detach()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .clip(0.0, 1.0)
    )
    return Image.fromarray((array * 255.0).astype(np.uint8))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.facelift_root))

    from gslrm.model.gaussians_renderer import GaussianModel, render_opencv_cam

    camera_path = args.facelift_root / "utils_folder" / "opencv_cameras.json"
    camera_data = json.loads(camera_path.read_text())["frames"]
    front_camera = camera_data[2]
    scale = args.resolution / float(front_camera["w"])
    intrinsics = np.array(
        [
            front_camera["fx"] * scale,
            front_camera["fy"] * scale,
            front_camera["cx"] * scale,
            front_camera["cy"] * scale,
        ],
        dtype=np.float64,
    )

    model = GaussianModel(sh_degree=3)
    model.load_ply(str(args.ply))
    model.to("cuda")

    front_c2w = make_c2w(0.0, args.radius)
    records = []
    panels = []
    for offset in (-20.0, 0.0, 20.0):
        camera_azimuth = offset if args.axis == "yaw" else 0.0
        camera_elevation = offset if args.axis == "pitch" else 0.0
        if args.axis == "roll":
            angle = np.deg2rad(offset)
            head_to_camera_rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = head_to_camera_rotation @ front_c2w[:3, :3].T
            w2c[:3, 3] = np.array([0.0, 0.0, args.radius])
            c2w = np.linalg.inv(w2c)
        else:
            c2w = make_c2w(camera_azimuth, args.radius, camera_elevation)
        w2c = np.linalg.inv(c2w)

        head_to_camera = np.eye(4, dtype=np.float64)
        head_to_camera[:3, :3] = w2c[:3, :3] @ front_c2w[:3, :3]
        head_to_camera[:3, 3] = w2c[:3, 3]

        name = f"camera_{args.axis}_{offset:+.0f}deg"
        image = render_image(
            model, render_opencv_cam, c2w, intrinsics, args.resolution
        )
        image.save(args.output_dir / f"{name}.png")
        panels.append((name, image))
        records.append(
            {
                "name": name,
                "camera_azimuth_deg": camera_azimuth,
                "camera_elevation_deg": camera_elevation,
                "requested_head_roll_deg": offset if args.axis == "roll" else 0.0,
                "intrinsics_fxfycxcy": intrinsics.tolist(),
                "c2w_opencv": c2w.tolist(),
                "w2c_opencv": w2c.tolist(),
                "head_to_camera_front_canonical": head_to_camera.tolist(),
            }
        )

    label_height = 42
    montage = Image.new(
        "RGB",
        (args.resolution * len(panels), args.resolution + label_height),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    for index, (name, image) in enumerate(panels):
        x = index * args.resolution
        montage.paste(image, (x, label_height))
        draw.text((x + 12, 13), name, fill="black")
    montage_path = args.output_dir / f"three_{args.axis}_views.png"
    montage.save(montage_path)

    metadata = {
        "ply": str(args.ply),
        "resolution": args.resolution,
        "radius_facelift_units": args.radius,
        "axis": args.axis,
        "coordinate_note": (
            "Camera offsets are measured from FaceLift's frontal camera at "
            "azimuth 270 degrees and elevation 0 degrees. "
        ),
        "views": records,
    }
    cameras_path = args.output_dir / f"{args.axis}_cameras.json"
    cameras_path.write_text(json.dumps(metadata, indent=2))
    print(montage_path)
    print(cameras_path)


if __name__ == "__main__":
    main()
