#!/usr/bin/env python3
"""Run official FaceLift inference."""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facelift-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--guidance-scale-2d", type=float, default=3.0)
    parser.add_argument("--step-2d", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.facelift_root))
    os.chdir(args.facelift_root)
    import inference

    def force_rgb(image):
        return image.convert("RGB") if hasattr(image, "convert") else image

    original_remove = inference.remove
    inference.remove = lambda *a, **kw: force_rgb(original_remove(*a, **kw))
    original_preprocess_image = inference.preprocess_image
    inference.preprocess_image = lambda *a, **kw: force_rgb(original_preprocess_image(*a, **kw))
    original_preprocess_without_crop = inference.preprocess_image_without_cropping
    inference.preprocess_image_without_cropping = (
        lambda *a, **kw: force_rgb(original_preprocess_without_crop(*a, **kw))
    )

    def skip_turntable(*args, **kwargs):
        images = args[0] if args else kwargs.get("images", [])
        try:
            count = len(images)
        except TypeError:
            count = "unknown"
        print(f"Skipping unused turntable video ({count} frames)")

    inference.imageseq2video = skip_turntable
    inference.main(
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        auto_crop=True,
        seed=args.seed,
        guidance_scale_2D=args.guidance_scale_2d,
        step_2D=args.step_2d,
    )


if __name__ == "__main__":
    main()
