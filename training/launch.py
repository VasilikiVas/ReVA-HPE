# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config",
        type=str,
        default="flame_fromvggt_largepairs_fixed_fov_unified_crop_depth_norm_delta_t_rot1_vggtinit",
        help="Config file name without the .yaml extension (default: the paper model)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Hydra overrides, e.g. exp_name=my_run_123"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


