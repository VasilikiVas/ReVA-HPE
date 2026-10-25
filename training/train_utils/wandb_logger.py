# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .distributed import get_machine_local_and_dist_rank


class WandBLogger:
    """A wrapper around Weights & Biases with distributed training support.

    This logger only writes from rank 0 in distributed settings to avoid conflicts.
    Automatically handles cleanup on exit.
    """

    def __init__(
        self,
        project: str = "reva-hpe",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        resume: bool = False,
        mode: str = "offline",
        dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name (optional, wandb will generate one if not provided)
            config: Configuration dictionary to log
            tags: List of tags for the run
            notes: Notes/description for the run
            resume: Whether to resume a previous run
            mode: WandB mode - "offline" (default, for nodes without internet),
                  "online", or "disabled"
            dir: Directory to store offline runs (defaults to ./wandb)
            **kwargs: Additional arguments passed to wandb.init
        """
        self._run = None
        _, self._rank = get_machine_local_and_dist_rank()

        if not WANDB_AVAILABLE:
            logging.warning("wandb not installed. Install with: pip install wandb")
            return

        if self._rank == 0:
            logging.info(f"Initializing WandB logger for project: {project} (mode={mode})")
            init_kwargs = {
                "project": project,
                "name": name,
                "config": config,
                "tags": tags,
                "notes": notes,
                "resume": "allow" if resume else False,
                "mode": mode,
            }
            if dir is not None:
                init_kwargs["dir"] = dir
            init_kwargs.update(kwargs)
            self._run = wandb.init(**init_kwargs)
            if mode == "offline":
                logging.info(f"WandB run initialized in OFFLINE mode. Sync later with: wandb sync {self._run.dir}")
            else:
                logging.info(f"WandB run initialized: {self._run.url}")
        else:
            logging.debug(
                f"Not logging on this process because rank {self._rank} != 0"
            )

        atexit.register(self.close)

    @property
    def run(self):
        """Get the underlying wandb run instance."""
        return self._run

    def flush(self) -> None:
        """Flush is a no-op for wandb (handled automatically)."""
        pass

    def close(self) -> None:
        """Finish the wandb run."""
        if self._run:
            self._run.finish()
            self._run = None

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        """Log multiple values to WandB.

        Args:
            payload: Dictionary mapping names to values
            step: Step value to record
        """
        if not self._run:
            return

        self._run.log(payload, step=step)

    def log(self, name: str, data: Any, step: int) -> None:
        """Log scalar data to WandB.

        Args:
            name: Metric name
            data: Scalar data to log (float/int/Tensor)
            step: Step value to record
        """
        if not self._run:
            return

        if isinstance(data, torch.Tensor):
            data = data.item()
        self._run.log({name: data}, step=step)

    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray, Any],
        step: int,
        fps: int = 4,
        caption: Optional[str] = None,
    ) -> None:
        """Log image or video data to WandB.

        Args:
            name: Tag name for the visual
            data: Image tensor (3D: C,H,W or H,W,C) or video tensor (5D: B,T,C,H,W)
            step: Step value to record
            fps: Frames per second for video data
            caption: Optional caption for the image/video

        Raises:
            ValueError: If data dimensions are not supported
        """
        if not self._run:
            return

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 3:
            # Image: (C, H, W) -> (H, W, C) for wandb
            if data.shape[0] in [1, 3, 4]:  # CHW format
                data = np.transpose(data, (1, 2, 0))
            self._run.log({name: wandb.Image(data, caption=caption)}, step=step)
        elif data.ndim == 4:
            # Video without batch: (T, C, H, W) -> (T, H, W, C)
            if data.shape[1] in [1, 3, 4]:  # TCHW format
                data = np.transpose(data, (0, 2, 3, 1))
            self._run.log({name: wandb.Video(data, fps=fps, caption=caption)}, step=step)
        elif data.ndim == 5:
            # Video with batch: (B, T, C, H, W) - take first sample
            data = data[0]  # (T, C, H, W)
            if data.shape[1] in [1, 3, 4]:
                data = np.transpose(data, (0, 2, 3, 1))
            self._run.log({name: wandb.Video(data, fps=fps, caption=caption)}, step=step)
        else:
            raise ValueError(
                f"Unsupported data dimensions: {data.ndim}. "
                "Expected 3D for images, 4D/5D for videos."
            )

    def log_image_grid(
        self,
        name: str,
        images: list,
        step: int,
        captions: Optional[list] = None,
    ) -> None:
        """Log a grid of images to WandB.

        Args:
            name: Tag name for the image grid
            images: List of image arrays or tensors
            step: Step value to record
            captions: Optional list of captions for each image
        """
        if not self._run:
            return

        wandb_images = []
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            caption = captions[i] if captions and i < len(captions) else None
            wandb_images.append(wandb.Image(img, caption=caption))

        self._run.log({name: wandb_images}, step=step)

    def log_table(
        self,
        name: str,
        columns: list,
        data: list,
        step: int,
    ) -> None:
        """Log a table to WandB.

        Args:
            name: Table name
            columns: List of column names
            data: List of rows (each row is a list of values)
            step: Step value to record
        """
        if not self._run:
            return

        table = wandb.Table(columns=columns, data=data)
        self._run.log({name: table}, step=step)
