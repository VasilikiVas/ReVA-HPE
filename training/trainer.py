# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers
from train_utils.wandb_logger import WandBLogger


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        if not self.checkpoint_conf.strict:
            current_state_dict = self.model.state_dict()
            filtered_state_dict = {}
            remapped_lora_base_keys = []
            skipped_keys = []
            for key, value in model_state_dict.items():
                target_key = key
                if target_key not in current_state_dict and (key.endswith(".weight") or key.endswith(".bias")):
                    module_key, param_name = key.rsplit(".", 1)
                    lora_base_key = f"{module_key}.base.{param_name}"
                    if lora_base_key in current_state_dict:
                        target_key = lora_base_key
                        remapped_lora_base_keys.append((key, target_key))
                if target_key not in current_state_dict:
                    skipped_keys.append(key)
                    continue
                if current_state_dict[target_key].shape != value.shape:
                    skipped_keys.append(key)
                    continue
                filtered_state_dict[target_key] = value
            model_state_dict = filtered_state_dict
            if self.rank == 0 and remapped_lora_base_keys:
                logging.info(
                    "Remapped %d checkpoint tensors into LoRA base modules.",
                    len(remapped_lora_base_keys),
                )
            if self.rank == 0 and skipped_keys:
                logging.info(
                    "Skipping %d checkpoint keys with missing/mismatched shapes: %s",
                    len(skipped_keys),
                    skipped_keys,
                )
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
        self._maybe_initialize_absolute_camera_from_camera_head(missing)

        # Load optimizer state if available and in training mode
        load_optimizer_state = getattr(self.checkpoint_conf, "load_optimizer_state", True)
        if "optimizer" in checkpoint and load_optimizer_state:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            optimizer_state = checkpoint["optimizer"]
            if isinstance(self.optims, (list, tuple)):
                if len(self.optims) == 1 and isinstance(optimizer_state, dict):
                    self.optims[0].optimizer.load_state_dict(optimizer_state)
                else:
                    assert isinstance(optimizer_state, (list, tuple)), (
                        "Expected a list of optimizer states when multiple optimizers are configured."
                    )
                    assert len(optimizer_state) == len(self.optims), (
                        f"Checkpoint has {len(optimizer_state)} optimizer states, "
                        f"but config has {len(self.optims)} optimizers."
                    )
                    for optim, state in zip(self.optims, optimizer_state):
                        optim.optimizer.load_state_dict(state)
            else:
                self.optims.optimizer.load_state_dict(optimizer_state)
        elif "optimizer" in checkpoint and self.rank == 0:
            logging.info("Skipping optimizer state dict because checkpoint.load_optimizer_state=False.")

        # Load training progress. Newer checkpoints store the just-finished
        # epoch as prev_epoch, so resume from the following epoch.
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        elif "prev_epoch" in checkpoint:
            self.epoch = int(checkpoint["prev_epoch"]) + 1
        if self.rank == 0:
            logging.info(f"Resume epoch set to {self.epoch}.")
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _maybe_initialize_absolute_camera_from_camera_head(self, missing_keys: Sequence[str]) -> None:
        """Optionally copy the loaded camera head into a newly-added absolute head."""
        if not getattr(self.checkpoint_conf, "init_missing_absolute_camera_from_camera_head", False):
            return
        if not hasattr(self.model, "camera_head") or not hasattr(self.model, "absolute_camera_head"):
            return
        if self.model.camera_head is None or self.model.absolute_camera_head is None:
            return
        if not any(key.startswith("absolute_camera_head.") for key in missing_keys):
            if self.rank == 0:
                logging.info(
                    "Skipping absolute camera init from camera head because checkpoint already had absolute head weights."
                )
            return

        source_state = self.model.camera_head.state_dict()
        target_state = self.model.absolute_camera_head.state_dict()
        copied_state = {}
        skipped_keys = []
        for key, value in source_state.items():
            if key in target_state and target_state[key].shape == value.shape:
                copied_state[key] = value.detach().clone()
            else:
                skipped_keys.append(key)

        target_state.update(copied_state)
        self.model.absolute_camera_head.load_state_dict(target_state, strict=False)
        if self.rank == 0:
            logging.info(
                "Initialized absolute_camera_head from camera_head: copied %d tensors, skipped %d tensors%s.",
                len(copied_state),
                len(skipped_keys),
                f" ({skipped_keys})" if skipped_keys else "",
            )

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)

        # Initialize WandB logger if enabled
        self.wandb_logger = None
        if getattr(self.logging_conf, "wandb", None) and self.logging_conf.wandb.get("enabled", False):
            from omegaconf import OmegaConf
            try:
                wandb_conf = self.logging_conf.wandb
                wandb_kwargs = {
                    "project": wandb_conf.get("project", "reva-hpe"),
                    "name": wandb_conf.get("name", None),
                    "tags": list(wandb_conf.get("tags", [])) if wandb_conf.get("tags") else None,
                    "notes": wandb_conf.get("notes", None),
                    "resume": wandb_conf.get("resume", False),
                    "mode": wandb_conf.get("mode", "offline"),  # Default to offline for HPC
                    "dir": wandb_conf.get("dir", None),
                }
                if wandb_conf.get("id", None) is not None:
                    wandb_kwargs["id"] = wandb_conf.get("id")
                # Try to add config, but don't fail if it errors
                try:
                    wandb_kwargs["config"] = {
                        "exp_name": getattr(self, "exp_name", None),
                        "max_epochs": self.max_epochs,
                        "img_size": getattr(self.model_conf, "img_size", 518),
                    }
                except Exception:
                    pass
                self.wandb_logger = WandBLogger(**wandb_kwargs)
            except Exception as e:
                logging.warning(f"Failed to initialize WandB logger: {e}")

        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            batch_size = batch["images"].shape[0]
            if batch_size == 0:
                logging.warning("Skipping empty batch.")
                continue

            accum_steps = min(self.accum_steps, batch_size)

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
            "crop_params",
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        camera_conf = getattr(self.loss_conf, "camera", None)
        if isinstance(camera_conf, Mapping):
            translation_target_type = camera_conf.get("translation_target_type", "relative_se3")
            absolute_anchor_supervision = camera_conf.get("absolute_anchor_supervision", False)
            absolute_pose_supervision = camera_conf.get("absolute_pose_supervision", False)
        else:
            translation_target_type = getattr(camera_conf, "translation_target_type", "relative_se3")
            absolute_anchor_supervision = getattr(camera_conf, "absolute_anchor_supervision", False)
            absolute_pose_supervision = getattr(camera_conf, "absolute_pose_supervision", False)
        needs_absolute_extrinsics = (
            translation_target_type in {"delta_t", "depth_norm_delta_t", "projective_delta_t"}
            or absolute_anchor_supervision
            or absolute_pose_supervision
        )
        if needs_absolute_extrinsics and "extrinsics" in batch:
            batch["absolute_extrinsics"] = batch["extrinsics"].clone()

        # Check if scale normalization should be disabled (e.g., for H2C training without depth)
        scale_by_points = getattr(self.data_conf.train.common_config, 'scale_by_points', True)
        normalize_to_first_frame = getattr(
            self.data_conf.train.common_config,
            "normalize_to_first_frame",
            True,
        )

        if not normalize_to_first_frame:
            if scale_by_points:
                logging.warning(
                    "normalize_to_first_frame=False with scale_by_points=True is not supported; "
                    "leaving the batch untouched."
                )
            return batch

        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
                scale_by_points=scale_by_points,
            )

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _get_phase_common_config(self, phase: str):
        """Return the per-phase common data config used for preprocessing and visualization."""
        phase_conf = getattr(self.data_conf, phase, None)
        if phase_conf is None:
            phase_conf = self.data_conf.train
        return getattr(phase_conf, "common_config", None)

    def _get_supervise_frame_idxs(self) -> Optional[List[int]]:
        """Return the frame indices that receive direct camera supervision, if configured."""
        camera_conf = getattr(self.loss_conf, "camera", None)
        pose_supervise_frame_idxs = getattr(camera_conf, "pose_supervise_frame_idxs", None)
        if pose_supervise_frame_idxs is not None:
            return [int(idx) for idx in pose_supervise_frame_idxs]
        supervise_frame_idxs = getattr(camera_conf, "supervise_frame_idxs", None)
        if supervise_frame_idxs is None:
            return None
        return [int(idx) for idx in supervise_frame_idxs]

    def _get_pose_visualization_mode(self, phase: str) -> str:
        """Infer how to decode pose visualizations from the current train/val setup."""
        common_config = self._get_phase_common_config(phase)
        normalize_to_first_frame = getattr(common_config, "normalize_to_first_frame", True)
        fix_img_num = getattr(common_config, "fix_img_num", -1)
        supervise_frame_idxs = self._get_supervise_frame_idxs()

        if not normalize_to_first_frame and fix_img_num == 1 and supervise_frame_idxs == [0]:
            return "absolute_single"

        if not normalize_to_first_frame and supervise_frame_idxs == [1]:
            return "absolute_second"

        return "relative_h2c"

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        query_points = None
        if "tracks" in batch:
            query_points = batch["tracks"][:, 0]
        y_hat = model(
            images=batch["images"],
            query_points=query_points,
            crop_params=batch.get("crop_params"),
        )
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])
        self._log_pose_visuals(log_data, phase, self.steps[phase])
        self._log_track_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _log_pose_visuals(self, data: Mapping, phase: str, step: int) -> None:
        """Logs pose comparison visualizations to WandB/TensorBoard and/or saves locally."""
        # Check if pose visualization is enabled and it's time to log
        visual_freq = getattr(self.logging_conf, "pose_visual_frequency", {})
        freq = visual_freq.get(phase, 0)
        if freq <= 0 or step % freq != 0:
            return

        # Check if we have the required data
        # Model returns "pose_enc_list" (list of per-stage predictions), not "pose_enc"
        has_pred = "pose_enc_list" in data or "pose_enc" in data
        if not has_pred or "pose_encoding" not in data or "images" not in data:
            return

        # Use last stage prediction from pose_enc_list, or fall back to pose_enc
        if "pose_enc_list" in data:
            pred_pose = data["pose_enc_list"][-1].detach()
        else:
            pred_pose = data["pose_enc"].detach()

        # Get pose encoding type from loss config
        pose_encoding_type = getattr(self.loss_conf.camera, "pose_encoding_type", "absT_quaR_FoV")
        gt_pose = data["pose_encoding"].detach()
        visual_pose_type = pose_encoding_type
        if pose_encoding_type == "quatR" and data.get("pose_encoding_visual") is not None:
            gt_pose_visual = data["pose_encoding_visual"].detach()
            pred_pose = torch.cat(
                [gt_pose_visual[..., :3], pred_pose, gt_pose_visual[..., 7:9]],
                dim=-1,
            )
            gt_pose = gt_pose_visual
            visual_pose_type = "absT_quaR_FoV"
        elif pose_encoding_type == "absT_quaR" and data.get("pose_encoding_visual") is not None:
            gt_pose_visual = data["pose_encoding_visual"].detach()
            pred_pose = torch.cat(
                [pred_pose, gt_pose_visual[..., 7:9]],
                dim=-1,
            )
            gt_pose = gt_pose_visual
            visual_pose_type = "absT_quaR_FoV"
        supervise_frame_idxs = self._get_supervise_frame_idxs()
        pose_mode = self._get_pose_visualization_mode(phase)

        # Import visualization functions
        from train_utils.pose_visualization import create_batch_visualization, create_mesh_batch_visualization

        save_locally = getattr(self.logging_conf, "save_visuals_locally", False)
        visuals_dir = getattr(self.logging_conf, "visuals_save_dir", "visualizations")

        # Create text-based pose comparison visualization
        try:
            vis = create_batch_visualization(
                images=data["images"].detach(),
                pred_pose=pred_pose,
                gt_pose=gt_pose,
                pose_encoding_type=visual_pose_type,
                max_samples=2,
                supervise_frame_idxs=supervise_frame_idxs,
            )

            if save_locally and self.rank == 0:
                from PIL import Image
                save_dir = os.path.join(self.logging_conf.log_dir, visuals_dir, phase)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"pose_step_{step:06d}.png")
                Image.fromarray(vis).save(save_path)

            logger = self.wandb_logger if self.wandb_logger else self.tb_writer
            if logger:
                import numpy as np
                vis_chw = np.transpose(vis, (2, 0, 1))
                logger.log_visuals(f"{phase}/pose_comparison", vis_chw, step)

        except Exception as e:
            logging.warning(f"Failed to log pose visualization: {e}")

        # Create mesh overlay visualization if mesh_paths available
        if "mesh_paths" in data and "intrinsics" in data:
            try:
                mesh_vis = create_mesh_batch_visualization(
                    images=data["images"].detach(),
                    pred_pose=pred_pose,
                    gt_pose=gt_pose,
                    mesh_paths=data["mesh_paths"],
                    intrinsics=data["intrinsics"].detach(),
                    pose_encoding_type=visual_pose_type,
                    pose_mode=pose_mode,
                    device=self.device,
                    max_samples=1,
                    view_paths=data.get("view_paths"),
                    translation_target_type=getattr(
                        self.loss_conf.camera,
                        "translation_target_type",
                        "relative_se3",
                    ),
                    absolute_extrinsics=(
                        data["absolute_extrinsics"].detach()
                        if isinstance(data.get("absolute_extrinsics"), torch.Tensor)
                        else data.get("absolute_extrinsics")
                    ),
                )

                if mesh_vis is not None:
                    if save_locally and self.rank == 0:
                        from PIL import Image
                        save_dir = os.path.join(self.logging_conf.log_dir, visuals_dir, phase, "mesh")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"mesh_step_{step:06d}.png")
                        Image.fromarray(mesh_vis).save(save_path)

                    logger = self.wandb_logger if self.wandb_logger else self.tb_writer
                    if logger:
                        import numpy as np
                        mesh_vis_chw = np.transpose(mesh_vis, (2, 0, 1))
                        logger.log_visuals(f"{phase}/mesh_overlay", mesh_vis_chw, step)

            except Exception as e:
                logging.warning(f"Failed to log mesh visualization: {e}")

    def _log_track_visuals(self, data: Mapping, phase: str, step: int) -> None:
        """Log predicted vs GT track overlays to WandB/TensorBoard and optionally save them locally."""
        visual_freq = getattr(self.logging_conf, "track_visual_frequency", {})
        freq = visual_freq.get(phase, 0)
        if freq <= 0 or step % freq != 0:
            return

        if "track" not in data or "tracks" not in data or "images" not in data:
            return
        if data["track"] is None or data["tracks"] is None:
            return

        try:
            import cv2
            import numpy as np

            pred_tracks = data["track"].detach().cpu().float()
            gt_tracks = data["tracks"].detach().cpu().float()
            images = data["images"].detach().cpu().float()

            b = 0
            _, seq_len, num_tracks, _ = pred_tracks.shape

            imgs_np = []
            for frame_idx in range(seq_len):
                img = images[b, frame_idx].permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                imgs_np.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            height, width = imgs_np[0].shape[:2]
            gt_canvas = [img.copy() for img in imgs_np]
            pred_canvas = [img.copy() for img in imgs_np]

            vis_mask = data.get("track_vis_mask")
            if vis_mask is not None:
                vis_mask = vis_mask.detach().cpu()
                valid = vis_mask[b, 0].bool()
            else:
                valid = torch.ones(num_tracks, dtype=torch.bool)

            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            max_tracks = 200
            if len(valid_idx) > max_tracks:
                perm = torch.randperm(len(valid_idx))[:max_tracks]
                valid_idx = valid_idx[perm]

            colors = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (180, 0, 255)]

            for track_idx in valid_idx.tolist():
                color = colors[track_idx % len(colors)]
                for frame_idx in range(seq_len - 1):
                    x0, y0 = gt_tracks[b, frame_idx, track_idx].tolist()
                    x1, y1 = gt_tracks[b, frame_idx + 1, track_idx].tolist()
                    if 0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height:
                        cv2.line(gt_canvas[frame_idx], (int(x0), int(y0)), (int(x1), int(y1)), color, 1)
                        cv2.line(gt_canvas[frame_idx + 1], (int(x0), int(y0)), (int(x1), int(y1)), color, 1)

                    px0, py0 = pred_tracks[b, frame_idx, track_idx].tolist()
                    px1, py1 = pred_tracks[b, frame_idx + 1, track_idx].tolist()
                    if 0 <= px0 < width and 0 <= py0 < height and 0 <= px1 < width and 0 <= py1 < height:
                        cv2.line(pred_canvas[frame_idx], (int(px0), int(py0)), (int(px1), int(py1)), color, 1)
                        cv2.line(pred_canvas[frame_idx + 1], (int(px0), int(py0)), (int(px1), int(py1)), color, 1)

                for frame_idx in range(seq_len):
                    x, y = gt_tracks[b, frame_idx, track_idx].tolist()
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(gt_canvas[frame_idx], (int(x), int(y)), 3, color, -1)

                    px, py = pred_tracks[b, frame_idx, track_idx].tolist()
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(pred_canvas[frame_idx], (int(px), int(py)), 3, color, -1)

            for frame_idx in range(seq_len):
                cv2.putText(
                    gt_canvas[frame_idx],
                    f"GT   v{frame_idx}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    pred_canvas[frame_idx],
                    f"Pred v{frame_idx}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            row_gt = np.concatenate(gt_canvas, axis=1)
            row_pred = np.concatenate(pred_canvas, axis=1)
            combined = np.concatenate([row_gt, row_pred], axis=0)
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

            save_locally = getattr(self.logging_conf, "save_visuals_locally", False)
            visuals_dir = getattr(self.logging_conf, "visuals_save_dir", "visualizations")

            if save_locally and self.rank == 0:
                from PIL import Image

                save_dir = os.path.join(self.logging_conf.log_dir, visuals_dir, phase, "tracks")
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(combined_rgb).save(
                    os.path.join(save_dir, f"tracks_step_{step:06d}.png")
                )

            logger = self.wandb_logger if self.wandb_logger else self.tb_writer
            if logger:
                vis_chw = np.transpose(combined_rgb, (2, 0, 1))
                logger.log_visuals(f"{phase}/track_comparison", vis_chw, step)

        except Exception as e:
            logging.warning(f"Failed to log track visualization: {e}")

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard and WandB."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]

        wandb_payload = {}
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
                    wandb_payload[f"{phase}/{key}"] = value

        # Log to WandB in a single call for efficiency
        if self.wandb_logger and wandb_payload and step % self.logging_conf.log_freq == 0:
            self.wandb_logger.log_dict(wandb_payload, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        log_visual_freq = getattr(self.logging_conf, "log_visual_frequency", {})
        visuals_keys = getattr(self.logging_conf, "visuals_keys_to_log", None)
        if not (
            getattr(self.logging_conf, "log_visuals", False)
            and log_visual_freq
            and (phase in log_visual_freq)
            and log_visual_freq[phase] > 0
            and (step % log_visual_freq[phase] == 0)
            and (visuals_keys is not None)
        ):
            return

        if phase in visuals_keys:
            keys_to_log = visuals_keys[phase]["keys_to_log"]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = visuals_keys[phase]["modality"]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def is_sequence_of_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of sequences of primitive types."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and is_sequence_of_primitives(data[0])
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif is_sequence_of_sequence_of_primitives(data):
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
