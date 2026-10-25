# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 pose_encoding_type="absT_quaR_FoV",
                 enable_absolute_camera=False,
                 absolute_pose_encoding_type=None,
                 crop_posenc_dim=0,
                 crop_posenc_hidden_dim=256,
                 # Aggregator configuration (for smaller models)
                 aggregator_depth=24,
                 aggregator_num_heads=16,
                 aggregator_patch_embed="dinov2_vitl14_reg",
                 lora: dict = None):
        """
        Initialize VGGT model.

        Args:
            img_size: Input image size
            patch_size: Patch size for vision transformer
            embed_dim: Embedding dimension
            enable_camera: Whether to enable camera head
            enable_point: Whether to enable point head
            enable_depth: Whether to enable depth head
            enable_track: Whether to enable tracking head
            pose_encoding_type: Type of pose encoding for camera head:
                - "absT_quaR_FoV": Translation (3) + Quaternion (4) + FoV (2) = 9D
                - "absT_eulerR_FoV": Translation (3) + Euler angles (3) + FoV (2) = 8D
            aggregator_depth: Number of alternating attention layers (default 24)
            aggregator_num_heads: Number of attention heads (default 16)
            aggregator_patch_embed: Patch embedding type:
                - "dinov2_vitl14_reg": DINOv2 ViT-Large (embed_dim=1024, default)
                - "dinov2_vitb14_reg": DINOv2 ViT-Base (embed_dim=768)
                - "dinov2_vits14_reg": DINOv2 ViT-Small (embed_dim=384)
        """
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=aggregator_depth,
            num_heads=aggregator_num_heads,
            patch_embed=aggregator_patch_embed
        )

        if lora and lora.get("enabled", False):
            from vggt.layers.lora import apply_lora
            apply_lora(
                self.aggregator,
                rank=lora.get("rank", 8),
                alpha=lora.get("alpha", 16),
                dropout=lora.get("dropout", 0.0),
                target_patterns=lora.get("target_patterns"),
                exclude_patterns=lora.get("exclude_patterns"),
                freeze_base=lora.get("freeze_base", False),
            )

        self.camera_head = CameraHead(
            dim_in=2 * embed_dim,
            pose_encoding_type=pose_encoding_type,
            crop_posenc_dim=crop_posenc_dim,
            crop_posenc_hidden_dim=crop_posenc_hidden_dim,
        ) if enable_camera else None
        self.absolute_camera_head = CameraHead(
            dim_in=2 * embed_dim,
            pose_encoding_type=absolute_pose_encoding_type or pose_encoding_type,
            crop_posenc_dim=crop_posenc_dim,
            crop_posenc_hidden_dim=crop_posenc_hidden_dim,
        ) if enable_absolute_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, crop_params: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list, crop_params=crop_params)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.absolute_camera_head is not None:
                pose_enc_abs_list = self.absolute_camera_head(aggregated_tokens_list, crop_params=crop_params)
                predictions["pose_enc_abs"] = pose_enc_abs_list[-1]
                predictions["pose_enc_abs_list"] = pose_enc_abs_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            # The tracking branch is numerically less stable under AMP than the camera heads.
            # Keep it in fp32 so visibility/confidence and tracker gradients do not blow up.
            with torch.cuda.amp.autocast(enabled=False):
                track_tokens = [token.float() for token in aggregated_tokens_list]
                track_images = images.float()
                track_queries = query_points.float()
                track_list, vis, conf = self.track_head(
                    track_tokens,
                    images=track_images,
                    patch_start_idx=patch_start_idx,
                    query_points=track_queries,
                )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["track_list"] = track_list
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
