# Contains code derived from ByteDance Ltd. software
# licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import os
from pathlib import Path

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "da3-large", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name

        # Build the underlying network
        config_pth = Path(MODEL_REGISTRY[self.model_name])
        config_4d = f"{config_pth.stem}-4d.yaml"
        custom_pth = config_pth.with_name(config_4d)
        if os.path.exists(custom_pth):
            config_pth = custom_pth
            print(f"Using custom config {config_4d}")
        else:
            print(f"Custom config {config_4d} not found!!!")
        self.config = load_config(str(config_pth))
        self.model = create_object(self.config)

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.
            infer_gs: Enable Gaussian Splatting branch.
            use_ray_pose: Use ray-based pose estimation instead of camera decoder.
            ref_view_strategy: Strategy for selecting reference view from multiple views.

        Returns:
            Dictionary containing model predictions
        """
        extrinsics = self._normalize_extrinsics(extrinsics).detach()

        # Determine optimal autocast dtype
        # autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
        #     return self.model(
        #         image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
        #     )
        output = self.model(
            image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
        )
        
        return output
    
    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)

