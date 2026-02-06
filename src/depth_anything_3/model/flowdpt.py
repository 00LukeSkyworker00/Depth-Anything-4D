# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict as TyDict
from typing import List, Sequence
import torch
import torch.nn as nn

from depth_anything_3.model.dpt import DPT
from depth_anything_3.model.utils.head_utils import custom_interpolate


class FlowDPT(DPT):
    """
    DPT variant specialized for optical flow estimation.
    Extends DPT with image context fusion and flow-specific activation.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 2,  # u, v flow channels
        activation: str = "linear",
        conf_activation: str = "linear",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        use_image_context: bool = True,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
    ) -> None:
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            head_name="opticflow",
            use_sky_head=False,
            norm_type=norm_type,
            fusion_block_inplace=fusion_block_inplace,
        )
        self.use_image_context = use_image_context

        if use_image_context:
            merger_out_dim = features if feature_only else features // 2
            self.images_merger = nn.Sequential(
                nn.Conv2d(3, merger_out_dim // 4, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(merger_out_dim // 4, merger_out_dim // 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(merger_out_dim // 2, merger_out_dim, 3, 1, 1),
                nn.GELU(),
            )

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        images: torch.Tensor = None,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]  # [B*S, N_patch, C]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # Align scale
            resized_feats.append(x)

        # Fusion pyramid
        fused = self._fuse(resized_feats)
        fused = self.scratch.output_conv1(fused)

        # Upsample to target resolution
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)

        # Optionally inject image context (use first frame of pair for spatial context)
        if self.use_image_context and images is not None:
            fused = fused + self.images_merger(images)

        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        # Flow head
        feat = fused
        main_logits = self.scratch.output_conv2(feat)
        
        outs: TyDict[str, torch.Tensor] = {}
        # Apply activation to flow output
        flow_output = self._apply_activation_single(main_logits, self.activation).squeeze(1)
        outs[self.head_main] = flow_output

        return outs
