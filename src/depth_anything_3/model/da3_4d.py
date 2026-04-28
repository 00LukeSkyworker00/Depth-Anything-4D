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

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
from depth_anything_3.specs import Gaussians

from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
# from fused_ssim import fused_ssim

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)

class DepthAnything3Net(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14
    DEFAULT_RASTERIZE_CFG = {
        "chunk_size": 1,
        "area_target": 1.0,
        "area_tau": 0.5,
        "spread_target": 0.03,
        "contrib_min": 1e-4,
        "contrib_max": 1e-2,
    }

    def __init__(
        self,
        net,
        head,
        raft_head,
        scene_head,
        cam_dec=None,
        cam_enc=None,
        gs_head=None,
        gs_adapter=None,
        rasterize_cfg=None,
    ):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_enc if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        if gs_head is not None and gs_adapter is not None:
            self.gs_adapter = (
                gs_adapter
                if isinstance(gs_adapter, nn.Module)
                else create_object(_wrap_cfg(gs_adapter))
            )
            gs_out_dim = self.gs_adapter.d_in + 1
            if isinstance(gs_head, nn.Module):
                assert (
                    gs_head.out_dim == gs_out_dim
                ), f"gs_head.out_dim should be {gs_out_dim}, got {gs_head.out_dim}"
                self.gs_head = gs_head
            else:
                assert (
                    gs_head["output_dim"] == gs_out_dim
                ), f"gs_head output_dim should set to {gs_out_dim}, got {gs_head['output_dim']}"
                self.gs_head = create_object(_wrap_cfg(gs_head))

        self.scene_head = scene_head if isinstance(scene_head, nn.Module) else create_object(_wrap_cfg(scene_head))
        # self.raft_head = raft_head if isinstance(raft_head, nn.Module) else create_object(_wrap_cfg(raft_head))
        self.vanilla_dino = None
        if isinstance(rasterize_cfg, DictConfig):
            rasterize_cfg = OmegaConf.to_container(rasterize_cfg, resolve=True)
        elif isinstance(rasterize_cfg, Dict):
            rasterize_cfg = dict(rasterize_cfg)
        elif rasterize_cfg is None:
            rasterize_cfg = {}

        unknown_keys = set(rasterize_cfg.keys()) - set(self.DEFAULT_RASTERIZE_CFG.keys())
        if len(unknown_keys) > 0:
            raise ValueError(f"Unknown keys in rasterize_cfg: {sorted(unknown_keys)}")

        self.rasterize_cfg = {**self.DEFAULT_RASTERIZE_CFG, **rasterize_cfg}
        # self.vanilla_dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        # self.vanilla_dino.eval()

        self.freeze()

    def freeze(self):
        """
        Freeze all parameters in the network (backbone, head, camera, gs modules).
        Any new layers added after freeze() will remain trainable.
        """
        freezed_modules = ""
        for module in [self.backbone, self.head, self.cam_dec, self.cam_enc, self.gs_adapter, self.gs_head, self.vanilla_dino]:
            if module is not None:
                freezed_modules += f"{module.__class__.__name__}, "
                for param in module.parameters():
                    param.requires_grad = False
        print(f"Frozen modules: {freezed_modules}")

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, S, 3, H, W)
            extrinsics: Camera extrinsics (B, S, 4, 4) 
            intrinsics: Camera intrinsics (B, S, 3, 3) 
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        # Extract features using backbone
        # print("Before Start   : ",vram())
        with torch.no_grad():
            B, S, C, H, W = x.shape

            if extrinsics is not None:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
            else:
                cam_token = None

            """
            feats: layers of (patch_tokens:(B,S,N,2C), cam_tokens:(B,S,2C))
            dino_feats: (cls_token:(B*S,C))
            """
            feats, aux_feats = self.backbone(
                x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
            )
            del aux_feats
            # dino_feats = self.vanilla_dino(x.flatten(0,1))
            # dino_feats = dino_feats.view(B,S,-1).detach()
            # feats = [[item.detach() for item in feat] for feat in feats]

            # print("After Backbone : ",vram())
            # Process features through depth head
            with torch.autocast(device_type=x.device.type, enabled=False):
                output = self._process_depth_head(feats, H, W)
                if use_ray_pose:
                    output = self._process_ray_pose_estimation(output, H, W)
                else:
                    output = self._process_camera_estimation(feats, H, W, output)
                # if infer_gs:
                #     output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)
            
                # output = self._process_mono_sky_estimation(output)

                # # Extract auxiliary features if requested
                # output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)
                
                
                torch.cuda.empty_cache()
            # print("After DPT : ",vram())
        
        output = self._process_scene_head(feats, H, W, output)
        
        # output = self._process_raft_head(output)        
        # output = self._process_flow_head(feats, H, W, output)

        return output

    def _process_scene_head(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process features to obtain scene-level representations."""
        feats_layers =tuple(torch.stack(t, dim=0) for t in zip(*feats))
        cam_layers = feats_layers[1]
        patch_layers = feats_layers[0]

        raw_gs, _ = self.scene_head(
            cam_layers = cam_layers, 
            patch_layers = patch_layers, 
            H = H, W = W,
            concat_gs = False
        )
        # output.scene = scene_token[-1]
        render_layers = False
        if isinstance(raw_gs, list):
            raw_gs_layers = raw_gs
            raw_gs = sum(raw_gs[1:], raw_gs[0])
            output.gs_layers = raw_gs_layers
            render_layers = True
        output.gs = raw_gs

        if "extrinsics"  in output and "intrinsics" in output:
            gs_render = self.rasterize_gs(output, raw_gs, **self.rasterize_cfg)
            output.gs_render = gs_render
            if render_layers:
                output.gs_render_layers = []
                for gs in raw_gs_layers:
                    gs_render = self.rasterize_gs(output, gs, **self.rasterize_cfg)
                    output.gs_render_layers.append(gs_render)
            
        return output

    def rasterize_gs(self, output, raw_gs, chunk_size:int=1, 
                     area_target:float=1.0, area_tau:float=0.5, spread_target:float=0.03,
                     contrib_min:float=1e-4, contrib_max:float=1e-2):
        ext = output.extrinsics
        ixt = output.intrinsics
        colors, depths, meta = run_renderer_in_chunk_w_trj_mode(
            gaussians=raw_gs,
            extrinsics=ext,
            intrinsics=ixt,
            image_shape=(504,504),
            chunk_size=chunk_size,
            trj_mode='original',
            use_sh=False,
            return_meta=True
        )
        
        # The contribution is calculated in log space for better numerical stability, 
        # and is a combination of visibility, opacity, and projected area (conics) of the Gaussian splats.
        radii = meta["radii"].float()           # [b,s,n,2]
        r = radii.amax(dim=-1)                  # [b,s,n]
        valid = (r > 0)
        
        conics = meta["conics"].float()         # [b,s,n,3]
        a, b, c = conics[..., 0], conics[..., 1], conics[..., 2]
        det = (a * c - b * b).clamp_min(1e-6)
        log_area = torch.log(det) * -0.5
        log_area_clip = torch.log(torch.tensor(area_target))
        area_gate = ((log_area - log_area_clip) / area_tau).sigmoid()
        log_area_gate = torch.log(area_gate.clamp_min(1e-6))

        opac = meta["opacities"].float()
        log_alpha = torch.log(opac.clamp_min(1e-6))

        log_contrib = log_alpha + log_area_gate

        # Contribution loss encourages the model to produce Gaussian splats that 
        # have a meaningful contribution to the final rendered image
        log_tau_min = torch.log(log_contrib.new_tensor(contrib_min))
        log_tau_max = torch.log(log_contrib.new_tensor(contrib_max))
        contrib_loss = F.relu(log_tau_min - log_contrib) + F.relu(log_contrib - log_tau_max)
        contrib_loss = (contrib_loss * valid.float()).sum() / (valid.sum() + 1e-6)

        # Prepare a log_contrib just for monitoring, detached from the computational graph
        contrib_vis = log_contrib.detach()
        contrib_vis = torch.where(valid, log_contrib, torch.full_like(log_contrib, float('-inf')))
        in_band = ((contrib_vis > log_tau_min) & (contrib_vis < log_tau_max)).float().mean(-1)
        contrib_vis = torch.exp(contrib_vis)  # Convert back to linear space for visualization

        # Spread of the 2D means can indicate how well the Gaussians are distributed across the image plane.
        means2d = meta["means2d"].float()           # [b,s,n,2]
        means2d_normed = (means2d - means2d.mean(dim=-2, keepdim=True)) / 504  # Centered and normalized

        # spread_loss v1: Encourage the standard deviation of the 2D means to be above a certain threshold, 
        # promoting a wider spread of Gaussians across the image plane.
        std = means2d_normed.std(dim=-2)
        std_loss = F.relu(0.2 - std).mean()
        spread_loss = std_loss

        # spread_loss v2: logdet will explode on start, commented out for later inspection
        # cov2d = means2d_normed.transpose(-1,-2) @ means2d_normed / (means2d_normed.shape[-2] - 1)
        # identity_eps = torch.eye(cov2d.shape[-1], device=cov2d.device) * 1e-6
        # _, logabsdet = torch.linalg.slogdet(cov2d + identity_eps)
        # logdet_target = torch.log(torch.tensor(spread_target**2))
        # spread_loss = F.relu(logdet_target - logabsdet).mean()  # Encourage spread-out means by maximizing the determinant of the covariance

        # spread_loss v3: Encourage the covariance matrix of the 2D means to be close to a diagonal matrix 
        # with a certain variance, promoting both spread and decorrelation of the Gaussians.
        # cov_xy = cov2d[..., 0, 1]
        # decorrelation_loss = cov_xy.abs().mean()
        # spread_loss = std_loss + decorrelation_loss

        return {
            "colors": colors,
            "depths": depths,
            "contrib_loss": contrib_loss,
            "spread_loss": spread_loss,
            "contrib_vis": contrib_vis,
            "in_band": in_band
        }

    def _process_raft_head(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process feature maps through custom raft variant"""

        if "fmap" in output and "cnet" in output:
            fmap = output.pop("fmap").detach()
            cnet   = output.pop("cnet").detach()

            # feat0 = torch.cat(
            #     [depth_fmap[:, :-1], ray_fmap[:, :-1]], dim=2
            # ).flatten(0,1).detach()

            # feat1 = torch.cat(
            #     [depth_fmap[:, 1:], ray_fmap[:, 1:]], dim=2
            # ).flatten(0,1).detach()

            # del depth_fmap, ray_fmap
            # torch.cuda.empty_cache()
            """
            SEA-RAFT Implementation (Copy from Official Repo)
            """
            flow_out = self.raft_head(fmap, cnet)

            output.flow = flow_out["preds"]
            output.flow_info = flow_out["infos"]

        return output

    def _process_mono_sky_estimation(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process mono sky estimation."""
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output
        
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        # Set sky regions to maximum depth and high confidence
        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _process_ray_pose_estimation(
        self, output: Dict[str, torch.Tensor], height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        if "ray" in output and "ray_conf" in output:
            pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
                output.ray,
                output.ray_conf,
                output.ray.shape[-3],
                output.ray.shape[-2],
            )
            pred_extrinsic = affine_inverse(pred_extrinsic) # w2c -> c2w
            pred_extrinsic = pred_extrinsic[:, :, :3, :]
            pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
            pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
            pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
            pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
            pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
            del output.ray
            del output.ray_conf
            output.extrinsics = pred_extrinsic
            output.intrinsics = pred_intrinsic
        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head(feats, H, W, patch_start_idx=0)
    
    # def _process_flow_head(
    #         self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Process features through the flow prediction head.
    #     feats: Layers of (feats, cam_token), where feats is (B,S,N,C) and cam_token is (B,S,C)
    #     """
    #     paired_feats = []
    #     for feat in feats:
    #         feat1 = feat[0][:, :-1]  # (B, S-1, N, C)
    #         feat2 = feat[0][:, 1:]   # (B, S-1, N, C)
    #         feat_pair = torch.cat([feat1, feat2], dim=-1)  # (B, S-1, N, 2C)
    #         # Camera tokens must match the paired temporal dimension (S-1).
    #         # Use camera token for the 'first' frame in each pair to align with feat1.
    #         cam_token_paired = feat[1][:, :-1]  # (B, S-1, C)
    #         paired_feats.append((feat_pair.detach(), cam_token_paired.detach()))

    #     flow_init = self.flow_head(paired_feats, H, W, patch_start_idx=0)

    #     #TODO add RAFT
        
    #     output.flow = flow_init

    #     return output


    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            # Remove ray information as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # The depth is defined in the DA3 model's camera space,
        # so even with provided GT camera poses,
        # we instead use the predicted camera poses for better alignment.
        ctx_extr = output.get("extrinsics", None)
        ctx_intr = output.get("intrinsics", None)
        assert (
            ctx_extr is not None and ctx_intr is not None
        ), "must process camera info first if GT is not available"

        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
        )
        output.gaussians = gs_world

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth Anything 3 network with metric scaling capabilities.

    This network combines two DepthAnything3Net branches:
    - Main branch: Standard depth estimation
    - Metric branch: Metric depth estimation for scaling alignment

    The network performs depth alignment using least squares scaling
    and handles sky region masking for improved depth estimation.

    Args:
        preset: Configuration for the main depth estimation branch
        second_preset: Configuration for the metric depth branch
    """

    def __init__(self, anyview: DictConfig, metric: DictConfig):
        """
        Initialize NestedDepthAnything3Net with two branches.

        Args:
            preset: Configuration for main depth estimation branch
            second_preset: Configuration for metric depth branch
        """
        super().__init__()
        self.da3:DepthAnything3Net = create_object(anyview)
        self.da3_metric:DepthAnything3Net = create_object(metric)

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both branches with metric scaling alignment.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from
            infer_gs: Enable Gaussian Splatting branch
            use_ray_pose: Use ray-based pose estimation
            ref_view_strategy: Strategy for selecting reference view

        Returns:
            Dictionary containing aligned depth predictions and camera parameters
        """
        # Get predictions from both branches
        output = self.da3(
            x, extrinsics, intrinsics, export_feat_layers=export_feat_layers, infer_gs=infer_gs, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy
        )
        metric_output = self.da3_metric(x)

        # Apply metric scaling and alignment
        output = self._apply_metric_scaling(output, metric_output)
        output = self._apply_depth_alignment(output, metric_output)
        output = self._handle_sky_regions(output, metric_output)

        return output

    def _apply_metric_scaling(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply metric scaling to the metric depth output."""
        # Scale metric depth based on camera intrinsics
        metric_output.depth = apply_metric_scaling(
            metric_output.depth,
            output.intrinsics,
        )
        return output

    def _apply_depth_alignment(
        self, output: Dict[str, torch.Tensor], metric_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply depth alignment using least squares scaling."""
        # Compute non-sky mask
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Ensure we have enough non-sky pixels
        assert non_sky_mask.sum() > 10, "Insufficient non-sky pixels for alignment"

        # Sample depth confidence for quantile computation
        depth_conf_ns = output.depth_conf[non_sky_mask]
        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        # Compute alignment mask
        align_mask = compute_alignment_mask(
            output.depth_conf, non_sky_mask, output.depth, metric_output.depth, median_conf
        )

        # Compute scale factor using least squares
        valid_depth = output.depth[align_mask]
        valid_metric_depth = metric_output.depth[align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)

        # Apply scaling to depth and extrinsics
        output.depth *= scale_factor
        output.extrinsics[:, :, :3, 3] *= scale_factor
        output.is_metric = 1
        output.scale_factor = scale_factor.item()

        return output

    def _handle_sky_regions(
        self,
        output: Dict[str, torch.Tensor],
        metric_output: Dict[str, torch.Tensor],
        sky_depth_def: float = 200.0,
    ) -> Dict[str, torch.Tensor]:
        """Handle sky regions by setting them to maximum depth."""
        non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)

        # Compute maximum depth for non-sky regions
        # Use sampling to safely compute quantile on large tensors
        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = min(torch.quantile(sampled_depth, 0.99), sky_depth_def)

        # Set sky regions to maximum depth and high confidence
        output.depth, output.depth_conf = set_sky_regions_to_max_depth(
            output.depth, output.depth_conf, non_sky_mask, max_depth=non_sky_max
        )

        return output
