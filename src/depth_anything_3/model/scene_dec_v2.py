import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from depth_anything_3.specs import Gaussians
from depth_anything_3.model.attn import SlotAttentionLayer
from depth_anything_3.model.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

class SceneAutoEncoder(nn.Module):
    def __init__(
            self,
            dim_in:int=2048,
            token_dim:int=512,
            feat_res:list[int]=[1,2,4,8],   # Coarse to fine feature resolutions
            num_latents:int=8,    # Number of latents for the scene
            start_tokens:int=50,   # Number of tokens sampled for the coarsest level (multiplied by num_latents)
            densify_tokens:int=[100, 200, 400, 800],   # Number of tokens sampled for next level (multiplied by num_latents)
            gs_params:list[str]=['pos','scale','rot','opac','col'],
        ):
        super().__init__()
        assert len(feat_res) == len(densify_tokens), "feat_res and num_tokens must have the same length"
        self.num_latents = num_latents
        self.start_tokens = start_tokens
        self.densify_tokens = densify_tokens
        self.token_dim = token_dim
    
        self.canonical_query = nn.Parameter(torch.empty(1, num_latents, dim_in))
        nn.init.normal_(self.canonical_query, mean=0.0, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim_in,
            num_heads=8,
            batch_first=True,
        )
        self.delta_mu_head = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        nn.init.zeros_(self.delta_mu_head[-1].weight)
        nn.init.zeros_(self.delta_mu_head[-1].bias)

        self.delta_log_sigma_head = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        nn.init.zeros_(self.delta_log_sigma_head[-1].weight)
        nn.init.zeros_(self.delta_log_sigma_head[-1].bias)

        self.token_init = SceneTokenInitializer(num_head=num_latents, token_dim=token_dim)
        self.decoders = nn.ModuleList()
        for i in range(len(feat_res)):
            dec = SceneDecoder(
                patch_dim=dim_in, patch_resize=feat_res[i], token_dim=token_dim,
                gs_params=gs_params
            )
            self.decoders.append(dec)
        
    def forward(
            self, 
            cam_layers:torch.Tensor, 
            patch_layers:torch.Tensor, 
            H:int, 
            W:int,
        ):
        """
        Args:
        cam_layers: List of camera feature tensors at different levels, each of shape (B, views, C)
        patch_layers: List of patch feature tensors at different levels, each of shape (B, t, N, C)
        H, W: Original image height and width for positional embedding

        Returns:
        A dictionary containing:
        - "gaussians": List of Gaussian outputs at each level
        - "scene_tokens": List of scene tokens at each level
        - "token_updates": List of token initialization updates (delta_mu and delta_log_sigma)
        """
        # print(f"Start AutoEncoder: {vram()}")
        assert cam_layers.size(0) == patch_layers.size(0) == len(self.decoders),\
            f"Number of layers in cam and patch must match number of decoders"\
            f"Got {cam_layers.size(0)} cam layers, {patch_layers.size(0)} patch layers, "\
            f"but {len(self.decoders)} decoders."
        
        # Reverser cam and patch layers to decode from coarse to fine
        cam_layers = cam_layers.flip(dims=[0]).detach()
        patch_layers = patch_layers.flip(dims=[0]).detach()
        # print(f"Load inputs: {vram()}")

        # Initialize scene tokens from the cam_layers of the coarsest level
        B, V, _ = cam_layers[0].shape
        cam_tokens = cam_layers[0]  # (B, views, C)
        query = self.canonical_query.expand(B, -1, -1)  # (B, num_latents, C)

        # Cross-attention to get a single canonical token
        canonical_token, _ = self.cross_attn(
            query=query, key=cam_tokens, value=cam_tokens
        )  # (B, num_latents, C)
        delta_mu = self.delta_mu_head(canonical_token)  # (B, num_latents, C)
        delta_log_sigma = self.delta_log_sigma_head(canonical_token)  # (B, num_latents, C)
        latent_updates = [torch.cat([delta_mu, delta_log_sigma], dim=-2)] # (B, 2*num_latents, C)
        # print(f"Initialize canonical token: {vram()}")
            
        scene_tokens = self.token_init(delta_mu, delta_log_sigma, self.start_tokens)  # (B, N*mode, C)
        # print(f"Initialize scene token: {vram()}")

        gs_layers = []
        token_layers = []
        for i in range(len(self.decoders)):
            prev_tokens = scene_tokens
            out = self.decoders[i](patch_layers[i], scene_tokens, H, W)
            out_gs = out["gaussians"]
            out_token = out["scene_token"]
            gs_layers.append(out_gs)
            token_layers.append(out_token)
            # print(f"Layer {i} Decoded: {vram()}")

            # Reshape tokens to separate latents and tokens per latent for easier processing
            out_token = out_token.reshape(B, self.num_latents, -1, self.token_dim)  # (B, num_latents, tokens_per_latent, C)
            prev_tokens = prev_tokens.reshape(B, self.num_latents, -1, self.token_dim)

            # Compute latent updates for next layer based on token differences
            mu = (out_token - prev_tokens).mean(-2)  # Use token difference as delta_mu [B, num_latents, C]
            std_prev = prev_tokens.std(dim=-2).clamp_min(1e-6)  # [B, num_latents, C]
            std_next = out_token.std(dim=-2).clamp_min(1e-6)  # [B, num_latents, C]
            log_sigma = torch.log(std_next / std_prev)  # Use log of std ratio as delta_log_sigma
            latent_updates.append(torch.cat([mu, log_sigma], dim=-2))
            delta_mu += mu
            delta_log_sigma += log_sigma
            
            # Initialize next layer tokens with updated parameters
            scene_tokens = self.token_init(delta_mu, delta_log_sigma, self.densify_tokens[i])  # (B, N'*mode, C)
            # print(f"Layer {i} Initialized: {vram()}")

        return {
            "gaussians": gs_layers,   # List of Gaussian outputs at each level
            "scene_tokens": token_layers,   # List of scene tokens at each level
            "token_updates": latent_updates, # List of token initialization updates at each level
        }

class SceneDecoder(nn.Module):
    def __init__(
        self,
        patch_dim:int=2048,
        patch_resize:int=1,
        token_dim:int=512,
        gs_params:list[str]=['pos','scale','rot','opac','col']
    ):
        super().__init__()
        self.token_dim = token_dim
        self.proj_patch = nn.Linear(patch_dim, token_dim)
        self.resize_patch = nn.Identity()
        if patch_resize > 1:
            scale = patch_resize
            self.resize_patch = nn.ConvTranspose2d(
                token_dim, token_dim, kernel_size=scale, 
                stride=scale, padding=0
            )
        elif patch_resize < 0:
            scale = -patch_resize
            self.resize_patch = nn.Conv2d(
                token_dim, token_dim, kernel_size=scale*2+1,
                stride=scale, padding=scale
            )
        self.encoder = SlotAttentionLayer(
            num_iterations=2,
            slot_dim=token_dim,
            mlp_hidden_dim=token_dim * 2
        )
        self.decoder = GaussianDecoder(d_model=token_dim, gs_params=gs_params)

    def forward(self, patch:Tensor, scene_token:Tensor, H:int, W:int):
        """
        patch:  (B, t, N, D)
        scene_token: (B, S, C)
        """
        B, t, N, _ = patch.shape

        # Project patch tokens to correct dimension
        ph = int(round(math.sqrt(N * H / W)))
        pw = int(round(math.sqrt(N * W / H)))
        patch:torch.Tensor = self.proj_patch(patch).flatten(0,1)    # (B*t, N, hid)
        patch = patch.permute(0,2,1).reshape(-1, self.token_dim, ph, pw)   # (B*t, hid, ph, pw)
        patch = self._add_pos_embed(patch, H, W)
        patch = self.resize_patch(patch)
        patch = patch.permute(0,2,3,1).reshape(B, t, -1, self.token_dim)    # (B, t, N', hid)
        patch_token = patch.permute(1,0,2,3)    # (t, B, N', hid)

        for patch in patch_token:
            scene_token = self.encoder(patch, scene_token) # (B, S, C)
        out_gs = self.decoder(scene_token)

        return {
            "gaussians": out_gs,
            "scene_token": scene_token
        }
    
    def _add_pos_embed(self, x: torch.Tensor, H: int, W: int, ratio: float = 0.1) -> torch.Tensor:
        """Simple UV positional embedding added to feature maps."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe
    
class GaussianDecoder(nn.Module):
    def __init__(self, d_model=512, gs_params=['pos','scale','rot','opac','col']):
        super().__init__()
        GS_DIMS = {"pos": 3,"scale": 3,"rot": 4,"opac": 1,"col": 3,}
        self.gs_params = gs_params
        self.num_gs_params = sum(GS_DIMS[name] for name in gs_params)
        
        # A strong 3-layer MLP is usually sufficient here
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_gs_params)
        )
        if self.num_gs_params != 14 :
            self.pad = nn.ConstantPad1d((0,14-self.num_gs_params),1.0)
        else:
            self.pad = nn.Identity()

    def check_nan(self, tensor:torch.Tensor, name:str):
        if torch.is_floating_point(tensor):
            if torch.isnan(tensor).any():
                raise ValueError(f"!!! NAN DETECTED in {name} !!!")
            if torch.isinf(tensor).any():
                raise ValueError(f"!!! INF DETECTED in {name} !!!")

    def forward(self, scene_tokens):
        """
        scene_tokens: [B, Q, 512]
        """
        
        raw_gaussians = self.mlp(scene_tokens)  # Output: [B, Q, N]
        
        # Reshape to distinct Gaussians and pad to correct dim: [B, Q, 14]
        gaussians = self.pad(raw_gaussians)
        self.check_nan(gaussians, "Gaussians")
        
        # Slice parameters and apply necessary activations
        (
            means,
            scales,
            rotations,
            opacities,
            colors
        ) = torch.split(gaussians, (3,3,4,1,3), dim=-1)
        opacities = opacities.squeeze(-1)   # Squeeze last dim for opac.
        colors = colors.unsqueeze(-1)       # Unsqueeze last dim for SH
        
        # Scales must be strictly positive
        if 'scale' in self.gs_params:
            scales = torch.exp(scales) 
        
        # Quaternions must be normalized
        if 'rot' in self.gs_params:
            rotations = F.normalize(rotations)
        
        # Opacity must be between 0 and 1
        if 'opac' in self.gs_params:
            opacities = torch.sigmoid(opacities)
        
        # Color (Spherical Harmonics DC band) can be sigmoid for standard RGB
        if 'col' in self.gs_params:
            colors = torch.sigmoid(colors)
        
        return Gaussians(
            means=means,
            scales=scales,
            rotations=rotations,
            harmonics=colors,
            opacities=opacities
        )

class SceneTokenInitializer(nn.Module):
    def __init__(
        self,
        num_head:int=8,
        token_dim:int=512,
    ):
        super().__init__()
        self.token_dim = token_dim

        # Parameters for Gaussian init, shared by all tokens.
        self.base_mu = nn.Parameter(torch.empty(1, num_head, token_dim))
        self.base_log_sigma = nn.Parameter(torch.empty(1, num_head, token_dim))

        # Initialize parameters to reasonable values, check whether it is effective in practice
        nn.init.normal_(self.base_mu, mean=0.0, std=0.02)
        nn.init.constant_(self.base_log_sigma, -1.0)

    def forward(self, delta_mu:Tensor, delta_log_sigma:Tensor, samples:int) -> Tensor:
        """
        delta_mu: (B, 1, D) - delta feature for token initialization
        delta_log_sigma: (B, 1, D) - delta log sigma for token initialization
        samples: int - number of samples to generate
        """
        # Combine base parameters with deltas to get final mu and log_sigma
        mu = self.base_mu + delta_mu
        log_sigma = self.base_log_sigma + delta_log_sigma
        
        # Expand to desired number of samples
        mu = mu.repeat(1, samples, 1)  # (B, heads * samples, D)
        log_sigma = log_sigma.repeat(1, samples, 1)    # (B, heads * samples, D)
        
        # Sample tokens from the Gaussian distribution
        sigma = torch.exp(log_sigma)
        tokens = mu + sigma * torch.randn_like(mu)
        
        return tokens
        