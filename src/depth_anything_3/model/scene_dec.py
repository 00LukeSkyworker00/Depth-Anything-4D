import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from depth_anything_3.specs import Gaussians
from depth_anything_3.model.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)
from depth_anything_3.model.attn import CrossAttnLayer, SlotAttentionLayer

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

class SceneDecoder(nn.Module):
    def __init__(
            self, 
            dim_in:int=2048, 
            hid_dim:int=512, 
            token_resize:int=8,
            base_tokens:int=200, 
            iters_per_frame:int=1,
            out_layers:list[int]=[],
            gs_params:list[str]=['pos','scale','rot','opac','col'],
            out_growth:int=4
        ):
        super().__init__()
        self.hid_dim = hid_dim
        self.iters_per_frame = iters_per_frame
        self.out_layers = out_layers
        self.init_mu = nn.Sequential(
            nn.Linear(dim_in, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.init_logvar = nn.Sequential(
            nn.Linear(dim_in, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.proj_patch = nn.Linear(dim_in, hid_dim)
        self.resize_patch = nn.Identity()
        if token_resize > 1:
            scale = token_resize
            self.resize_patch = nn.ConvTranspose2d(
                hid_dim, hid_dim, kernel_size=scale, 
                stride=scale, padding=0
            )
        elif token_resize < 0:
            scale = -token_resize
            self.resize_patch = nn.Conv2d(
                hid_dim, hid_dim, kernel_size=scale*2+1,
                stride=scale, padding=scale
            )
        self.base_tokens = base_tokens

        self.token_decoder = CrossAttnLayer(
            d_model=hid_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        # self.token_decoder = nn.TransformerDecoderLayer(
        #     d_model=hid_dim,
        #     nhead=8,
        #     dim_feedforward=2048,
        #     dropout=0.0,
        #     activation='gelu',
        #     batch_first=True,
        #     norm_first=True
        # )

        self.decoder = GaussianDecoder(
            d_model=hid_dim,
            gs_per_token=out_growth,
            gs_params=gs_params
        )

    def forward(self, readout:Tensor, patch:Tensor, H:int, W:int):
        """
        readout: (B, S, C)
        patch:  (B, S, N, C)
        """
        # print("SceneDec Start:", vram())
        B, S, N, C = patch.shape
        out_token: list[torch.Tensor] = []
        out_gs: list[Gaussians] = []

        # Initialize scene tokens from readout
        mu:torch.Tensor = self.init_mu(readout).repeat(1,self.base_tokens,1)   # (B, S*P, hid)
        sigma:torch.Tensor = self.init_logvar(readout).exp().repeat(1,self.base_tokens,1)    # (B, S*P, hid)
        eps = torch.randn(mu.shape, device=mu.device, dtype = mu.dtype)
        scene_token = mu + sigma * eps  #(B, S*P, hid)

        # Project patch tokens to correct dimension
        ph = int(round(math.sqrt(N * H / W)))
        pw = int(round(math.sqrt(N * W / H)))
        patch:torch.Tensor = self.proj_patch(patch).flatten(0,1)  # (B*S, N, hid)
        patch = patch.permute(0,2,1).contiguous().reshape(-1, self.hid_dim, ph, pw)     # (B*S, hid, ph, pw)
        patch = self._add_pos_embed(patch, H, W)
        patch_token = self.resize_patch(patch).reshape(B, S, self.hid_dim, -1).permute(1,0,3,2).contiguous()     # (S, B, N', hid)
        # print("SceneDec Init:", vram())

        for i in range(patch_token.shape[0]):
            for _ in range(self.iters_per_frame):
                scene_token = self.token_decoder(
                    tgt=scene_token,
                    memory=patch_token[i]
                )   #(B, S*P, hid)
            if i in self.out_layers or i+1 == patch_token.shape[0]:
                out_token.append(scene_token)
                out_gs.append(self.decoder(scene_token))
            # print("SceneDec Transformer Layer:", vram())
        
        return out_gs, out_token

    def _add_pos_embed(self, x: torch.Tensor, H: int, W: int, ratio: float = 0.1) -> torch.Tensor:
        """Simple UV positional embedding added to feature maps."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

class GaussianDecoder(nn.Module):
    def __init__(self, d_model=512, gs_per_token=4, gs_params=['pos','scale','rot','opac','col']):
        super().__init__()
        self.K = gs_per_token
        GS_DIMS = {"pos": 3,"scale": 3,"rot": 4,"opacity": 1,"col": 3,}
        self.gs_params = gs_params
        self.num_gs_params = sum(GS_DIMS[name] for name in gs_params)
        out_dim = self.K * self.num_gs_params
        
        # A strong 3-layer MLP is usually sufficient here
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim)
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
        # scene_tokens: [B, Q, 512]
        B, Q, _ = scene_tokens.shape
        
        # Output: [B, Q, K * N]
        raw_gaussians = self.mlp(scene_tokens)
        
        # Reshape to distinct Gaussians and pad to correct dim: [B, Q * K, 14]
        gaussians = raw_gaussians.view(B, Q * self.K, self.num_gs_params)
        gaussians = self.pad(gaussians)
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
