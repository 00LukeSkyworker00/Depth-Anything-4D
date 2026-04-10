import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, Optional, Union

from depth_anything_3.specs import Gaussians
from depth_anything_3.model.utils.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)

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
            gs_dim:int=7,
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
        self.resize_patch =  nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=token_resize, stride=token_resize, padding=0)
        self.base_tokens = base_tokens

        self.cross_attn = CrossAttnLayer(
            d_model=hid_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.decoder = GaussianDecoder(
            d_model=hid_dim,
            gs_per_token=out_growth,
            gs_dim=gs_dim
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
                scene_token = self.cross_attn(
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
    def __init__(self, d_model=512, gs_per_token=4, gs_dim=7):
        super().__init__()
        self.K = gs_per_token
        self.num_gs_params = gs_dim
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

    def check_nan(self, tensor:torch.Tensor, name:str):
        if torch.is_floating_point(tensor):
            if torch.isnan(tensor).any():
                raise ValueError(f"!!! NAN DETECTED in {name} !!!")
            if torch.isinf(tensor).any():
                raise ValueError(f"!!! INF DETECTED in {name} !!!")

    def forward(self, scene_tokens):
        # scene_tokens: [B, Q, 512]
        B, Q, C = scene_tokens.shape
        
        # Output: [B, Q, K * 14]
        raw_gaussians = self.mlp(scene_tokens)
        
        # Reshape to distinct Gaussians: [B, Q * K, 14]
        gaussians = raw_gaussians.view(B, Q * self.K, self.num_gs_params)
        
        # Slice parameters and apply necessary activations
        # Positions: Add to token's base 3D coordinate (if applicable) or use directly
        means = gaussians[..., 0:3] 
        self.check_nan(means, "Means")
        
        # Scales must be strictly positive
        scales = torch.exp(gaussians[..., 3:6]) 
        self.check_nan(scales, "Scales")
        
        # Quaternions must be normalized
        rotations = torch.nn.functional.normalize(gaussians[..., 6:10], dim=-1)
        self.check_nan(rotations, "Rotations")
        
        # # Opacity must be between 0 and 1
        # opacities = torch.sigmoid(gaussians[..., 10:11]).squeeze(-1)
        opacities = torch.ones_like(gaussians[..., 0].detach(), requires_grad=False)
        
        # # Color (Spherical Harmonics DC band) can be sigmoid for standard RGB
        # colors = torch.sigmoid(gaussians[..., 11:14]).unsqueeze(-1)
        colors = torch.ones_like(gaussians[..., :3].detach(), requires_grad=False).unsqueeze(-1)
        
        return Gaussians(
            means=means,
            scales=scales,
            rotations=rotations,
            harmonics=colors,
            opacities=opacities
        )


class CrossAttnLayer(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the cross-attn layer.

        Args:
            tgt: the sequence to the cross-attn layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def _get_activation_fn(self, activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
