from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from depth_anything_3.specs import Gaussians
from depth_anything_3.model.attn import CrossAttnLayer

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

class SceneAutoEncoder(nn.Module):
    def __init__(
            self,
            dim_in:int=2048,
            gs_params:list[str]=['pos','scale','rot','opac','col'],
        ):
        super().__init__()
    
        # self.canonical_query = nn.Parameter(torch.empty(1, num_latents, dim_in))
        # nn.init.normal_(self.canonical_query, mean=0.0, std=0.02)

        self.scene_decoder = SceneDecoder(feat_dim=dim_in, gs_params=gs_params)

        
    def forward(
            self, 
            cam_layers:torch.Tensor, 
            patch_layers:torch.Tensor, 
            H:int, 
            W:int,
        ):
        out_dict = self.scene_decoder(patch_layers)

        return out_dict
    
class SceneDecoder(nn.Module):
    def __init__(self, feat_dim=2048, gs_params=['pos','scale','rot','opac','col']):
        super().__init__()

        tsfm_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, 
            nhead=8, 
            dim_feedforward=feat_dim * 2, 
            activation='gelu',
            batch_first=True
        )
        self.latent_encoder = TransformerFPN(feat_dim=feat_dim, channel_compression=True)
        latent_dim = feat_dim // 8 if self.latent_encoder.is_compress else feat_dim
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.log_var_head = nn.Linear(latent_dim, latent_dim)
        self.latent_decoder = GaussianDecoder(
            d_model=latent_dim, 
            gs_params=gs_params
        )

    def forward(
            self, 
            patch_layers:torch.Tensor, 
        ):
        # encode to latent space and sample scene tokens
        latent = self.latent_encoder(patch_layers)  # (B, t, N, D//8)
        mu = self.mu_head(latent)  # (B, t, N, D//8)
        log_var = self.log_var_head(latent)  # (B, t, N, D//8)
        scene_tokens = self.reparameterize(mu=mu, log_var=log_var)  # (B, t, N, D//8)

        # decode scene tokens to Gaussian parameters
        gs_per_view = self.latent_decoder(scene_tokens)

        return {
            "gaussians": gs_per_view,   # List of Gaussian outputs per camera in sequence
            "kl_div": self.kl_divergence(mu, log_var)  # KL divergence loss for regularization
        }

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)  # (B, t, N, D)
        eps = torch.randn_like(std)  # (B, t, N, D)
        return mu + eps * std  # (B, t, N, D)

    def kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # KL divergence between the learned distribution and a standard normal distribution
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)  # (B, t, N)
        return kl.mean()  # Average over batch, time, and tokens

class TransformerFPN(nn.Module):
    def __init__(self, feat_dim=2048, channel_compression=False):
        super().__init__()
        self.is_compress = channel_compression
        if channel_compression:
            out_channels = [feat_dim, feat_dim, feat_dim//2, feat_dim//4]
            self.compress = nn.ModuleList()
        else:
            out_channels = [feat_dim, feat_dim, feat_dim, feat_dim]
        self.l4_project = nn.Linear(feat_dim, out_channels[0])
        self.l3_project = nn.Linear(feat_dim, out_channels[1])
        self.l2_project = nn.Linear(feat_dim, out_channels[2])
        self.l1_project = nn.Linear(feat_dim, out_channels[3])

        self.fuse = nn.ModuleList()
        for i in range(3):
            d_model = out_channels[i+1]
            fuse = CrossAttnLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 2,
                activation='gelu',
                batch_first=True
            )
            self.fuse.append(fuse)
            if channel_compression:
                compress_layer = nn.Linear(d_model, d_model//2)
                self.compress.append(compress_layer)

    def forward(self, patch_layers):
        l1 = self.l1_project(patch_layers[0])  # finest level
        l2 = self.l2_project(patch_layers[1])
        l3 = self.l3_project(patch_layers[2])
        l4 = self.l4_project(patch_layers[3])  # coarsest level

        B, T, N, _ = l4.shape
        feat = l4.flatten(0,1)  # (B*t, N, D) Start from the coarsest level
        for i, memory in enumerate([l3, l2, l1]):
            memory = memory.flatten(0,1)  # (B*t, N, D)
            feat = self.fuse[i](tgt=feat, memory=memory)  # Fuse with the next finer level
            if self.is_compress:
                feat = self.compress[i](feat)  # Compress the feature dimension

        return feat.reshape(B, T, N, -1)  # (B, t, N, D//8) or (B, t, N, D) if no compression

class GaussianDecoder(nn.Module):
    def __init__(self, d_model=512, gs_params=['pos','scale','rot','opac','col'], gs_per_token=14):
        super().__init__()
        GS_DIMS = {"pos": 3,"scale": 3,"rot": 4,"opac": 1,"col": 3,}
        self.gs_params = gs_params
        self.num_gs_params = sum(GS_DIMS[name] for name in gs_params)
        self.gs_per_token = gs_per_token
        
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_gs_params * gs_per_token)
        )

    def check_nan(self, tensor:torch.Tensor, name:str):
        if torch.is_floating_point(tensor):
            if torch.isnan(tensor).any():
                raise ValueError(f"!!! NAN DETECTED in {name} !!!")
            if torch.isinf(tensor).any():
                raise ValueError(f"!!! INF DETECTED in {name} !!!")

    def forward(self, latent_sample:torch.Tensor) -> List[Gaussians]:
        """
        latent_sample: [B, T, N, d_model] - latent representation for each token
        """
        B, T, _, _ = latent_sample.shape
        gs_logits = self.decoder(latent_sample)
        self.check_nan(gs_logits, "Gaussians")

        gs_logits = gs_logits.permute(1,0,2,3)  # (T, B, N, gs_per_token*num_gs_params)
        gs_logits = gs_logits.reshape(T, B, -1, self.num_gs_params)  # (T, B, N', num_gs_params)
        N = gs_logits.shape[-2]
        
        # Initialize output tensors for each Gaussian parameter with default values
        means = torch.zeros(T, B, N, 3, device=gs_logits.device)
        rotations = torch.zeros(T, B, N, 4, device=gs_logits.device)
        scales = torch.ones(T, B, N, 3, device=gs_logits.device)
        opacities = torch.ones(T, B, N, device=gs_logits.device)
        colors = torch.ones(T, B, N, 3, 1, device=gs_logits.device)

        # Slice parameters and apply necessary activations
        for param in self.gs_params:
            if param == 'pos':
                means = gs_logits[..., :3]  # (T, B, N, 3)
                gs_logits = gs_logits[..., 3:]  # Remove used params
            elif param == 'rot':
                rotations = gs_logits[..., :4]  # (T, B, N, 4)
                gs_logits = gs_logits[..., 4:]
                # Quaternions must be normalized
                rotations = F.normalize(rotations)
            elif param == 'scale':
                scales = gs_logits[..., :3]  # (T, B, N, 3)
                gs_logits = gs_logits[..., 3:]
                # Scales must be strictly positive
                scales = torch.exp(scales) 
            elif param == 'opac':
                opacities = gs_logits[..., 0]  # (T, B, N)
                gs_logits = gs_logits[..., 1:]
                # Opacity must be between 0 and 1
                opacities = torch.sigmoid(opacities)
            elif param == 'col':
                colors = gs_logits[..., :3]  # (T, B, N, 3)
                colors = colors.unsqueeze(-1)   # (T, B, N, 3, 1) Unsqueeze last dim for SH
                gs_logits = gs_logits[..., 3:]
                # Color (Spherical Harmonics DC band) can be sigmoid for standard RGB
                colors = torch.sigmoid(colors)
        gs_per_view = [
            Gaussians(
                means=m,
                scales=s,
                rotations=r,
                harmonics=c,
                opacities=o
            ) for (m,s,r,c,o) in zip(
                means,
                scales,
                rotations,
                colors,
                opacities
            )
        ]
        return gs_per_view

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
        