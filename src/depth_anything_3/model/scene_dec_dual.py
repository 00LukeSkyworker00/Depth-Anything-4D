import torch
import torch.nn as nn

from .scene_dec import SceneDecoder
from depth_anything_3.specs import Gaussians

class DualSceneDecoder(nn.Module):
    def __init__(
            self,
            dim_in:int=2048,
            hid_dim:list[int]=[256, 512, 1024, 1024],
            token_resize:list[int]=[8,4,2,1],
            base_tokens:int=200,
            iters_per_frame:int=1,
            out_layers:list[int]=[0,1,2,3],
            gs_dim:int=7,
            gs_per_token:int=32
        ):
        super().__init__()
        assert len(out_layers) == len(hid_dim) and len(out_layers) == len(token_resize), "Layers config not match"

        self.decoders = nn.ModuleList()

        for i in range(len(hid_dim)):
            dec = SceneDecoder(
                dim_in=dim_in, hid_dim=hid_dim[i], token_resize=token_resize[i], 
                base_tokens=base_tokens, iters_per_frame=iters_per_frame, 
                out_layers=out_layers, gs_dim=gs_dim, out_growth=gs_per_token//token_resize[i]
            )
            self.decoders.append(dec)
        
    def forward(
            self, 
            cam_layers:torch.Tensor, 
            patch_layers:torch.Tensor, 
            H:int, 
            W:int,
            concat_gs:bool=True
        ):
        out_gs_layers:list[Gaussians] = []
        out_token_layers:list[torch.Tensor] = []
        for i in range(len(self.decoders)):
            out = self.decoders[i](cam_layers[i], patch_layers[i], H, W)
            out_gs_layers.append(out[0][-1])
            out_token_layers.append(out[1][-1])
        if concat_gs:
            out_gs = sum(out_gs_layers[1:], out_gs_layers[0])
        else:
            out_gs = out_gs_layers
        
        return out_gs, out_token_layers


