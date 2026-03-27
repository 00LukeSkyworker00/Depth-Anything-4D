import torch
import torch.nn as nn
import torch.nn.functional as F

from addict import Dict

# import SEA-RAFT
import sys
sys.path.append("/home/skyworker/workspace")
from sea_raft import *

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        super().__init__()
        args = Dict(args)
        self.args = args
        self.output_dim = args.dim * 2
        
        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        # self.fnet = conv1x1(2 * args.dim, args.dim)
        # self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=True)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        if args.iters > 0:
            # self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=True)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim, cdim=args.dim)

    def forward(self, fmap:torch.Tensor, cnet:torch.Tensor, iters=None):
        flow_predictions = []
        info_predictions = []
        B, S, _, _, _ = fmap.shape

        if iters is None:
            iters = self.args.iters

        # padder = InputPadder(feat0.shape)
        # feat0, feat1 = padder.pad(feat0, feat1)
        # print("SEA-RAFT/ Start: ", vram())
        feat0 = fmap[:, :-1].flatten(0,1)
        feat1 = fmap[:, 1:].flatten(0,1)
        N, _, H, W = feat0.shape
        dilation = torch.ones(N, 1, H, W, device=feat0.device)
        
        # run the context network
        cnet0 = cnet[:, :-1].flatten(0,1)
        cnet1 = cnet[:, 1:].flatten(0,1)
        cnet_01 = torch.cat([cnet0,cnet1], dim=1)
        cnet = self.init_conv(cnet_01)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        flow_predictions.append(flow_up)
        info_predictions.append(info_up)
        # print("SEA-RAFT/ Init : ", vram())

        if self.args.iters > 0:
            # run the feature network
            # fmap1_8x = self.fnet(image1)
            # fmap2_8x = self.fnet(image2)
            corr_fn = CorrBlock(feat0, feat1, self.args)
        # print("SEA-RAFT/ CorrBlock : ", vram())

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=feat0.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
        # print("SEA-RAFT/ Iter : ", vram())

        # for i in range(len(info_predictions)):
        #     flow_predictions[i] = padder.unpad(flow_predictions[i])
        #     info_predictions[i] = padder.unpad(info_predictions[i])

        return {
            'preds': flow_predictions,
            'infos': info_predictions
        }
    
     
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords2 = coords_grid(N, H//8, W//8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)