from PIL import Image
import cv2, os, random
from dataclasses import dataclass
import numpy as np
from pathlib import PosixPath

import torch
from torch.utils.data import Dataset
from .utils import read_video

class RealEstate10K256(Dataset):
    def __init__(self, root:str, isVal:bool, ep_len=2):
        super().__init__()

        self.frame_skip = 20
        self.ep_len = ep_len
        safe_len = self.frame_skip * (self.ep_len + 2)

        metadata_pth = os.path.join(root,'metadata','training.pt')
        data = torch.load(metadata_pth, weights_only=False)
        self.metadata = []
        for i in range(len(data["video_paths"])):
            if len(data["video_pts"][i]) > safe_len:
                val = {key: data[key][i] for key in data.keys()}
                pths:PosixPath = val['video_paths']
                val['video_paths'] = os.path.join(root,pths.relative_to("data/real-estate-10k"))
                self.metadata.append(val)
        clip = int(len(self.metadata) * 0.5)
        self. metadata = self.metadata[:clip]
        split = int(len(self.metadata) * 0.8)
        if not isVal:
            self.metadata = self.metadata[:split]
        else:
            self.metadata = self.metadata[split:]
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, index):
        vid_pts = self.metadata[index]['video_pts']
        vid_len = len(vid_pts)
        start_min = self.frame_skip-1
        start_max = vid_len - (self.frame_skip * self.ep_len)
        start = random.randrange(start_min, start_max)
        end = start + (self.frame_skip * self.ep_len - 1)
        vid = read_video(
            self.metadata[index]['video_paths'], 
            start_pts=vid_pts[start].item(), 
            end_pts=vid_pts[end].item(),
            resize=504
        )
        vid = vid[::self.frame_skip]

        return {
            'img': vid, # (s,c,h,w)
        }