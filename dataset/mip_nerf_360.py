from PIL import Image
import cv2
import glob, os, random
from dataclasses import dataclass
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MipNerf360(Dataset):
    def __init__(self, root:str, isVal:bool, ep_len=2):
        super().__init__()

        self.root = os.path.join(root,"*")
        self.total_dirs = sorted(glob.glob(self.root))

        split = int(len(self.total_dirs) * 0.8)
        if not isVal:
            self.total_dirs = self.total_dirs[:split]
        else:
            self.total_dirs = self.total_dirs[split:]
                    
        """
        Mip-Nerf 360 Dataset folder structure.
        |---Scene_Name
            |---images      Original Images (5068 x 3326)
            |---images_2    Downscaled Images (2534 x 1663)
            |---images_4    Downscaled Images (1267 x 832)
            |---images_8    Downscaled Images (634 x 416)        
        """

        def get_pths(root:str, type:str, file_extension:str):
            pattern = os.path.join(root, type, f"*.{file_extension}")
            pths = sorted(glob.glob(pattern))
            random.shuffle(pths)
            count = len(pths)
            if count == 0:
                raise ValueError(f"No files in {root} matched pattern: {pattern}")
            return pths, count
                
        # chunk into episodes
        self.ep_pth = []
        for dir in self.total_dirs:
            frame_pths, frame_count = get_pths(dir, 'images_4','JPG')            
            for base_idx in range(0, frame_count, ep_len-1):
                if base_idx + ep_len >= frame_count:
                    break
                self.ep_pth.append([frame_pths[base_idx + i] for i in range(ep_len)])
        self.transform = T.Compose([
            T.RandomCrop(size=(504,504)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(self.ep_pth)
        
    def __getitem__(self, index):
        img_pth = self.ep_pth[index]
        imgs_pil = [
            self._resize(self._load_image(pth), 512) for pth in img_pth
        ]
        imgs_cpu = [self.transform(imgs) for imgs in imgs_pil]
        imgs_cpu = torch.stack(imgs_cpu, dim=0)
        return {
            'img': imgs_cpu, # (s,c,h,w)
        }

    def _load_image(self, img: np.ndarray | Image.Image | str) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            # Assume HxWxC uint8/RGB
            return Image.fromarray(img).convert("RGB")
        elif isinstance(img, Image.Image):
            return img.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
    def _resize(self, img: Image.Image, target_size: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)
        if shortest == target_size:
            return img
        scale = target_size / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)
