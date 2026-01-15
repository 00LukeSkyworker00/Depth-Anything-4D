from PIL import Image
import h5py
import glob, os
from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np

import torch
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor

@dataclass
class FrameChunk:
    frame: List[Path]
    disp1: List[Path]
    disp2: List[Path]
    flow: List[Path]
    K: List[np.ndarray]
    w2c: List[np.ndarray]

class SpringDataset(Dataset):
    def __init__(self, root:str, isVal:bool, ep_len=9):
        super().__init__()

        self.root = root
        self.total_dirs = sorted(glob.glob(root))

        split = int(len(self.total_dirs) * 0.8)
        if not isVal:
            self.total_dirs = self.total_dirs[:split]
        else:
            self.total_dirs = self.total_dirs[split:]
                    
        """
        Spring Dataset naming convention. Use left view and forward for now.

        Frame: frame_left_0001.png
        Disparity: disp1_left_0001.dsp5
        SceneFlow: disp2_FW_left_0001.dsp5 / disp2_BW_left_0002.dsp5(ignored)
        OpticalFlow: flow_FW_left_0001.flo5 / flow_BW_left_0002.flo5(ignored)
        maps: detailmap(ignored) / matchmap(ignored) / rigidmap(ignored) / skymap(ignored)
        """

        def get_pths(root:str, pattern:str):
            glob = os.path.join(root, pattern)
            pths = sorted(glob.glob(glob))
            count = len(pths)
            if count == 0:
                raise ValueError(f"No files in {root} matched pattern: {pattern}")
            return pths, count
                
        # chunk into episodes
        self.ep_pth = []
        for dir in self.total_dirs:
            frame_pths, frame_count = get_pths(dir, 'frame_left_????.png')
            disp1_pths, disp1_count = get_pths(dir, 'disp1_left_????.dsp5')
            disp2_pths, disp2_count = get_pths(dir, 'disp2_FW_left_????.dsp5')
            flow_pths, flow_count = get_pths(dir, 'flow_FW_left_????.dsp5')

            # read cam pose txt
            K_all = load_intrinsics(os.path.join(dir, 'cam_data', 'intrinsics.txt'))
            w2c_all = load_extrinsics(os.path.join(dir, 'cam_data', 'extrinsics.txt'))

            len_matched = frame_count == disp1_count == (disp2_count+1) == (flow_count+1) == len(K_all) == len(w2c_all)
            if not len_matched:
                raise ValueError(
                    f"Length of data not matched in  {root}: \n"
                    f"Frame: {frame_count}, Disp1: {disp1_count}, Disp2: {disp2_count}, Flow: {flow_count}"
                    )
            
            for base_idx in range(0, frame_count, ep_len-1):
                if base_idx + ep_len > frame_count:
                    break
                chunk = FrameChunk(
                    frame=[Path(frame_pths[base_idx + i]) for i in range(ep_len)],
                    disp1=[Path(disp1_pths[base_idx + i]) for i in range(ep_len-1)],
                    disp2=[Path(disp2_pths[base_idx + i]) for i in range(ep_len-1)],
                    flow=[Path(flow_pths[base_idx + i]) for i in range(ep_len)],
                    K=[K_all[base_idx + i] for i in range(ep_len)],
                    w2c=[w2c_all[base_idx + i] for i in range(ep_len)]
                )
                self.ep_pth.append(chunk)
        self.input_processor = InputProcessor()
        self.out_processor = OutputProcessor()

        
    def __len__(self):
        return len(self.ep_pth)
        
    def __getitem__(self, index):

        chunk:FrameChunk = self.ep_pth[index]

        image = [Image.open(pth) for pth in chunk.frame]
        intrinsics = chunk.K
        extrinsics = chunk.w2c
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
                image,
                extrinsics.copy() if extrinsics is not None else None,
                intrinsics.copy() if intrinsics is not None else None,
                504,
                "lower_bound_resize",
            )
        
        disparity = [readDsp5Disp(pth) for pth in chunk.disp1]
        #TODO: convert disparity to depth.
        scene_flow = [readDsp5Disp(pth) for pth in chunk.disp2]
        optical_flow = [readFlo5Flow(pth) for pth in chunk.flow]

        return {
            'img': imgs_cpu,
            'pose': (intrinsics, extrinsics),
            'disp': disparity,
            'flow3d':scene_flow,
            'flow2d':optical_flow
        }
        

def load_extrinsics(path):
    # (N, 16)
    data = np.loadtxt(path, dtype=np.float64)
    assert data.shape[1] == 16, f"Expected 16 values per line, got {data.shape[1]}"
    # (N, 4, 4)
    T = data.reshape(-1, 4, 4)
    return T

def load_intrinsics(path):
    # (N, 4): fx fy cx cy
    intr = np.loadtxt(path, dtype=np.float64)
    assert intr.shape[1] == 4, f"Expected 4 values per line, got {intr.shape[1]}"
    fx, fy, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]

    K = np.zeros((intr.shape[0], 3, 3), dtype=np.float64)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    return K

def writeFlo5File(flow, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)


def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]


def writeDsp5File(disp, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("disparity", data=disp, compression="gzip", compression_opts=5)


def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]


def writePngMapFile(map_, filename):
    Image.fromarray(map_).save(filename)