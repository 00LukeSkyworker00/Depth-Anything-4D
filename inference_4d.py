import numpy as np
import random, argparse, os, psutil
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler

# from dataset.spring import SpringDataset
# from dataset.mip_nerf_360 import MipNerf360
from dataset.real_estate_10k_256 import RealEstate10K256

from logger import Logger
from model import DepthAnything3

def set_rnd_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def Eval(args):
    
    # Set random seed for reproducibility
    set_rnd_seed(args.seed)

    # Set device for each process
    device = torch.device(f"cuda")
    
    # Set number of workers
    if args.num_worker < 0:
        num_cores = psutil.cpu_count(logical=False) # number of physical cores
        args.num_worker = max(0, num_cores-1)
        args.num_worker = min(args.num_worker, 8)
    print(f"{args.num_worker} workers.")

    # Init model
    model = DepthAnything3.from_pretrained(args.model_dir, custom_config=args.custom_config)
    model = model.to(device)
    ckpt = torch.load(args.ckpt_pth,map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded with checkpoint.")

    # Create dataloader
    def init_dataloader(data_glob:str):
        dataset = RealEstate10K256(root=data_glob, isVal=True, ep_len=args.ep_len)
        return DataLoader(
            dataset=dataset, batch_size=args.batch, shuffle=False,
            num_workers=args.num_worker, persistent_workers=True, prefetch_factor=4
        )
    val_dataloader = init_dataloader(args.data_dir)
    
    # Create logger
    logger = Logger(args, device, val_dataloader, val_dataloader)

    with torch.no_grad():
        iters = 0
        for sample in tqdm(val_dataloader):
            # Load sample
            img = sample['img'].to(device)
            ixts = None
            exts = None

            # Run model forward
            out = model(
                image=img, extrinsics=exts, intrinsics=ixts,
                export_feat_layers=[], infer_gs=False,
                use_ray_pose=False, ref_view_strategy="saddle_balanced"
            )
            logger.record_loss(out.loss_dict)
            logger.export_gsplat(out.gs, f'scene_{iters:04}.ply')
            iters += 1
            if iters >= args.num_iters:
                break


def main():
    parser = argparse.ArgumentParser()
    # Directory configuration
    parser.add_argument(
        "--model",
        default="large",
        choices=["giant_large", "giant", "large"],
        help="Type of model on Huggingface",
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the dataset directory",
        required=True
    )
    parser.add_argument(
        "--ckpt-dir",
        help="Path to the checkpoint directory",
        required=True
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--batch", type=int, default=3, help="Total batch size")
    parser.add_argument("--num-worker", type=int, default=-1, help="Number of workers, -1 for automatic detection.")
    parser.add_argument("--num-iter", type=int, default=-1, help="Number of iters to inference from dataset.")

    args = parser.parse_args()

    model_pth = {
        "giant_large": ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", "da3nested-giant-large-4d.yaml"),
        "giant": ("depth-anything/DA3-GIANT-1.1", "da3-giant-4d.yaml"),
        "large": ("depth-anything/DA3-LARGE-1.1", "da3-large-4d.yaml"),
        # "base": "depth-anything/DA3-BASE",
        # "small": "depth-anything/DA3-SMALL",
    }

    args.model_dir = model_pth[args.model][0]
    args.custom_config = model_pth[args.model][1]
  
    # Check checkpoint exist
    args.ckpt_pth = os.path.join(args.out_dir,'ckpts','best.pt')
    if os.path.exists(args.ckpt_pth):
        Eval(args)
    else:
        raise FileNotFoundError

if __name__ == '__main__':
    main()