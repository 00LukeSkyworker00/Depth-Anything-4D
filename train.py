import math
import argparse, os
import numpy as np
import traceback
import psutil
from tqdm import tqdm
import socket, shutil, glob, random
from datetime import datetime
import flow_vis

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

# from dataset.spring import SpringDataset
# from dataset.mip_nerf_360 import MipNerf360
from dataset.real_estate_10k_256 import RealEstate10K256
from fused_ssim import fused_ssim

from logger import Logger, LoggerBase

from model import DepthAnything3

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

def set_rnd_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def set_network(port=12355, host='localhost', max_tries=100):
    tries = 0  # initialize tries
    while tries < max_tries:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                break
        print(f"Port {port} is in use, trying {port + 1}...")
        port += 1
        tries += 1
    else:
        raise RuntimeError(f"Could not find a free port after {max_tries} attempts.")

    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = host
    print(f"MASTER_ADDR={host}, MASTER_PORT={port}")
    return port

def save_env(out_dir:str):
    # Copy Python scripts to the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_pth = glob.glob(os.path.join(script_dir, '*.py'))
    for file in python_pth:
        shutil.copy(file, out_dir)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        destroy_process_group()

def Trainer(rank, args):
    set_rnd_seed(args.seed)

    try:
        process(rank, args)

    except KeyboardInterrupt:
        print(f"[Rank {rank}] KeyboardInterrupt", flush=True)
        raise

    except Exception as e:
        print(f"[Rank {rank}] failed: {e}", flush=True)
        traceback.print_exc()
        # Abort all workers immediately so mp.spawn does not hang in join().
        if dist.is_available() and dist.is_initialized():
            try:
                dist.abort()
            except Exception:
                pass
        raise

    finally:
        cleanup()


def process(rank, args):
    is_main_rank = (rank == (args.world_size - 1))
    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    def logger_print(msg:str):
        if is_main_rank:
            print(msg)
    
    # Initialize process group for DDP
    init_process_group(backend='nccl', rank=rank, world_size=args.world_size)
    torch.set_printoptions(precision=10) 

    # Set number of workers
    if args.num_worker < 0:
        num_cores = psutil.cpu_count(logical=False) # number of physical cores
        args.num_worker = max(0, (num_cores // args.world_size)-1)
        args.num_worker = min(args.num_worker, 8)
    logger_print(f"Log process on Rank {rank}, each rank has {args.num_worker} workers.")

    # Init model and wrap it in DDP
    model = DepthAnything3.from_pretrained(args.model_dir, custom_config=args.custom_config)
    model = model.to(device)
    
    # Model already has freeze() called in __init__ for NestedDepthAnything3Net
    # Just ensure training mode is enabled (for batch norm, dropout, etc.)
    model.train()
    model = DDP(model, device_ids=[rank])
    logger_print("Model loaded on device.")

    # Create dataloader
    def init_dataloader(data_glob:str, isVal:bool):
        dataset = RealEstate10K256(root=data_glob, isVal=isVal, ep_len=args.ep_len)
        shuffle = not isVal
        
        sampler = None
        if args.world_size > 0:
            sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank,
                shuffle=shuffle, seed=args.seed)
            shuffle = None

        return DataLoader(
            dataset=dataset, sampler=sampler, batch_size=args.batch, shuffle=shuffle,
            num_workers=args.num_worker, persistent_workers=True, prefetch_factor=4
        )
    train_dataloader = init_dataloader(args.data_dir, isVal=False)
    val_dataloader = init_dataloader(args.data_dir, isVal=True)
    logger_print(f"[Dataloader] Train: {len(train_dataloader)} iters | Val: {len(val_dataloader)} iters")

    # Setup logger
    if is_main_rank:
        train_sample = train_dataloader.dataset[0]
        val_sample = val_dataloader.dataset[0]
        logger = Logger(args, device, train_sample, val_sample)
    else:
        logger = LoggerBase(args, device, None, None)

    # Create optimizer
    params = [{'params': model.parameters()}]
    optimizer = optim.AdamW(params, lr=0)

    # Create schedular
    max_lr = args.max_lr * args.world_size
    min_epoch = (args.max_steps // len(train_dataloader)) + 1
    total_epochs = max(min_epoch, args.epoch)
    args.max_steps = total_epochs * len(train_dataloader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=max_lr, 
        total_steps=args.max_steps
    )

    logger_print(f"[Optimizer & Schedular] Max LR: {max_lr} | Max Steps: {args.max_steps} | Total Epochs: {total_epochs}")

    # Backup training scripts
    logger.save_env(args, model.module.config)
    logger_print(f"Training scripts backup to folder.")

    def compute_loss(output, img, is_reduce:bool=False):
        """Compute loss from model output."""
        if not hasattr(output, 'gs_render') or output.gs_render is None:
            return None, {}
        
        loss = 0

        ssim_lambda = 0.2
        depth_lambda = 5e-2
        
        # recon_loss = F.l1_loss(output.gs_render[0], img)
        # recon_loss *= (1.0 - ssim_lambda)
        # recon_loss = reduce_loss(recon_loss) if is_reduce else recon_loss
        # loss += recon_loss
        
        # ssim_loss = 1.0 - fused_ssim(output.gs_render[0], img)
        # ssim_loss *= ssim_lambda
        # ssim_loss = reduce_loss(ssim_loss) if is_reduce else ssim_loss
        # loss += ssim_loss
        
        depth_pred:torch.Tensor = output.gs_render[1]
        depth_gt:torch.Tensor = output.depth
        
        # depth_mask:torch.Tensor = output.depth_conf > 1.0
        # depth_pred = depth_pred[depth_mask]
        # depth_gt = depth_gt[depth_mask]
        
        # depth_pred = depth_pred.log()
        # depth_gt = depth_gt.log()

        depth_loss = F.l1_loss(depth_pred, depth_gt)
        depth_loss *= depth_lambda
        depth_loss = reduce_loss(depth_loss) if is_reduce else depth_loss
        loss += depth_loss
        
        loss_dict = {
            # 'recon': recon_loss.item(),
            # 'ssim': ssim_loss.item(),
            'depth': depth_loss.item(),
            }
        return loss, loss_dict

    def reduce_loss(loss:torch.Tensor):
        """Reduce loss dictionary across all ranks in DDP."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()        
        return loss

    # Start Training
    def step(sample, step:int, mode='train'):
        assert mode in ['train', 'val']
        isVal = (mode == 'val')

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
        
        # Compute loss
        loss, loss_dict = compute_loss(out, img, is_reduce=isVal)
        
        if loss is not None:  
            logger.record_loss(loss_dict)
            
            if not isVal:
                # Backward pass and optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                logger.plt_lr(scheduler.get_last_lr()[0], step)
        else:
            raise ValueError("loss is None")

    step_counter = 0
    for i in range(total_epochs):

        logger_print(f"Epoch {i}")

        model.train()
        train_dataloader.sampler.set_epoch(i)
        for sample in tqdm(train_dataloader, disable=not is_main_rank):
            step(sample, step_counter, mode='train')
            if step_counter % 10 == 0:
                logger.plt_loss(step_counter, mode='Train')
            if step_counter % 100 == 0:
                logger.log_viz(model, step_counter, 'Train')
            step_counter += 1

        model.eval()
        val_dataloader.sampler.set_epoch(i)
        with torch.no_grad():
            for sample in tqdm(val_dataloader, disable=not is_main_rank):
                step(sample, step_counter, mode='val')
        logger.plt_loss(i, mode='Val')
        logger.log_viz(model, i, 'Val')
        logger.save_model(model, optimizer, i)

    logger.cleanup()
    
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
        "--out-dir",
        help="Path to the output directory",
        required=True
    )
    parser.add_argument('--run_name', type=str, default='Scene Token Test')

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--batch", type=int, default=3, help="Total batch size")
    parser.add_argument("--num-worker", type=int, default=-1, help="Number of workers per rank, -1 for automatic detection.")
    parser.add_argument("--epoch", type=int, default=100, help="Total epochs for training")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max iterations")
    parser.add_argument("--max-lr", type=float, default=1e-5, help="Max learning rate")
    parser.add_argument("--ep-len", type=int, default=4, help="Episode length of the input clips")
    # parser.add_argument("--custom-config", type=str, default="da3nested-giant-large-4d.yaml", help="Points to custom config file")

    # DDP setup
    parser.add_argument("--port", type=int, default=12355, help="Master port for DDP")
    
    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--eval", action="store_true", help="Evaluate model only")

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

    # Set random seed for reproducibility
    set_rnd_seed(args.seed)

    # Set environment
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    set_network(args.port)
  
    # Spawn trainer
    args.world_size = torch.cuda.device_count()
    print(f'Spawning processes on {args.world_size} GPUs...')
    mp.spawn(Trainer, args=(args,), nprocs=args.world_size, join=True)

    print("Exiting main process...")

if __name__ == '__main__':
    main()
