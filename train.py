import argparse, os
import numpy as np
import psutil
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

from model import DepthAnything3

def Trainer(rank, world_size, args):
    is_main_rank = rank == (world_size-1)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    def logger_print(msg:str):
        if is_main_rank:
            print(msg)

    logger_print(f"Log process on Rank {rank}.")

    # Initialize process group for DDP
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.set_printoptions(precision=10) 

    # Set number of workers
    if args.num_worker < 0:
        num_cores = psutil.cpu_count(logical=False) # number of physical cores
        num_workers = max(0, (num_cores // world_size)-1)
    print(f"Rank {rank} using {num_workers} workers.")

    # Init model and wrap it in DDP
    model = DepthAnything3.from_pretrained(args.model_dir)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Create dataloader

    # Create optimizer

    # Resume from checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(args.out_dir,'checkpoints', 'last.ckpt')
    if os.path.exists(checkpoint_path):
        logger_print(f"Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if logger is not None:
            logger.load_best(checkpoint['best_metrics'])
        logger_print(f"Checkpoint loaded from {args.out_dir}")
    else:
        logger_print(f"Save training environment to output folder...")
        if is_main_rank:
            save_env(cfg)

    # Define loss fn

def main():
    parser = argparse.ArgumentParser()
    # Directory configuration
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Path to model directory for Huggingface (default: depth-anything/DA3NESTED-GIANT-LARGE)",
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--out-dir",
        default="./tmp/out",
        help="Path to the output directory (default: ./tmp/out)"
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--batch", type=int, default=24, help="Total batch size")
    parser.add_argument("--num-worker", type=int, default=1, help="Number of workers per rank, set to -1 for automatic detection.")
    parser.add_argument("--epoch", type=int, default=200, help="Max epoch")
    parser.add_argument("--max-steps", type=int, default=50000, help="Max iterations")

if __name__ == '__main__':
    main()
