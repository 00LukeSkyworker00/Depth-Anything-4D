import argparse, os
import numpy as np
import psutil
from tqdm import tqdm
import socket, shutil, glob
from datetime import datetime
import flow_vis

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from data import SpringDataset

from model import DepthAnything3

def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB | reserved={torch.cuda.memory_reserved()/1e9:.2f}GB"

def set_rnd_seed(seed:int):
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

def cleanup(rank):    
    try:
        destroy_process_group()
    except:
        pass
    # print(f"Process {rank} cleaned up.")

def Trainer(rank, args):
    is_main_rank = (rank == (args.world_size - 1))

    # Set random seed for reproducibility
    set_rnd_seed(args.seed)

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Setup logger
    writer = SummaryWriter(os.path.join(args.out_dir, 'logs'))
    losses = []
    def record_loss(loss_list:list[float]):
        losses.append(torch.tensor(loss_list)) # (N )

    def plt_loss(epoch:int, mode='Train'):
        assert mode in ['Train', 'Val']
        if len(losses) == 0:
            return
        loss_list = torch.stack(losses, dim=0).to(device).mean(dim=0).cpu() # (N,)
        loss = loss_list[0].item()
        epe = loss_list[1].item()
        smooth = loss_list[2].item()
        writer.add_scalar(f'{mode}/Total Loss', loss, epoch)
        writer.add_scalar(f'{mode}/EPE', epe, epoch)
        writer.add_scalar(f'{mode}/Smoothness', smooth, epoch)

        print(f'==== {mode} Result ====')
        print(f'{"Total Loss":<15}: {loss:12.3e}')
        print(f'{"EPE":<15}: {epe:12.3e}')
        print(f'{"Smoothness Loss":<15}: {smooth:12.3e}')
        losses.clear()

    def plt_lr(lr:torch.Tensor, step:int):
        writer.add_scalars('Learn Rate', {'value': lr[0]}, step)

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
        dataset = SpringDataset(root=data_glob, isVal=isVal, ep_len=args.ep_len)
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
    data_glob = os.path.join(args.data_dir, "train", "*")
    train_dataloader = init_dataloader(data_glob, isVal=False)
    val_dataloader = init_dataloader(data_glob, isVal=True)
    logger_print(f"[Dataloader] Train: {len(train_dataloader)} sets | Val: {len(val_dataloader)} sets")

    # Create optimizer
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=0)

    # Create schedular
    max_lr = args.max_lr * args.world_size
    min_epoch = (args.max_steps // len(train_dataloader)) + 1
    total_epochs = max(min_epoch, args.epoch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=max_lr, 
        total_steps=args.max_steps
    )

    logger_print(f"[Optimizer & Schedular] Max LR: {max_lr} | Max Steps: {args.max_steps} | Total Epochs: {total_epochs}")

    # Backup training scripts
    if is_main_rank:
        save_env(args.out_dir)
    logger_print(f"Training scripts backup to folder.")

    # Define loss fn
    def epe_loss(pred_flow, gt_flow, mask=None):
        diff = pred_flow - gt_flow
        epe = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)  # sum over 2 flow channels (u,v)
        if mask is not None:
            epe = epe * mask
            return epe.sum() / (mask.sum() + 1e-8)
        return epe.mean()
    
    def smoothness_loss(pred_flow, img):
        # pred_flow: (B, S-1, 2, H, W) - paired frames
        # img: (B, S, 3, H, W) - original frames
        # Match temporal dimensions: use first S-1 frames of img to align with flow
        img = img[:, :-1, :, :, :]  # (B, S-1, 3, H, W)
        # Compute spatial gradients: dx (horizontal), dy (vertical)
        dx = torch.abs(pred_flow[:, :, :, :, :-1] - pred_flow[:, :, :, :, 1:])  # (B,S-1,2,H,W-1)
        dy = torch.abs(pred_flow[:, :, :, :-1, :] - pred_flow[:, :, :, 1:, :])  # (B,S-1,2,H-1,W)
        # Compute image-based weights by averaging over RGB channels (dim=2)
        img_dx = torch.abs(img[:, :, :, :, :-1] - img[:, :, :, :, 1:])  # (B,S-1,3,H,W-1)
        img_dy = torch.abs(img[:, :, :, :-1, :] - img[:, :, :, 1:, :])  # (B,S-1,3,H-1,W)
        weights_x = torch.exp(-img_dx.mean(dim=2, keepdim=True))  # (B,S-1,1,H,W-1)
        weights_y = torch.exp(-img_dy.mean(dim=2, keepdim=True))  # (B,S-1,1,H-1,W)
        return (dx * weights_x).mean() + (dy * weights_y).mean()
    
    def reduce_loss(loss_tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist.get_world_size()
        return loss_tensor

    def compute_loss(pred_flow, gt_flow, img, λ=0.05, mask=None, isVal=False):
        # Create mask from gt_flow to exclude NaN/invalid regions
        if mask is None:
            # Mask is 1 where flow is valid (not NaN), 0 where invalid
            valid_mask = ~torch.isnan(gt_flow).any(dim=2, keepdim=False)  # (B, S-1, H, W)
            mask = valid_mask.float()
        
        # Replace NaN in gt_flow with 0 to prevent NaN propagation
        gt_flow = torch.nan_to_num(gt_flow, nan=0.0)
        # print(f"Pred Flow: {pred_flow.shape}, GT Flow: {gt_flow.shape}, Mask: {mask.shape}")
        epe = epe_loss(pred_flow, gt_flow, mask)
        smooth = smoothness_loss(pred_flow, img)

        if isVal:
            # Sync across ranks for logging
            epe = reduce_loss(epe)
            smooth = reduce_loss(smooth)

        loss = epe + λ * smooth
        if is_main_rank:
            record_loss([loss.detach(), epe.detach(), smooth.detach()])
        return loss

    # Start Training

    def step(sample, step:int, mode='train'):
        assert mode in ['train', 'val']
        isVal = (mode == 'val')

        # Load sample
        img = sample['img'].to(device)
        flow2d = sample['flow2d'].to(device)  # (B, S-1, 2, H, W)
        ixts = sample['ixts'].to(device)
        exts = sample['exts'].to(device)
        
        # Run model forward
        out = model(
            image=img, extrinsics=exts, intrinsics=ixts,
            export_feat_layers=[], infer_gs=False,
            use_ray_pose=False, ref_view_strategy="saddle_balanced"
        )

        # Compute flow loss
        pred_flow = out.flow['opticflow']  # (B, S-1, 2, H, W)
        loss = compute_loss(pred_flow=pred_flow, gt_flow=flow2d, img=img)
        
        if not isVal:
            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            plt_lr(scheduler.get_last_lr(), step)

        return (
            img[0,0].permute(1,2,0).detach().cpu().numpy(),
            pred_flow[0,0].permute(1,2,0).detach().cpu().numpy(),
            flow2d[0,0].permute(1,2,0).detach().cpu().numpy(),
        )

    def log_viz(img, pred, gt, epoch, mode='Train'):
        if is_main_rank:
            # Visualize and save results
            img_vis = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Replace NaN values in flow for visualization
            pred = np.nan_to_num(pred, nan=0.0)
            gt = np.nan_to_num(gt, nan=0.0)
            flow_pred_vis = flow_vis.flow_to_color(pred, convert_to_bgr=False)
            flow_gt_vis = flow_vis.flow_to_color(gt, convert_to_bgr=False)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_vis)
            axs[0].set_title('Input Image')
            axs[0].axis('off')
            axs[1].imshow(flow_pred_vis)
            axs[1].set_title('Predicted Flow')
            axs[1].axis('off')
            axs[2].imshow(flow_gt_vis)
            axs[2].set_title('Ground Truth Flow')
            axs[2].axis('off')
            plt.tight_layout()
            writer.add_figure(f'{mode}/Visualize', fig, epoch)
            plt.close()

    step_counter = 0
    for i in range(total_epochs):

        logger_print(f"Epoch {i}")

        train_dataloader.sampler.set_epoch(i)
        for sample in tqdm(train_dataloader, disable=not is_main_rank):
            img, pred, gt = step(sample, step_counter, mode='train')
            step_counter += 1
        plt_loss(i, mode='Train')
        log_viz(img, pred, gt, i, 'Train')

        val_dataloader.sampler.set_epoch(i)
        for sample in tqdm(val_dataloader, disable=not is_main_rank):
            img, pred, gt = step(sample, step_counter, mode='val')
        plt_loss(i, mode='Val')
        log_viz(img, pred, gt, i, 'Val')

    cleanup(rank)
    
def main():
    parser = argparse.ArgumentParser()
    # Directory configuration
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3-LARGE-1.1",
        help="Path to model directory for Huggingface",
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

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--batch", type=int, default=6, help="Total batch size")
    parser.add_argument("--num-worker", type=int, default=-1, help="Number of workers per rank, -1 for automatic detection.")
    parser.add_argument("--epoch", type=int, default=100, help="Total epochs for training")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max iterations")
    parser.add_argument("--max-lr", type=float, default=1e-5, help="Max learning rate")
    parser.add_argument("--ep-len", type=int, default=3, help="Episode length of the input clips")
    parser.add_argument("--custom-config", type=str, default="da3nested-giant-large-4d.yaml", help="Points to custom config file")

    # DDP setup
    parser.add_argument("--port", type=int, default=12355, help="Master port for DDP")
    
    # Flags
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--eval", action="store_true", help="Evaluate model only")

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_rnd_seed(args.seed)

    # Set environment
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    set_network(args.port)
  
    # Create output directory
    args.out_dir = os.path.join(args.out_dir, datetime.today().isoformat())
    ckpt_pth = os.path.join(args.out_dir,'ckpts')
    args.ckpt_pth = ckpt_pth
    os.makedirs(args.out_dir, exist_ok=False)
    os.makedirs(args.ckpt_pth, exist_ok=False)
    print(f"Output Directory: {args.out_dir}")

    # Spawn trainer
    args.world_size = torch.cuda.device_count()
    print(f'Spawning processes on {args.world_size} GPUs...')
    mp.spawn(Trainer, args=(args,), nprocs=args.world_size, join=True)

    print("Exiting main process...")

if __name__ == '__main__':
    main()
