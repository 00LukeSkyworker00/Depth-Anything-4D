from pathlib import Path
import time
import os, subprocess
import wandb
from datetime import datetime
import numpy as np
from collections import Counter
from typing import Any, Callable, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from depth_anything_3.utils.gsply_helpers import export_ply
from depth_anything_3.specs import Gaussians

class LoggerBase():
    def __init__(self, args, device, train_sample, val_sample):
        pass

    def save_env(self, args, config:Union[DictConfig, ListConfig]):
        pass

    def timed_print(self, prefix_msg:str):
        pass

    def load_best(self, best_loss):
        pass

    def save_best(self):
        pass

    def plt_lr(self, lr:torch.Tensor, step:int):
        pass

    def record_loss(self, d: dict[str, float]):
        pass

    def plt_loss(self, epoch:int, mode='Train') -> bool:
        pass
    
    def log_viz(self, model, epoch, mode='Train'):
        pass

    def save_model(self, model:nn.Module, optimizer:torch.optim.Optimizer, epoch):
        pass

    def cleanup(self):
        pass

class Logger(LoggerBase):
    def __init__(self, args, device, train_sample, val_sample, inference:bool=False):
        if inference:
            assert os.path.exists(args.out_dir), f"out_dir not found: {args.out_dir}"
            assert os.path.exists(args.ckpt_pth), f"ckpt_pth not found: {args.ckpt_pth}"
            self.out_dir = args.out_dir
            return
        # Create output directory
        timestamp = datetime.today().isoformat()
        self.out_dir = os.path.join(args.out_dir, timestamp)
        self.log_dir = os.path.join(self.out_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=False)
        
        ckpt_dir = os.path.join(self.out_dir,'ckpts')
        os.makedirs(ckpt_dir, exist_ok=False)
        self.best_pth = os.path.join(ckpt_dir,'best.pt')
        self.last_pth = os.path.join(ckpt_dir,'last.pt.tar')
        print(f"Output Directory: {self.out_dir}")

        self.device = device
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup WanDB
        wandb.login()
        self.wandb_run = wandb.init(
            project=args.run_name,
            config=vars(args),
            name=timestamp,
            settings=wandb.Settings(code_dir=".")
        )
        # Define custom step axes:
        self.wandb_run.define_metric("Train/*", step_metric="Train_step", hidden=True)
        self.wandb_run.define_metric("Val/*", step_metric="Val_step", hidden=True)

        self.train_vis = train_sample
        self.val_vis = val_sample

        self.start_time = time.time()
        self.best_loss = float('inf')
        self.has_best = False

        self.loss_dict = Counter()
        self.sample_count = 0

    def save_env(self, args, config:Union[DictConfig, ListConfig]):
        hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        diff = subprocess.check_output(["git", "diff"]).decode()
        self.writer.add_text("Git Hash", hash)
        self.writer.add_text("Git Diff", diff)
        self.writer.add_hparams(hparam_dict=vars(args), metric_dict={"final_loss": 0.0})
        OmegaConf.save(config, os.path.join(self.out_dir, "conf.yaml"))
    
    def timed_print(self, prefix_msg:str):
        now = time.time()
        duration = now - self.start_time
        print(f"{prefix_msg}| Time: {duration:.2f}s")

    def plt_lr(self, lr:torch.Tensor, step:int):
        self.writer.add_scalar('Optimizer/lr', lr, step)
        self.wandb_run.log({"Optimizer/lr": lr, "Train_step": step})

    def record_loss(self, d: dict[str, float]):
        self.loss_dict.update(d)
        self.sample_count += 1

    def plt_loss(self, epoch:int, mode='Train') -> bool:
        assert mode in ['Train', 'Val']
        if len(self.loss_dict) == 0:
            return
        total_loss = 0

        if mode=='Val':
            print(f'==== {mode} Loss ====')
        
        run_dict = {}
        for key, loss in self.loss_dict.items():
            tag = f'{mode}/{key} loss'
            mean_loss = loss / self.sample_count
            self.writer.add_scalar(tag, mean_loss, epoch)
            run_dict[tag] = mean_loss
            if mode=='Val':
                print(f'{key:<15} loss: { mean_loss:12.3e}')
            total_loss += mean_loss
        self.writer.add_scalar(f'{mode}/total loss', total_loss, epoch)
        run_dict[f'{mode}/total loss'] = total_loss
        run_dict[f'{mode}_step'] = epoch
        self.wandb_run.log(run_dict)

        if mode=='Val':
            print(f'{"total":<15} loss: { total_loss:12.3e}')
            if self.best_loss > total_loss:
                self.best_loss = total_loss
                self.has_best = True
        self.sample_count = 0
        self.loss_dict.clear()
    
    def log_viz(self, model:nn.Module, epoch, mode='Train'):
        # Visualize and save results
        if mode=='Train':
            x = self.train_vis['img'][None,].to(self.device)
        else:
            x = self.val_vis['img'][None,].to(self.device)

        out = model(
            image=x, extrinsics=None, intrinsics=None,
            export_feat_layers=[], infer_gs=False,
            use_ray_pose=False, ref_view_strategy="saddle_balanced"
        )
        vid = self.construct_vis(x, out).cpu()
        self.writer.add_video(f'{mode}/Visualization',vid, epoch)

        # Visualize contribution
        contrib_vis = out.gs_render['contrib_vis'][0].detach()
        in_band = out.gs_render['in_band'][0].detach().mean()
        self.writer.add_histogram(f'{mode}/contrib_vis', contrib_vis, epoch)
        self.writer.add_scalar(f'{mode}/contrib_in_band', in_band.item(), epoch)

        # Log to WanDB
        self.wandb_run.log({
            f'{mode}/Visualization': wandb.Video((vid*255.0).clip(0,255), fps=4, format="gif"),
            f'{mode}/contrib_vis': wandb.Histogram(contrib_vis.cpu().numpy()),
            f'{mode}/contrib_in_band': in_band.item(),
            f'{mode}_step': epoch
        })

        if mode == 'Val':
            self.export_gsplat(out.gs, f'{epoch:04}_Val.ply')
    
    def construct_vis(self, x:torch.Tensor, out:dict):
        img = x[0].detach()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        gs_render = out.gs_render['colors'][0].detach()
        depth = out.depth[0].unsqueeze(-3).repeat(1,3,1,1)
        depth_render = out.gs_render['depths'][0].unsqueeze(-3).repeat(1,3,1,1)
        depth = self.min_max_norm(depth).detach()
        depth_render = self.min_max_norm(depth_render).detach()
        vid = torch.stack([img, gs_render, depth, depth_render])
        return vid

    def min_max_norm(self, x:torch.Tensor):
        min = x.min()
        max = x.max()
        return (x - min)/(max-min)

    def save_model(self, model:nn.Module, optimizer:torch.optim.Optimizer, epoch):
        last = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch':epoch,
            'best_loss':self.best_loss
        }
        torch.save(last, self.last_pth)
        if self.has_best:
            torch.save(model.module.state_dict(), self.best_pth)
            self.has_best = False
    
    def export_gsplat(self, gsplat:Gaussians, file_name='gsplat.ply'):
        export_pth = os.path.join(self.out_dir,'export')
        os.makedirs(export_pth, exist_ok=True)
        export_ply(
            means=gsplat.means[0],
            scales=gsplat.scales[0],
            rotations=gsplat.rotations[0],
            harmonics=gsplat.harmonics[0],
            opacities=gsplat.opacities[0],
            path=Path(os.path.join(export_pth,file_name))
        )

    def cleanup(self):
        artifact_best = wandb.Artifact("best", type="model")
        artifact_best.add_file(self.best_pth)
        artifact_best.ttl = None
        self.wandb_run.log_artifact(artifact_best)

        artifact_ckpt = wandb.Artifact("ckpt", type="checkpoint")
        artifact_ckpt.add_file(self.last_pth)
        artifact_ckpt.ttl = None
        self.wandb_run.log_artifact(artifact_ckpt)

        artifact_log = wandb.Artifact("tensorboard", type="log")
        artifact_log.add_dir(self.log_dir)
        artifact_log.ttl = None
        self.wandb_run.log_artifact(artifact_log)

        self.writer.close()
        self.wandb_run.finish()

    
def print_vram(device, msg=""):
    """
    Print the current VRAM usage of the specified device.
    """
    if torch.cuda.current_device() != 0:
        return
    if msg != "":
        print(msg)
    if device.type == 'cuda':
        print(f"Device: {device} / VRAM Usage: {torch.cuda.memory_allocated(device) / (1024 ** 2)} MB / VRAM Allocated: {torch.cuda.memory_reserved(device) / (1024 ** 2)} MB")
        # smi = subprocess.check_output(['nvidia-smi']).decode('utf-8')
        # print(smi)
    else:
        print("Device is not a CUDA device.")

def lin_exp_scheduler(optimizer:optim.Optimizer, warmup_steps:int, decay_rate:float)->optim.lr_scheduler.SequentialLR:
    warmup = optim.lr_scheduler.LinearLR(optimizer, 0.0, 1.0, total_iters=warmup_steps)
    decay = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup,decay],
        milestones=[warmup_steps]
    )
    return scheduler