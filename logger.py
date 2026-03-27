
from datetime import time
import os
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, args, device):
        self.writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
        self.device = device
        os.makedirs(self.export_dir, exist_ok=True)

        self.start_time = time.time()

        self.loss_names = []
        self.loss_list = []

        self.best_loss = float('inf')
        self.best_ari = 0
        self.best_arifg = 0

        self.ari = []
        self.ari_fg = []

    def timed_print(self, prefix_msg:str):
        now = time.time()
        duration = now - self.start_time
        print(f"{prefix_msg}| Time: {duration:.2f}s")

    def load_best(self, best_score):
        self.best_loss = best_score[0]
        self.best_ari = best_score[1]
        self.best_arifg = best_score[2]

    def save_best(self):
        return (self.best_loss, self.best_ari, self.best_arifg)

    def plt_lr(self, lr:torch.Tensor, step:int):
        self.writer.add_scalars('Learn Rate', {'value': lr[0]}, step)

    def record_loss(self, loss_list:torch.Tensor):
        """
        loss_list: [N,]
        """
        self.loss_list.append(loss_list.unsqueeze(-1).detach().cpu())

    def plt_loss(self, epoch:int, mode='Train') -> bool:
        assert mode in ['Train', 'Val']  

        loss_list = self.caculate_mean(self.loss_list) # [N,]

        result = {}
        total_loss = loss_list.sum().item()
        result['total'] = total_loss
        for i, name in zip(loss_list, self.loss_names):
            # print(name,i)
            result[name] = i.item()

        isBest = False
        if mode == 'Val' and self.best_loss > total_loss:
            self.best_loss = total_loss
            isBest = True

        
        self.writer.add_scalars(f'{mode} Loss', result, epoch)

        self.timed_print(f"{mode} Loss: {total_loss}")
        
        self.loss_list = []

        return isBest

    # def record_metrics(self, ari, ari_fg):
    #     self.ari.append(ari.detach().cpu())
    #     self.ari_fg.append(ari_fg.detach().cpu())

    # def plt_metrics(self, epoch) -> tuple[bool,bool]:
    #     ari = self.caculate_mean(self.ari)
    #     ari_fg = self.caculate_mean(self.ari_fg)

    #     self.writer.add_scalars('Metrics', {
    #         'ARI': ari,
    #         'ARI-FG': ari_fg,
    #     }, epoch)

    #     isBestAri = False
    #     isBestArifg = False

    #     if self.best_ari < ari:
    #         self.best_ari = ari
    #         isBestAri = True
    #     if self.best_arifg < ari_fg:
    #         self.best_arifg = ari_fg
    #         isBestArifg = True

    #     self.ari = []
    #     self.ari_fg = []
        
    #     return isBestAri, isBestArifg
    
    def print_eval(self):

        total_loss = self.caculate_mean(self.total_loss)
        p_loss = self.caculate_mean(self.p_loss)
        c_loss = self.caculate_mean(self.c_loss)
        ari = self.caculate_mean(self.ari)
        ari_fg = self.caculate_mean(self.ari_fg)
        
        self.total_loss = []
        self.p_loss = []
        self.c_loss = []
        self.ari = []
        self.ari_fg = []

        print('==== Eval Result ====')
        print(f'{"Total Loss":<15}: {total_loss:12.3e}')
        print(f'{"Position Loss":<15}: {p_loss:12.3e}')
        print(f'{"Color Loss":<15}: {c_loss:12.3e}')
        print(f'{"ARI":<15}: {ari:12.1%}')
        print(f'{"ARI-FG":<15}: {ari_fg:12.1%}')
    
    def caculate_mean(self,stack:list[torch.Tensor]):
        return torch.cat(stack,dim=-1).to(self.device).mean(dim=-1).cpu()

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