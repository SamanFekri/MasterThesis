import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb  # Import Weights & Biases


class WandbImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, project_name: str = ""):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        
        if project_name == "":
            wandb.init()
        else:
            wandb.init(project=project_name)

    @rank_zero_only
    def log_img_wandb(self, split, images, global_step, current_epoch, batch_idx):
        wandb_images = []
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.numpy().transpose((1, 2, 0))  # c,h,w -> h,w,c
            grid = (grid * 255).astype(np.uint8)
            pil_img = Image.fromarray(grid)
            # Convert to wandb.Image format for logging
            wandb_images.append(wandb.Image(pil_img, caption=f"{k}_e-{current_epoch:02}_gs-{global_step:06}_b-{batch_idx:06}"))

        # Log the list of wandb.Image objects
        if split == "val": 
            wandb.log({f"{split}_images": wandb_images})
        else:
            wandb.log({f"{split}_images": wandb_images}, step=global_step)
    
    @rank_zero_only
    def log_loss_wandb(self, split, loss, global_step, current_epoch, batch_idx):
        wandb.log({f"{split}_loss": loss}, step=global_step)
        

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if ((self.check_frequency(check_idx) or split == "val") and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            self.log_img_wandb(split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            if is_train:
                pl_module.train()
                
    def log_outputs(self, pl_module, outputs, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx)):  # batch_idx % self.batch_freq == 0
            is_train = pl_module.training
            self.log_loss_wandb(split, outputs['loss'].item(), pl_module.global_step, pl_module.current_epoch, batch_idx)
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_outputs(pl_module, outputs, batch_idx, split="train")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:
            return
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")
            self.log_outputs(pl_module, outputs, batch_idx, split="val")