from .cldm import ControlLDM, ControlNet
from .ddim_hacked import DDIMSampler
import torch

class ExtendedControlLDM(ControlLDM):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        
    def validation_step(self, batch, batch_idx):
        loss_no_ema, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            loss_ema, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val_loss_no_ema", loss_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val_loss_ema", loss_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return {"loss": loss_no_ema}
        
        
class NegativeControlLDM(ExtendedControlLDM):
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        # cond_2 = {'c_concat': [cond["c_concat"][0] * 0], 'c_crossattn': [cond['c_crossattn'][0] * 0]}
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                     shape, cond, verbose=False,
                                     unconditional_guidance_scale=1.0,
                                     unconditional_conditioning=cond)
        return samples, intermediates