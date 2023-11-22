from .cldm import ControlLDM

class ExtendedControlLDM(ControlLDM):
    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(control_stage_config, control_key, only_mid_control, *args, **kwargs)
        print("I used the Extended Control LDM")
        
    def validation_step(self, batch, batch_idx):
        print("Validation Step")
        loss_no_ema, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            loss_ema, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val_loss_no_ema", loss_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val_loss_ema", loss_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        
