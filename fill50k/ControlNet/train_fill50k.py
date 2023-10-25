from dataset.Fill50KDataset import Fill50KDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from cldm.logger import ImageLogger
from cldm.wandb_logger import WandbImageLogger
from cldm.model import create_model, load_state_dict
# from pytorch_lightning.loggers import WandbLogger
import wandb


DATASET_PATH = '../../../../dataset/fill50k'

# Configs
resume_path = '../../../models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# WanDB config
# from pytorch_lightning.loggers.wandb import WandbLogger
# wandb_logger = WandbLogger(project="MNIST", log_model="all")


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# wandb_logger.watch(model, log_freq=logger_freq)

print('Start image logger part')

# Misc
dataset = Fill50KDataset(DATASET_PATH)
print(len(dataset))
item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=logger_freq)
logger = WandbImageLogger(batch_frequency=logger_freq)

print('End image logger part')

print('Start pytorch Lightening part')
# import torch
# torch.set_float32_matmul_precision("medium")

# model.cuda()

# trainer = pl.Trainer(devices=1, precision="bf16-mixed", callbacks=[logger], accumulate_grad_batches=4, accelerator="gpu")  # But this will be 4x slower
trainer = pl.Trainer(devices=1, callbacks=[logger], accumulate_grad_batches=4, accelerator="gpu", strategy="deepspeed_stage_2_offload", max_epochs=1) #You might also try this strategy but it needs a python script (not interactive environment)

trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

print('End pytorch Lightening part')

print('Start fitting part')

# Train!
try:
    trainer.fit(model, dataloader)
except Exception as e:
    wandb.finish()


