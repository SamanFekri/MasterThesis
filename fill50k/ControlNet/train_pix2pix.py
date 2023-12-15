from dataset.ControlNetDataset import ControlNetDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
# from cldm.logger import ImageLogger
from cldm.wandb_logger import WandbImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
import wandb


DATASET_PATH = '../../../../dataset/pix2pix_lite/output'

# Configs
resume_path = '../../../models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 10000
learning_rate = 5 * 1e-5
sd_locked = True
only_mid_control = False
validation_ratio = 0.0002
seed = 42
validation_interval = 10000

checkpoint_freq = 30000
checkpoint_dir ='../../../models/checkpoints'


pl.seed_everything(seed, workers=True)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/scldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Config the checkpoint
checkpoint_cb = ModelCheckpoint(
    monitor="train/loss_simple",
    dirpath=checkpoint_dir,
    save_top_k= 1,
    every_n_train_steps=checkpoint_freq,
    filename="pix2pix-{epoch:02}-{step:05}",
    mode="min"
)
    

print('Start image logger part')

# Create dataset
dataset = ControlNetDataset(DATASET_PATH, backward=True)
# Split dataset to train and validation
train_size = int(len(dataset) * (1 - validation_ratio))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Show the size of train and validation dataset
print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

# Create dataloader
train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# logger = ImageLogger(batch_frequency=logger_freq)
logger = WandbImageLogger(batch_frequency=logger_freq, project_name="pix2pix_lite", validation_size=len(val_dataset))

print('End image logger part')

print('Start pytorch Lightening part')
# import torch
# torch.set_float32_matmul_precision("medium")

# model.cuda()

# trainer = pl.Trainer(devices=1, precision="bf16-mixed", callbacks=[logger], accumulate_grad_batches=4, accelerator="gpu")  # But this will be 4x slower
trainer = pl.Trainer(devices=1, callbacks=[logger, checkpoint_cb], accumulate_grad_batches=4, accelerator="gpu", strategy="deepspeed_stage_2_offload", max_epochs=10000, val_check_interval=validation_interval) #You might also try this strategy but it needs a python script (not interactive environment)

trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

print('End pytorch Lightening part')

print('Start fitting part')

# Train!
try:
    trainer.fit(model, train_dataloader, val_dataloader)
except Exception as e:
    wandb.finish()


