import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset.ControlNetDataset import ControlNetDataset
from cldm.model import create_model, load_state_dict
from cldm.wandb_logger import WandbImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from dotenv import dotenv_values

# Load the configuration file
with open('train_pix2pix.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATASET_PATH = config['dataset']['path_processed']

# Configs from YAML
resume_path = config['model']['control_net_path']
batch_size = config['training']['batch_size']
logger_freq = config['training']['logger_freq']
learning_rate = config['model']['learning_rate']
sd_locked = config['model']['sd_locked']
only_mid_control = config['model']['only_mid_control']
validation_ratio = config['dataset']['validation_ratio']
seed = config['training']['seed']
validation_interval = config['training']['validation_interval']
max_epochs = config['training']['max_epochs']
checkpoint_freq = config['training']['checkpoint_freq']
checkpoint_dir = config['training']['checkpoint_dir']

pl.seed_everything(seed, workers=True)

# Model Creation
model = create_model(config['model']['config_file']).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Checkpoint configuration
checkpoint_cb = ModelCheckpoint(
    monitor="train/loss_simple",
    dirpath=checkpoint_dir,
    save_top_k= 1,
    every_n_train_steps=checkpoint_freq,
    filename="pix2pix-{epoch:02}-{step:05}",
    mode="min"
)

print('Start image logger part')

# Dataset and Dataloader setup
dataset = ControlNetDataset(DATASET_PATH, backward=True)
train_size = int(len(dataset) * (1 - validation_ratio))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))

train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=True)

logger = WandbImageLogger(batch_frequency=logger_freq, project_name="pix2pix_lite", validation_size=len(val_dataset))

print('End image logger part')

print('Start PyTorch Lightning part')

# Trainer setup
trainer = pl.Trainer(
    devices=1, 
    callbacks=[logger, checkpoint_cb], 
    accumulate_grad_batches=4, 
    accelerator="gpu", 
    max_epochs=max_epochs, 
    val_check_interval=validation_interval
)
# trainer = pl.Trainer(devices=1, callbacks=[logger, checkpoint_cb], accumulate_grad_batches=4, accelerator="gpu", strategy="deepspeed_stage_2_offload", max_epochs=10000, val_check_interval=validation_interval) #You might also try this strategy but it needs a python script (not interactive environment)
# trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

print('End PyTorch Lightning part')

print('Start fitting part')

# Train
try:
    trainer.fit(model, train_dataloader, val_dataloader)
except Exception as e:
    print(f"Training failed due to {e}")
    wandb.finish()

print('End fitting part')
