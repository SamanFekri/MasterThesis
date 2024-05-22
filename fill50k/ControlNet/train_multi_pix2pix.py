import yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset.MultiControlNetDataset import MultiControlNetDataset
from cldm.model import create_model, load_state_dict
from cldm.wandb_logger import WandbImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from dotenv import dotenv_values
import torch


# Load the configuration file
with open('train_multi_pix2pix.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Configs from YAML
resume_path = config['model']['control_net_path']
batch_size = config['training']['batch_size']
logger_freq = config['training']['logger_freq']
learning_rate = config['model']['learning_rate']
sd_locked = config['model']['sd_locked']
only_mid_control = config['model']['only_mid_control']
validation_ratio = config['dataset']['validation_ratio']
seed = config['training']['seed']

pl.seed_everything(seed, workers=True)

# Model Creation
model = create_model(config['model']['config_file']).cpu()

lsd = load_state_dict(resume_path, location='cpu')

# Convert the list of tensors to a single tensor
repeated_tensor = torch.stack([torch.tensor(item).repeat(1, config['model']['num_hints'], 1, 1) for item in lsd['control_model.input_hint_block.0.weight']]).squeeze(1)

# Assign the corrected tensor to the state dictionary
lsd['control_model.input_hint_block.0.weight'] = repeated_tensor

model.load_state_dict(lsd, strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Checkpoint configuration
checkpoint_cb = ModelCheckpoint(
    monitor=config['checkpoint']['monitor'],
    dirpath=config['checkpoint']['path'],
    save_top_k=config['checkpoint']['save_top_k'],
    every_n_train_steps=config['checkpoint']['frequency'],
    filename=config['checkpoint']['filename'],
    mode=config['checkpoint']['mode']
)

print('Start image logger part')

# Dataset and Dataloader setup
dataset = MultiControlNetDataset(config['dataset']['path_processed'],
                            data_file=config['dataset']['data_file'],
                            source=config['dataset']['source'],
                            target=config['dataset']['target'],
                            prompt=config['dataset']['prompt'],
                            backward=config['dataset']['backward'])
train_size = int(len(dataset) * (1 - validation_ratio))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('Train dataset size:', len(train_dataset))
print('Validation dataset size:', len(val_dataset))

train_dataloader = DataLoader(train_dataset, num_workers=config['dataset']['num_workers'], batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=config['dataset']['num_workers'], batch_size=batch_size, shuffle=True)

logger = WandbImageLogger(batch_frequency=logger_freq, project_name=config['wandb']['project'], validation_size=len(val_dataset))

print('End image logger part')

print('Start PyTorch Lightning part')

# Trainer setup
trainer = pl.Trainer(
    devices=config['trainer']['devices'], 
    callbacks=[logger, checkpoint_cb], 
    accumulate_grad_batches=config['trainer']['accumulate_grad_batches'], 
    accelerator=config['trainer']['accelerator'],
    max_epochs=config['trainer']['max_epochs'], 
    val_check_interval=config['trainer']['validation_interval'],
    strategy=config['trainer']['strategy']
)
# if the strategy contains deepspeed, then we need to set the zero_force_ds_cpu_optimizer to False
if "deepspeed" in config['trainer']['strategy']:
    trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

print('End PyTorch Lightning part')

print('Start fitting part')

# Train
try:
    trainer.fit(model, train_dataloader, val_dataloader)
except Exception as e:
    print(f"Training failed due to {e}")
    wandb.finish()

print('End fitting part')
