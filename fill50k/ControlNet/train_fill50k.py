from dataset.Fill50KDataset import Fill50KDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

DATASET_PATH = '../../../../dataset/fill50k'

# Configs
resume_path = '../../../models/control_sd15_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 1


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

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

# Adding the validation part
training_portion = 0.8
training_set, validation_set = random_split(dataset, [int(len(dataset) * training_portion), len(dataset) - int(len(dataset) * training_portion)])

# Dataloader
dataloader_train = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
dataloader_val = DataLoader(validation_set, batch_size=batch_size, num_workers=4, drop_last=True)


logger = ImageLogger(batch_frequency=logger_freq)

print('End image logger part')

print('Start pytorch Lightening part')
# import torch
# torch.set_float32_matmul_precision("medium")

# model.cuda()

# trainer = pl.Trainer(devices=1, precision="bf16-mixed", callbacks=[logger], accumulate_grad_batches=4, accelerator="gpu")  # But this will be 4x slower
trainer = pl.Trainer(devices=1, callbacks=[logger], accumulate_grad_batches=4, accelerator="gpu", strategy="deepspeed_stage_2_offload", max_epochs=max_epochs) #You might also try this strategy but it needs a python script (not interactive environment)

trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False

print('End pytorch Lightening part')

print('Start fitting part')

# Train!
trainer.fit(model, dataloader_train, dataloader_val)


