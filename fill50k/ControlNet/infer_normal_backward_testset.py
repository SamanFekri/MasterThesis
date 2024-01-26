from cldm.model import load_state_dict, create_model
import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from cldm.model import create_model, load_state_dict
from dataset.Fill50KDataset import Fill50KDataset
from inference import run_sampler
from share import *

import json

DATASET_PATH = '../../../test_dataset'
DATA_FILE = 'prompt.json'
SRC_KEY = 'dst'
DST_KEY = 'cln'
PROMPT_KEY = 'prompt'


# Load model from hugging face
checkpoint_path = '../../../models/pix2pix-epoch=02-step=184000.ckpt'
state_dict = load_state_dict(checkpoint_path)
model = create_model('models/scldm_v15.yaml').cpu()
model.load_state_dict(state_dict)
model.eval()

# Loop over test and make the pictures
with open(f'{DATASET_PATH}/{DATA_FILE}') as file:
    # Load 
    items = file.readlines()
    with tqdm(total=len(items)) as pbar:
        for item in items:
            item = item.strip()
            # Convert to dict
            item = json.loads(item)
            # prompt
            prompt = item[PROMPT_KEY]
            # read Image from source and It is RGB
            img = cv2.imread(f'{DATASET_PATH}/{item[SRC_KEY]}')
            # Make the new image and save it
            results = run_sampler(model, img, prompt, seed=42)
            image = Image.fromarray(results[0], "RGB")
            image.save(f'{DATASET_PATH}/{item[DST_KEY]}')
            pbar.update(1)
print("DONE")