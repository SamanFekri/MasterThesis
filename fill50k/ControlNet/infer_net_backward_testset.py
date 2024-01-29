import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from tqdm import tqdm
import json

DATASET_PATH = '../../../test_dataset'
DATA_FILE = 'prompt.json'
SRC_KEY = 'dst'
DST_KEY = 'net'
PROMPT_KEY = 'prompt_reverse'

# Load model from hugging face
model_id = "lllyasviel/control_v11e_sd15_ip2p"
controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
generator = torch.manual_seed(42)


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
            img = Image.open(f'{DATASET_PATH}/{item[SRC_KEY]}')
            image = pipe(prompt, num_inference_steps=30, generator=generator, image=img).images[0]
            image.save(f'{DATASET_PATH}/{item[DST_KEY]}')
            pbar.update(1)
print("DONE")