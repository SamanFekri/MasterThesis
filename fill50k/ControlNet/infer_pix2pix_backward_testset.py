import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tqdm import tqdm
from PIL import Image
import json

DATASET_PATH = '../../../test_dataset'
DATA_FILE = 'prompt.json'
SRC_KEY = 'dst'
DST_KEY = 'p2p'
PROMPT_KEY = 'prompt_reverse'


# Load model from hugging face
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


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
            images = pipe(prompt, image=img, num_inference_steps=10, image_guidance_scale=1).images
            images[0].save(f'{DATASET_PATH}/{item[DST_KEY]}')
            pbar.update(1)
print("DONE")