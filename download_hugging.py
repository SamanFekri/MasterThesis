from datasets import load_dataset
from dotenv import dotenv_values

config = dotenv_values(".env")

dataset = load_dataset("timbrooks/instructpix2pix-clip-filtered", cache_dir=config['DATASET_PATH_RAW'])