from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, convert_zero_checkpoint_to_fp32_state_dict
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch


checkpoint_dir = '../../../models/checkpoints/fill50k-epoch=00-step= 8000.ckpt/checkpoint/mp_rank_00_model_states.pt'

from cldm.model import load_state_dict, create_model

state_dict = load_state_dict(checkpoint_dir)

model = create_model('models/cldm_v15.yaml')

model.load_state_dict(state_dict['module'])



model(dict(jpg=[],hint=[], txt="red circle inside blue bacjground"))