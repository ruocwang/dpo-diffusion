## 
import torch
import torch.nn.functional as F
import copy
from utils.exp_utils import json_load, text_load


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def load_prompts(path):
    if '.json' in path:
        prompts = json_load(path)
    elif '.txt' in path:
        prompts = text_load(path)
    
    return prompts
