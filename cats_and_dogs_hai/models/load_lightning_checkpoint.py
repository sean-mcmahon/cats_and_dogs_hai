
from pathlib import Path

import torch

def load_pytorch_lightning_checkpoint(model:torch.nn.Module, weights_path:Path):
    checkpoint = torch.load(str(weights_path))
    model_weights = checkpoint['state_dict']
    for key in list(model_weights.keys()):
        model_weights[key.replace('model.', '')] = checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint['state_dict'])