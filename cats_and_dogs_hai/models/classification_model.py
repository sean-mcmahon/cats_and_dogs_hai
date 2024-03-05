from typing import Optional
from pathlib import Path

import torch
import torchvision

def create_classification_model(
    number_classes: int, weights_path: Optional[Path] = None
) -> torch.nn.Module:
    if weights_path is None:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        num_fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc_features, number_classes)
    else:
        model = torchvision.models.resnet18(weights=None, num_classes=number_classes)
        load_pytorch_lightning_checkpoint(model, weights_path)
        if model.fc.out_features != number_classes:
            raise ValueError(
                f"Final layer should output {number_classes} classes not {model.fc.out_features}"
            )
    return model


def load_pytorch_lightning_checkpoint(model:torch.nn.Module, weights_path:Path):
    checkpoint = torch.load(str(weights_path))
    model_weights = checkpoint['state_dict']
    for key in list(model_weights.keys()):
        model_weights[key.replace('model.', '')] = checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint['state_dict'])