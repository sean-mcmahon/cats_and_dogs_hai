from typing import Optional
from pathlib import Path

import torch
import torchvision

from cats_and_dogs_hai.models.load_lightning_checkpoint import load_pytorch_lightning_checkpoint


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
