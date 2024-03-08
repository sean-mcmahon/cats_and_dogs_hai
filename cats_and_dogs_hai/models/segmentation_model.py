from typing import Optional
from pathlib import Path

import torch
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from cats_and_dogs_hai.models.load_lightning_checkpoint import load_pytorch_lightning_checkpoint


def create_segmentation_model(weights_path:Optional[Path]=None) -> torch.nn.Module:
    num_classes = 1
    if weights_path is None:
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)
        low_channels = model.classifier.low_classifier.in_channels
        inter_channels = model.classifier.high_classifier.in_channels
        model.classifier.low_classifier = torch.nn.Conv2d(low_channels, num_classes, 1)
        model.classifier.high_classifier = torch.nn.Conv2d(inter_channels, num_classes, 1)
    else:
        model = lraspp_mobilenet_v3_large(weights=None, num_classes=1)
        load_pytorch_lightning_checkpoint(model, weights_path)

    return model