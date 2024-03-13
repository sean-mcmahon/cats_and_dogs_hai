from typing import Tuple
from typing import Optional
from pathlib import Path

import torch
from torchvision.transforms import v2

from cats_and_dogs_hai.models.segmentation_model import create_segmentation_model
from cats_and_dogs_hai.data_loading.data_transforms import (
    image_segmentation_preprocessing_transforms,
)
from cats_and_dogs_hai.inference.inference_base import InferenceBase


class SegmentationInference(InferenceBase):

    def __init__(
        self,
        model_path: Path,
        device_name: str = "cpu",
        prediction_threshold: float = 0.7,
        preprocess_transforms: Optional[v2.Compose] = None,
    ):
        if preprocess_transforms is None:
            self.preprocess_transforms = image_segmentation_preprocessing_transforms
        self.threshold = prediction_threshold
        self.model = create_segmentation_model(model_path)
        device = torch.device(device_name)
        self.model.to(device)
        self.model.eval()

    def predict(self, image_raw: torch.Tensor) -> torch.Tensor:
        image_processed = self.preprocess_transforms(image_raw).unsqueeze(0)
        predictions = self.model(image_processed)
        predictions_ = torch.nn.functional.sigmoid(predictions["out"].squeeze())
        predictions_binary = (predictions_ > self.threshold).to(torch.uint8)

        return predictions_binary
