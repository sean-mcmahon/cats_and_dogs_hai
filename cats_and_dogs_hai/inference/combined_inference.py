from typing import Type
from pathlib import Path

import torch

from cats_and_dogs_hai.inference.inference_base import InferenceBase
from cats_and_dogs_hai.inference.classification_inference import ClassificationInference
from cats_and_dogs_hai.inference.segmentation_inference import SegmentationInference
from cats_and_dogs_hai.inference.combined_prediction import CombinedPrediction


class CombinedInference(InferenceBase):

    def __init__(
        self,
        classifer_weights: Path,
        segmenter_weights: Path,
        classifer: Type[ClassificationInference] = ClassificationInference,
        segmenter: Type[SegmentationInference] = SegmentationInference,
        device: str = "cpu",
    ):
        self.classifer = classifer(classifer_weights, device)
        self.segmenter = segmenter(segmenter_weights, device)

    def predict(self, image_raw: torch.Tensor) -> CombinedPrediction:
        pet_breed, pet_species = self.classifer.predict(image_raw)
        segmentation_mask = self.segmenter.predict(image_raw)
        return CombinedPrediction(pet_species, pet_breed, segmentation_mask)
