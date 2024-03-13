from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

from cats_and_dogs_hai.models.classification_model import create_classification_model
from cats_and_dogs_hai.labels.pet_classes import number_pet_breed_classes
from cats_and_dogs_hai.data_loading.data_transforms import classification_preprocessing_transforms
from cats_and_dogs_hai.inference.inference_base import InferenceBase
from cats_and_dogs_hai.inference.pet_breed_prediction import PetBreedPrediction
from cats_and_dogs_hai.inference.pet_species_prediction import PetSpeciesPrediction


class ClassificationInference(InferenceBase):
    def __init__(
        self,
        model_path: Path,
        device_name: str = "cpu",
        prediction_threshold: float = 0.7,
        preprocess_transforms: Optional[v2.Compose] = None,
        number_of_classes: Optional[int] = None,
    ):
        if number_of_classes is None:
            number_of_classes = number_pet_breed_classes
        if preprocess_transforms is None:
            self.preprocess_transforms = classification_preprocessing_transforms

        self.threshold = prediction_threshold
        self.number_of_classes = number_of_classes
        device = torch.device(device_name)
        self.model = create_classification_model(self.number_of_classes, model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, image_raw: torch.Tensor) -> tuple[PetBreedPrediction, PetSpeciesPrediction]:
        image_processed = self.preprocess_transforms(image_raw).unsqueeze(0)
        predictions = self.model(image_processed)
        predictions_binary = (torch.nn.functional.sigmoid(predictions) >= self.threshold).type(
            torch.int
        )
        pet_species = self.process_species_prediction(predictions_binary)
        pet_breed = self.process_breed_prediction(predictions_binary)
        return pet_breed, pet_species

    def process_breed_prediction(self, prediction: torch.Tensor) -> PetBreedPrediction:
        return PetBreedPrediction(prediction)

    def process_species_prediction(self, prediction: torch.Tensor) -> PetSpeciesPrediction:
        return PetSpeciesPrediction(prediction)
