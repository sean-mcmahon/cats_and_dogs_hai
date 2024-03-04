from typing import Optional
from typing import Tuple
from pathlib import Path

import numpy as np
import torch

from cats_and_dogs_hai.models.classification_model import create_classification_model
from cats_and_dogs_hai.labels.pet_classes import number_pet_breed_classes


class PetBreedPrediction:
    pass

class PetSpeciesPrediction:
    pass

class ClassificationInference():
    def __init__(self, model_path:Path, number_of_classes:Optional[int]=None):
        if number_of_classes is None:
            number_of_classes = number_pet_breed_classes
        self.number_of_classes = number_of_classes
        self.model = create_classification_model(self.number_of_classes, model_path)
        self.model.eval()

    def predict(self, img:np.ndarray) -> Tuple[PetSpeciesPrediction, PetBreedPrediction]:
        predictions = self.model(torch.Tensor(img))
        pet_species = self.process_species_prediction(predictions)
        pet_breed = self.process_breed_prediction(predictions)
        return pet_species, pet_breed

    def process_breed_prediction(self, prediction:torch.Tensor) -> PetBreedPrediction:
        return PetBreedPrediction()

    def process_species_prediction(self, prediction:torch.Tensor) -> PetSpeciesPrediction:
        return PetSpeciesPrediction()