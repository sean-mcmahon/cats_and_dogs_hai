from dataclasses import dataclass

import torch

from cats_and_dogs_hai.inference.pet_breed_prediction import PetBreedPrediction
from cats_and_dogs_hai.inference.pet_species_prediction import PetSpeciesPrediction


@dataclass
class CombinedPrediction:
    pet_species_prediction: PetSpeciesPrediction
    pet_breed_prediction: PetBreedPrediction
    pet_mask_prediction: torch.Tensor  # binary mask

    def __init__(
        self,
        pet_species_prediction: PetSpeciesPrediction,
        pet_breed_prediction: PetBreedPrediction,
        pet_mask_prediction: torch.Tensor,
    ):
        assert isinstance(pet_species_prediction, PetSpeciesPrediction) and isinstance(
            pet_breed_prediction, PetBreedPrediction
        ), (
            "Pet species and Pet breed must be of type "
            f"{PetSpeciesPrediction} and {PetBreedPrediction} respectively"
        )
        self.pet_species_prediction = pet_species_prediction
        self.pet_breed_prediction = pet_breed_prediction
        self.pet_mask_prediction = pet_mask_prediction
