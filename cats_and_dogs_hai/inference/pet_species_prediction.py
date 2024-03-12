from typing import Optional
from enum import IntEnum

import torch
from torch import Tensor

from cats_and_dogs_hai.labels.pet_classes import pet_breed_id_to_species_id
from cats_and_dogs_hai.labels.pet_classes import PetSpeciesIds
from cats_and_dogs_hai.inference.classification_prediction_base import ClassificationPredictionBase


class PetSpeciesPrediction(ClassificationPredictionBase):

    def __init__(
        self,
        logits: Tensor,
        class_id_lookup: Optional[dict[int, int]] = None,
        # pet_species_ids: Optional[IntEnum] = None,
    ):
        self._logits = logits
        self.pet_species_lookup = (
            pet_breed_id_to_species_id if class_id_lookup is None else class_id_lookup
        )
        # self.pet_species_ids = PetSpeciesIds if pet_species_ids is None else pet_species_ids

        self.__predictions: list[int] = self._logits.nonzero().squeeze().tolist()

    @property
    def integer(self) -> list[int]:
        return [int(self.pet_species_lookup[p]) for p in self.__predictions]

    @property
    def names(self) -> list[str]:
        return [str(self.pet_species_lookup[i].name) for i in self.integer]
