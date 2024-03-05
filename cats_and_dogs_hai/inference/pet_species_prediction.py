from typing import Optional
from typing import Type
from enum import IntEnum

import torch

from cats_and_dogs_hai.labels.pet_classes import pet_ids_to_breed
from cats_and_dogs_hai.labels.pet_classes import pet_breed_to_species_id
from cats_and_dogs_hai.labels.pet_classes import SpeciesIds

from cats_and_dogs_hai.inference.pet_breed_prediction import PetBreedPrediction


class PetSpeciesPrediction(PetBreedPrediction):
    pet_species_lookup: dict[str, int] = pet_breed_to_species_id

    def __init__(
        self,
        logits: torch.Tensor,
        pet_breed_lookup: Optional[dict[int, str]] = None,
        pet_species_lookup: Optional[dict[str, int]] = None,
        pet_species_ids: Optional[Type[IntEnum]] = SpeciesIds,
    ):
        super().__init__(logits, pet_breed_lookup=pet_breed_lookup)

        if pet_species_lookup is not None:
            self.pet_species_lookup = pet_species_lookup
        if pet_species_ids is not None:
            self.pet_species_ids = pet_species_ids

    @property
    def integer(self) -> list[int]:
        return [
            int(self.pet_species_lookup[b])
            for b in [self.pet_breed_lookup[p] for p in self.__predictions]
        ]

    @property
    def string(self) -> list[str]:
        int_lookup = [
            self.pet_species_lookup[b]
            for b in [self.pet_breed_lookup[p] for p in self.__predictions]
        ]
        return [str(self.pet_species_ids(i).name) for i in int_lookup]
