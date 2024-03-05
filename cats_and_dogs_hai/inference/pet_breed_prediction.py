from typing import Optional

import torch

from cats_and_dogs_hai.labels.pet_classes import pet_ids_to_breed


class PetBreedPrediction:
    # Shape [1, number_pet_breed_classes]
    # Dtype torch.int
    logits: torch.Tensor
    pet_breed_lookup: dict[int, str] = pet_ids_to_breed

    def __init__(self, logits: torch.Tensor, pet_breed_lookup: Optional[dict[int, str]] = None):
        self.logits = logits
        if pet_breed_lookup is not None:
            self.pet_breed_lookup = pet_breed_lookup

        if not torch.equal(torch.unique(self.logits, sorted=True), torch.tensor([0, 1])):
            raise ValueError(f"Logits should only contain 0 and 1:\n{self.logits.tolist()}")
        self.__predictions: list[int] = self.logits.nonzero().tolist()

    @property
    def string(self) -> list[str]:
        return [self.pet_breed_lookup[p] for p in self.__predictions]

    @property
    def integer(self) -> list[int]:
        return self.__predictions

    def __str__(self):
        return '{} has classes\n{}'.format(self.__class__, self.string)