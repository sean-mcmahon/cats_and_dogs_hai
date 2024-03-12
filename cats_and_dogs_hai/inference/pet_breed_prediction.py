from typing import Optional

import torch
from torch import Tensor

from cats_and_dogs_hai.labels.pet_classes import pet_ids_to_breed
from cats_and_dogs_hai.inference.classification_prediction_base import ClassificationPredictionBase


class PetBreedPrediction(ClassificationPredictionBase):

    def __init__(self, logits: Tensor, class_id_lookup: Optional[dict[int, str]] = None):
        self._logits = logits
        self.class_id_lookup = pet_ids_to_breed if class_id_lookup is None else class_id_lookup

        self.__predictions: list[int] = self._logits.nonzero().squeeze().tolist()

    def _check_logits(self, logits:torch.Tensor):
        unique = torch.unique(logits, sorted=True)
        if not torch.any(unique == 1) or torch.any(unique == 0):
            raise ValueError(f"Logits should only contain 0 and 1:\n{logits.tolist()}")

    @property
    def names(self) -> list[str]:
        return [self.class_id_lookup[p] for p in self.__predictions]

    @property
    def integer(self) -> list[int]:
        return self.__predictions

    def __str__(self):
        return '{} has classes\n{}'.format(self.__class__, self.names)