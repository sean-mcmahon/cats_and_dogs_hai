from typing import Optional
from abc import ABCMeta
from abc import abstractmethod

from torch import Tensor


class ClassificationPredictionBase(metaclass=ABCMeta):
    # Shape [1, number_pet_breed_classes]
    _logits: Tensor

    @abstractmethod
    def __init__(self, logits: Tensor, class_id_lookup: Optional[dict[int, str | int]] = None):
        pass

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    @abstractmethod
    def names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def integer(self) -> list[int]:
        pass
