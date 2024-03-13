from abc import ABCMeta
from abc import abstractmethod
from typing import Any

from torch import Tensor


class InferenceBase(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, image_raw: Tensor) -> Any:
        pass
