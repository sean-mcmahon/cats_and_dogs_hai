from abc import ABCMeta
from abc import abstractmethod 

from torch import Tensor

class InferenceBase(metaclass=ABCMeta):
    
    @abstractmethod
    def predict(self, image_raw: Tensor):
        pass