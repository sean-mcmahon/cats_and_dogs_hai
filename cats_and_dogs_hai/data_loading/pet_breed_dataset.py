from typing import Tuple
import ast

import pandas as pd
import torch

from cats_and_dogs_hai.data_loading.pet_dataset_base import PetDatasetBase
from cats_and_dogs_hai.labels import pet_breeds_to_id


class PetBreedsDataset(PetDatasetBase):
    def _apply_transform(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def _load_label(self, database_row: pd.DataFrame):
        breed_names: list[str] = ast.literal_eval(database_row.Breed)
        breed_ids = [pet_breeds_to_id[n] for n in breed_names]
        sudo_onehot_labels = self.__sudo_one_hot_encode(torch.tensor(breed_ids))
        return sudo_onehot_labels

    def __sudo_one_hot_encode(self, labels: torch.Tensor) -> torch.Tensor:
        # pylint: disable=not-callable
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.number_of_classes)
        multi_label_one_hot = one_hot.sum(dim=0).float()
        return multi_label_one_hot
