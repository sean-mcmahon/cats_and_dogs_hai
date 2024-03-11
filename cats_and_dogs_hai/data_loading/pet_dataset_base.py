from typing import Tuple
from typing import Optional
from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.load_image import load_image


class PetDatasetBase(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        images_directory: Path,
        dataset_info_file: Path,
        number_of_classes: Optional[int] = None,
        transforms: Optional[v2.Compose] = None,
    ):
        self.images_directory = images_directory
        self.dataset_df = pd.read_csv(dataset_info_file)
        self.number_of_classes = (
            len(pet_breeds_to_id) if number_of_classes is None else number_of_classes
        )
        self.transforms = transforms

    def __len__(self):
        return self.dataset_df.shape[0]

    def __getitem__(self, idx):
        datum_row = self.dataset_df.iloc[idx]
        image_path = self.images_directory / datum_row.Sample_ID / "image.jpg"
        image = load_image(image_path)
        label = self._load_label(datum_row)

        image, label = self._apply_transform(image, label)
        return image, label

    @abstractmethod
    def _apply_transform(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _load_label(self, database_row: pd.DataFrame):
        return None

    def __sudo_one_hot_encode(self, labels: torch.Tensor) -> torch.Tensor:
        # pylint: disable=not-callable
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.number_of_classes)
        multi_label_one_hot = one_hot.sum(dim=0).float()
        return multi_label_one_hot
