from typing import Tuple

import pandas as pd
from torchvision import tv_tensors
import torch

from cats_and_dogs_hai.data_loading.pet_dataset_base import PetDatasetBase
from cats_and_dogs_hai.data_loading.load_image import load_image


class PetMaskDataset(PetDatasetBase):

    def _apply_transform(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label

    def _load_label(self, database_row: pd.DataFrame):
        mask_path = self.images_directory / database_row.Sample_ID / "mask.jpg"
        mask = load_image(mask_path)

        mask_tv_tensor = tv_tensors.Mask(mask, requires_grad=False)
        return mask_tv_tensor
