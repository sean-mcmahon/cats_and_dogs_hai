from pathlib import Path
from typing import Optional

import torch
from torchvision.transforms import v2

from cats_and_dogs_hai.data_loading.pet_breed_dataset import PetBreedsDataset
from cats_and_dogs_hai.data_paths import dataset_images_directory
from cats_and_dogs_hai.data_paths import dataset_info_filename


def create_classification_dataloader(
    batch_size: int,
    dataset_info_file_: Path,
    transforms: v2.Compose | None = None,
    dataset_images_directory_: Optional[Path] = None,
) -> torch.utils.data.DataLoader:
    if dataset_images_directory_ is None:
        dataset_images_directory_ = dataset_images_directory

    dataset = PetBreedsDataset(dataset_images_directory_, dataset_info_file_, transforms=transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
