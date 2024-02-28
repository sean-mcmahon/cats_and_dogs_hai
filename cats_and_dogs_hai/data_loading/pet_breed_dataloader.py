from pathlib import Path
from typing import Optional

import torch
from torchvision.transforms import v2

from cats_and_dogs_hai.data_loading.pet_breed_dataset import PetBreedsDataset
from cats_and_dogs_hai.data_loading.data_transforms import image_classification_transforms
from cats_and_dogs_hai.data_paths import dataset_images_directory
from cats_and_dogs_hai.data_paths import dataset_info_filename


def create_classification_dataloader(
    batch_size: int,
    dataset_info_file_: Optional[Path] = None,
    dataset_images_directory_: Optional[Path] = None,
    transforms: Optional[v2.Compose] = None,
) -> torch.utils.data.DataLoader:
    if dataset_images_directory_ is None:
        dataset_images_directory_ = dataset_images_directory
    if dataset_info_file_ is None:
        dataset_info_file_ = dataset_info_filename
    if transforms is None:
        transforms = image_classification_transforms

    dataset = PetBreedsDataset(dataset_images_directory_, dataset_info_file_, transforms=transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
