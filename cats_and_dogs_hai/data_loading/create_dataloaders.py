from typing import Tuple
from typing import Type
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.create_datasplits import CreateDataSplits
from cats_and_dogs_hai.data_loading.data_transforms import image_classification_training_transforms
from cats_and_dogs_hai.data_loading.data_transforms import classification_preprocessing_transforms
from cats_and_dogs_hai.data_loading.data_transforms import (
    image_segmentation_preprocessing_transforms,
)
from cats_and_dogs_hai.data_loading.data_transforms import image_segmentation_training_transforms
from cats_and_dogs_hai.data_loading import create_dataloader
from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.data_loading.pet_dataset_base import PetDatasetBase
from cats_and_dogs_hai.data_loading.pet_mask_dataset import PetMaskDataset
from cats_and_dogs_hai.data_loading.pet_breed_dataset import PetBreedsDataset


@dataclass
class TrainingDataLoaders:
    train_dl: DataLoader
    val_dl: DataLoader

def create_classification_dataloaders(debug: bool = False) -> TrainingDataLoaders:
    return _create_dataloaders(
        PetBreedsDataset,
        image_classification_training_transforms,
        classification_preprocessing_transforms,
        debug=debug,
    )

def create_segmentation_dataloaders(debug: bool = False) -> TrainingDataLoaders:
    return _create_dataloaders(
        PetMaskDataset,
        image_segmentation_preprocessing_transforms,
        image_segmentation_training_transforms,
        debug=debug,
    )

def _create_dataloaders(
    dataset: Type[PetDatasetBase],
    training_augmentation: v2.Compose,
    validation_augmentation: v2.Compose,
    debug: bool = False,
):
    train_fn, val_fn = _get_train_val_splits(debug)
    train_dl = create_dataloader(dataset, 1, train_fn, transforms=training_augmentation)
    val_dl = create_dataloader(dataset, 1, val_fn, transforms=validation_augmentation)
    return TrainingDataLoaders(train_dl=train_dl, val_dl=val_dl)


def _get_train_val_splits(debug: bool) -> Tuple[Path, Path]:
    if debug:
        train_fn = dataset_info_filename.parent / "tiny_train.csv"
        val_fn = dataset_info_filename.parent / "tiny_validation.csv"
        if not train_fn.is_file() or not val_fn.is_file():
            splitter = CreateDataSplits(dataset_info_filename, dataset_info_filename.parent)
            splitter.create_tiny_db(10)
    else:
        save_dir = Path("data/cats_and_dogs/train_val_splits")
        save_dir.mkdir(exist_ok=True)
        splitter = CreateDataSplits(dataset_info_filename, save_dir)
        train_fn, val_fn = splitter(0.8)

    return train_fn, val_fn
