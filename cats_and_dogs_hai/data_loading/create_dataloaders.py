from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.create_datasplits import CreateDataSplits
from cats_and_dogs_hai.data_loading.data_transforms import image_classification_transforms
from cats_and_dogs_hai.data_loading.data_transforms import resnet_preprocessing_transforms
from cats_and_dogs_hai.data_loading import create_classification_dataloader
from cats_and_dogs_hai.data_paths import dataset_info_filename


@dataclass
class TrainingDataLoaders:
    train_dl: DataLoader
    val_dl: DataLoader


def create_dataloaders(debug: bool = False) -> TrainingDataLoaders:
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

    train_dl = create_classification_dataloader(
        1, train_fn, transforms=image_classification_transforms
    )
    val_dl = create_classification_dataloader(1, val_fn, transforms=resnet_preprocessing_transforms)
    return TrainingDataLoaders(train_dl=train_dl, val_dl=val_dl)
