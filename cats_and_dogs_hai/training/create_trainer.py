from typing import Optional
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.create_datasplits import CreateDataSplits
from cats_and_dogs_hai.data_loading.data_transforms import image_classification_transforms
from cats_and_dogs_hai.data_loading import create_classification_dataloader
from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.training.training_module_classification import ResnetModule


def run_train(max_epochs: int, save_dir: Path, debug: bool, accelerator: Optional[str] = None):
    number_of_classes = len(pet_breeds_to_id)
    resnet_module = ResnetModule(number_classes=number_of_classes)
    train_dl, val_dl = create_dataloaders(debug=debug)
    if accelerator is None:
        accelerator = "auto"
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, logger=True, default_root_dir=save_dir
    )
    trainer.fit(resnet_module, train_dataloaders=train_dl, val_dataloaders=val_dl)


def create_dataloaders(debug: bool = False) -> tuple[DataLoader, DataLoader]:
    if debug:
        train_fn = dataset_info_filename.parent / "tiny_train.csv"
        val_fn = dataset_info_filename.parent / "tiny_validation.csv"
    else:
        save_dir = Path("data/cats_and_dogs/train_val_splits")
        save_dir.mkdir(exist_ok=True)
        splitter = CreateDataSplits(dataset_info_filename, save_dir)
        train_fn, val_fn = splitter(0.8)

    train_dl = create_classification_dataloader(
        1, train_fn, transforms=image_classification_transforms
    )
    val_dl = create_classification_dataloader(1, val_fn, transforms=None)
    return train_dl, val_dl
