from typing import Optional
from pathlib import Path

import lightning as L

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.create_dataloaders import create_classification_dataloaders
from cats_and_dogs_hai.data_loading.create_dataloaders import create_segmentation_dataloaders
from cats_and_dogs_hai.data_loading.create_dataloaders import TrainingDataLoaders
from cats_and_dogs_hai.training.training_module_classification import ClassificationTrainModule
from cats_and_dogs_hai.training.training_module_segmentation import SegmentationTrainModule


def run_classification_train(
    max_epochs: int, save_dir: Path, debug: bool, accelerator: Optional[str] = None
):
    number_of_classes = len(pet_breeds_to_id)
    classification_module = ClassificationTrainModule(number_classes=number_of_classes)
    train_dls = create_classification_dataloaders(debug=debug)
    if accelerator is None:
        accelerator = "auto"
    _create_trainer_and_fit(train_dls, classification_module, max_epochs, accelerator, save_dir)


def run_segmentation_train(
    max_epochs: int, save_dir: Path, debug: bool, accelerator: Optional[str] = None
):
    segmentation_module = SegmentationTrainModule()
    train_dls = create_segmentation_dataloaders(debug=debug)
    if accelerator is None:
        accelerator = "auto"
    _create_trainer_and_fit(train_dls, segmentation_module, max_epochs, accelerator, save_dir)


def _create_trainer_and_fit(
    train_dls: TrainingDataLoaders,
    train_module: L.LightningModule,
    max_epochs: int,
    accelerator: str,
    save_dir: Path,
):
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=True,
        default_root_dir=save_dir,
        enable_checkpointing=True,
    )
    trainer.fit(
        train_module,
        train_dataloaders=train_dls.train_dl,
        val_dataloaders=train_dls.val_dl,
    )
