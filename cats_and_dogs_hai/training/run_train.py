from typing import Optional
from pathlib import Path

import lightning as L

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.create_dataloaders import create_dataloaders
from cats_and_dogs_hai.training.training_module_classification import ResnetModule


def run_train(max_epochs: int, save_dir: Path, debug: bool, accelerator: Optional[str] = None):
    number_of_classes = len(pet_breeds_to_id)
    resnet_module = ResnetModule(number_classes=number_of_classes)
    train_dls = create_dataloaders(debug=debug)
    if accelerator is None:
        accelerator = "auto"
    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, logger=True, default_root_dir=save_dir
    )
    trainer.fit(
        resnet_module,
        train_dataloaders=train_dls.train_dl,
        val_dataloaders=train_dls.val_dl,
    )
