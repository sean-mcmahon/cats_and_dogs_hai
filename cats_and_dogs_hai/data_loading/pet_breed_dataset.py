from typing import Optional
from pathlib import Path
import ast

import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
import torch

from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.load_image import load_image


class PetBreedsDataset(Dataset):
    def __init__(
        self, images_directory: Path, dataset_info_file: Path,
        number_of_classes:Optional[int]=None,
        transforms: Optional[v2.Compose] = None
    ):
        self.images_directory = images_directory
        self.dataset_df = pd.read_csv(dataset_info_file)
        self.number_of_classes = len(pet_breeds_to_id) if number_of_classes is None else number_of_classes
        self.transforms = transforms

    def __len__(self):
        return self.dataset_df.shape[0]

    def __getitem__(self, idx):
        datum_row = self.dataset_df.iloc[idx]
        datum_id = datum_row.Sample_ID
        breed_names: list[str] = ast.literal_eval(datum_row.Breed)
        breed_ids = [pet_breeds_to_id[n] for n in breed_names]
        image_path = self.images_directory / datum_id / "image.jpg"
        image = load_image(image_path)
        if self.transforms is not None:
            image = self.transforms(image)

        sudo_onehot_labels = self.sudo_one_hot_encode(torch.tensor(breed_ids))

        return image, sudo_onehot_labels


    def sudo_one_hot_encode(self, labels:torch.tensor) -> torch.tensor:
        # pylint: disable=not-callable
        one_hot =  torch.nn.functional.one_hot(labels, num_classes=self.number_of_classes)
        multi_label_one_hot = one_hot.sum(dim=0).float()
        return multi_label_one_hot
