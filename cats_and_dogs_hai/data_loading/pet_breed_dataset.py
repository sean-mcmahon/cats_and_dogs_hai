from typing import Optional
from pathlib import Path
import ast

import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
import torch

from cats_and_dogs_hai.labels import pet_breeds_to_id


class PetBreedsDataset(Dataset):
    def __init__(
        self, images_directory: Path, dataset_info_file: Path, transforms: Optional[v2.Compose] = None
    ):
        self.images_directory = images_directory
        self.dataset_df = pd.read_csv(dataset_info_file)
        self.transforms = transforms

    def __len__(self):
        return self.dataset_df.shape[0]

    def __getitem__(self, idx):
        datum_row = self.dataset_df.iloc[idx]
        datum_id = datum_row.Sample_ID
        breed_names: list[str] = ast.literal_eval(datum_row.Breed)
        breed_ids = [pet_breeds_to_id[n] for n in breed_names]
        image_path = self.images_directory / datum_id / "image.jpg"
        image = read_image(str(image_path))
        if self.transforms is not None:
            image = self.transforms(image)

        return image, torch.tensor(breed_ids)
