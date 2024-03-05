from typing import Tuple
from pathlib import Path

import pandas as pd


class CreateDataSplits():
    def __init__(self, data_info_csv:Path, save_directory:Path):
        self.full_data_df = pd.read_csv(data_info_csv)
        self.save_dir = save_directory

    def __call__(self, trainset_percentage:float=0.8) -> Tuple[Path, Path]:
        self.__create_df_splits(self.full_data_df, trainset_percentage)

        val_fn = self.save_dir / "validation.csv"
        train_fn = self.save_dir / "train.csv"
        self.train_df.to_csv(train_fn, index=False)
        self.val_df.to_csv(val_fn, index=False)
        return train_fn, val_fn

    def __create_df_splits(self, df:pd.DataFrame, trainset_percentage:float):
        self.train_df = df.sample(frac=trainset_percentage)
        self.val_df = df.drop(self.train_df.index)


    def create_tiny_db(self, dataset_size:int) -> Tuple[Path, Path]:
        fraction = dataset_size / self.full_data_df.shape[0]
        dataset_subset = self.full_data_df.sample(frac=fraction, random_state=42)
        self.__create_df_splits(dataset_subset, 0.8)
        train_fn = self.save_dir / "tiny_train.csv"
        val_fn = self.save_dir / "tiny_validation.csv"
        self.train_df.to_csv(train_fn, index=False)
        self.val_df.to_csv(val_fn, index=False)
        return train_fn, val_fn