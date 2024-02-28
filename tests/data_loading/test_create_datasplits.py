from pathlib import Path
import os

import pandas as pd

from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.data_loading import CreateDataSplits

def test_create_datasplits():
    save_path = Path('tests/data_loading/')
    full_db_df = pd.read_csv(dataset_info_filename)
    db_size = full_db_df.shape[0]
    train_split = 0.8

    splitter = CreateDataSplits(dataset_info_filename, save_path)
    splitter(train_split)

    expected_train_fn = save_path / 'train.csv'
    expected_val_fn = save_path / 'validation.csv'
    assert expected_train_fn.is_file()
    assert expected_val_fn.is_file()
    assert splitter.train_df.shape[0] == round(db_size * train_split)
    assert splitter.val_df.shape[0] == round(db_size * (1 - train_split))

    os.remove(expected_train_fn)
    os.remove(expected_val_fn)

def test_create_tiny_datasplits():
    # save_path = Path('data/cats_and_dogs/')
    save_path = Path('tests/data_loading/')
    expected_train_fn = save_path / 'tiny_train.csv'
    expected_val_fn = save_path / 'tiny_validation.csv'
    full_db_df = pd.read_csv(dataset_info_filename)
    db_size = full_db_df.shape[0]

    db_size = 10
    splitter = CreateDataSplits(dataset_info_filename, save_path)
    splitter.create_tiny_db(db_size)

    assert expected_train_fn.is_file()
    assert expected_val_fn.is_file()
    assert splitter.train_df.shape[0] == round(db_size * 0.8)
    assert splitter.val_df.shape[0] == round(db_size * 0.2)
    os.remove(expected_train_fn)
    os.remove(expected_val_fn)