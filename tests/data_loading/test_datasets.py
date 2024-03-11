import ast

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest


from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.labels.pet_classes import pet_ids_to_breed
from cats_and_dogs_hai.data_loading.pet_breed_dataset import PetBreedsDataset
from cats_and_dogs_hai.data_loading.pet_mask_dataset import PetMaskDataset
from cats_and_dogs_hai.data_paths import dataset_images_directory
from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.data_loading.data_transforms import image_classification_training_transforms
from cats_and_dogs_hai.data_loading.data_transforms import image_segmentation_training_transforms


def plot_images(img1: np.ndarray, img2: np.ndarray):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()


def sudo_onehot_to_labels(y_tensor: np.ndarray) -> list[str]:
    label_ids = np.where(y_tensor >= 1)[0]
    return [pet_ids_to_breed[idx] for idx in label_ids]


def test_pet_breed_dataset_constructor():
    dataset = PetBreedsDataset(dataset_images_directory, dataset_info_filename)


@pytest.mark.parametrize(
    "test_dataset_cls",
    [PetBreedsDataset, PetMaskDataset],
    ids=["Pet Breeds Classification Dataset", "Pet Mask Segmentation Dataset"],
)
def test_pet_dataset_len(test_dataset_cls):
    dataset = test_dataset_cls(dataset_images_directory, dataset_info_filename)
    dataDF = pd.read_csv(dataset_info_filename)
    assert len(dataset) == dataDF.shape[0]


@pytest.mark.parametrize(
    "test_dataset_cls",
    [PetBreedsDataset, PetMaskDataset],
    ids=["Pet Breeds Classification Dataset", "Pet Mask Segmentation Dataset"],
)
def test_pet_dataset_getitem(test_dataset_cls):
    dataset = test_dataset_cls(dataset_images_directory, dataset_info_filename)
    idx = 10
    dataDF = pd.read_csv(dataset_info_filename)
    expected_image = plt.imread(dataset_images_directory / dataDF.iloc[idx].Sample_ID / "image.jpg")
    expected_label = ast.literal_eval(dataDF.iloc[idx].Breed)

    datum = dataset[idx]

    dataset_img = datum[0].numpy().transpose(1, 2, 0)
    assert dataset_img.shape == expected_image.shape
    assert np.array_equiv(dataset_img, expected_image)


def test_pet_breed_dataset_label():
    dataset = PetBreedsDataset(dataset_images_directory, dataset_info_filename)
    idx = 10
    dataDF = pd.read_csv(dataset_info_filename)
    expected_image = plt.imread(dataset_images_directory / dataDF.iloc[idx].Sample_ID / "image.jpg")
    expected_label = ast.literal_eval(dataDF.iloc[idx].Breed)

    datum = dataset[idx]

    assert sudo_onehot_to_labels(datum[1].numpy()) == expected_label


def test_pet_mask_dataset_label():
    dataset = PetMaskDataset(dataset_images_directory, dataset_info_filename)
    idx = 10
    dataDF = pd.read_csv(dataset_info_filename)
    expected_label = plt.imread(dataset_images_directory / dataDF.iloc[idx].Sample_ID / "mask.jpg")

    datum = dataset[idx]

    result_label = np.squeeze(datum[1].numpy().transpose(1, 2, 0))
    # plot_images(result_label, expected_label)
    assert np.array_equal(result_label, expected_label)


def test_pet_breed_dataset_transform_getitem():
    dataset = PetBreedsDataset(
        dataset_images_directory,
        dataset_info_filename,
        transforms=image_classification_training_transforms,
    )
    idx = 6494
    dataDF = pd.read_csv(dataset_info_filename)
    expected_image = plt.imread(dataset_images_directory / dataDF.iloc[idx].Sample_ID / "image.jpg")
    expected_label = ast.literal_eval(dataDF.iloc[idx].Breed)

    datum = dataset[idx]
    result_image = datum[0].numpy().transpose(1, 2, 0)
    # plot_images(expected_image, result_image)
    assert result_image.shape == expected_image.shape
    assert sudo_onehot_to_labels(datum[1].numpy()) == expected_label


def test_pet_mask_dataset_transform_getitem():

    dataset = PetMaskDataset(
        dataset_images_directory,
        dataset_info_filename,
        transforms=image_segmentation_training_transforms,
    )
    idx = 6494
    dataDF = pd.read_csv(dataset_info_filename)
    expected_image = plt.imread(dataset_images_directory / dataDF.iloc[idx].Sample_ID / "image.jpg")
    expected_label = plt.imread(dataset_images_directory / dataDF.iloc[idx].Sample_ID / "mask.jpg")

    datum = dataset[idx]
    result_image = datum[0].numpy().transpose(1, 2, 0)
    result_label = datum[1].numpy().transpose(1, 2, 0)
    # plot_images(expected_image, result_image)
    # plot_images(result_image, result_label)
    assert result_image.shape[:2] == result_label.shape[:2]
