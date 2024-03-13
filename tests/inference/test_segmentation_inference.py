from pathlib import Path

import pytest
import pandas as pd
import torch

from cats_and_dogs_hai.data_loading.create_datasplits import CreateDataSplits
from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.data_loading.load_image import load_image
from cats_and_dogs_hai.inference.segmentation_inference import SegmentationInference


def get_model_path() -> Path:
    model_path = Path("tests/test_data/segmentation_efficientnet_epoch=1-step=16.ckpt")
    assert model_path.is_file()
    return model_path


def test_inference_constructor():
    classifier = SegmentationInference(get_model_path())


# test multiple thresholds to ensure prediction classes can handle cases with no predictions or more than two.
def test_validation_set():
    val_fn = dataset_info_filename.parent / "tiny_validation.csv"
    if not val_fn.is_file():
        splitter = CreateDataSplits(dataset_info_filename, dataset_info_filename.parent)
        splitter.create_tiny_db(10)

    val_df = pd.read_csv(val_fn)

    segmenter = SegmentationInference(get_model_path())

    for _, row in val_df.iterrows():
        image_path = dataset_info_filename.parent / "data" / row.Sample_ID / "image.jpg"
        assert image_path.is_file(), f'cannot find image at "{str(image_path)}'

        image = load_image(image_path)
        mask_prediction = segmenter.predict(image)

        assert mask_prediction.size() == image.size()[1:]
        assert torch.all(mask_prediction == 0) or torch.equal(
            torch.unique(mask_prediction, sorted=True), torch.tensor([0, 1])
        )
