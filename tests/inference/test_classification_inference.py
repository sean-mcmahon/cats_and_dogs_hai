from pathlib import Path

import pytest
import pandas as pd

from cats_and_dogs_hai.data_loading.create_datasplits import CreateDataSplits
from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.data_loading.load_image import load_image
from cats_and_dogs_hai.inference.classification_inference import ClassificationInference

def get_model_path() -> Path:
    model_path = Path('tests/test_data/classification_resnet_epoch=1-step=16.ckpt')
    assert model_path.is_file()
    return model_path

def test_inference_constructor():
    classifier = ClassificationInference(get_model_path())

# test multiple thresholds to ensure prediction classes can handle cases with no predictions or more than two.
@pytest.mark.parametrize("thresholds", [0.01, 0.5, 0.99])
def test_validation_set(thresholds):
    val_fn = dataset_info_filename.parent / "tiny_validation.csv"
    if not val_fn.is_file():
        splitter = CreateDataSplits(dataset_info_filename, dataset_info_filename.parent)
        splitter.create_tiny_db(10)

    val_df = pd.read_csv(val_fn)

    classifer = ClassificationInference(get_model_path(), prediction_threshold=thresholds)

    for _, row in val_df.iterrows():
        image_path = dataset_info_filename.parent / 'data' / row.Sample_ID / "image.jpg"
        assert image_path.is_file(), f'cannot find image at "{str(image_path)}'

        image = load_image(image_path)
        species_prediction, breed_prediction = classifer.predict(image)