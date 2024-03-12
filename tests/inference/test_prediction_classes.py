from typing import Type

import pytest
import torch

from cats_and_dogs_hai.labels.pet_classes import number_pet_breed_classes
from cats_and_dogs_hai.labels.pet_classes import pet_ids_to_breed
from cats_and_dogs_hai.inference.classification_prediction_base import ClassificationPredictionBase
from cats_and_dogs_hai.inference.pet_breed_prediction import PetBreedPrediction
from cats_and_dogs_hai.inference.pet_species_prediction import PetSpeciesPrediction


@pytest.fixture
def logits_():
    torch.manual_seed(42)
    return (torch.rand(number_pet_breed_classes) > 0.7).to(torch.uint8)

@pytest.mark.parametrize(
    "prediction_class",
    [
        PetBreedPrediction,
        PetSpeciesPrediction,
    ],
)
def test_prediction_class_integer(prediction_class: Type[ClassificationPredictionBase], logits_:torch.Tensor):
    predictions = prediction_class(logits_)

    list_of_integers = predictions.integer

    assert isinstance(list_of_integers, list)
    assert all(isinstance(x, int) for x in list_of_integers)


@pytest.mark.parametrize(
    "prediction_class",
    [
        PetBreedPrediction,
        PetSpeciesPrediction,
    ],
)
def test_prediction_class_string(prediction_class: Type[ClassificationPredictionBase], logits_:torch.Tensor):
    predictions = prediction_class(logits_)

    list_of_outputs = predictions.names

    assert isinstance(list_of_outputs, list)
    assert all(isinstance(x, str) for x in list_of_outputs)
