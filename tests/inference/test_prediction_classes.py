from typing import Callable
from typing import Type

import pytest
import torch

from cats_and_dogs_hai.labels.pet_classes import number_pet_breed_classes
from cats_and_dogs_hai.labels.pet_classes import pet_ids_to_breed
from cats_and_dogs_hai.labels.pet_classes import PetSpeciesIds
from cats_and_dogs_hai.labels.pet_classes import pet_breeds_to_id
from cats_and_dogs_hai.labels.pet_classes import pet_breed_id_to_species_id
from cats_and_dogs_hai.inference.classification_prediction_base import ClassificationPredictionBase
from cats_and_dogs_hai.inference.pet_breed_prediction import PetBreedPrediction
from cats_and_dogs_hai.inference.pet_species_prediction import PetSpeciesPrediction


def prediction_breed_names() -> list[str]:
    cat_name = "Egyptian_Mau"
    dog_name = "english_cocker_spaniel"
    return [cat_name, dog_name]


def prediction_breed_ids() -> list[int]:
    names = prediction_breed_names()
    return [pet_breeds_to_id[n] for n in names]


def prediction_species_ids() -> list[int]:
    breed_ids = prediction_breed_ids()
    species_ids = [pet_breed_id_to_species_id[b_id] for b_id in breed_ids]
    return species_ids


def prediction_species_names() -> list[str]:
    species_ids = prediction_species_ids()
    return [str(PetSpeciesIds(i).name) for i in species_ids]


@pytest.fixture
def prediction_logits():
    # cat_name, dog_name = prediction_breed_names()
    logits = torch.zeros(number_pet_breed_classes, dtype=torch.uint8)
    for i in prediction_breed_ids():
        logits[i] = 1
    return logits


@pytest.mark.parametrize(
    "prediction_class, expected_outputs",
    [
        (PetBreedPrediction, prediction_breed_ids),
        (PetSpeciesPrediction, prediction_species_ids),
    ],
)
def test_prediction_class_integer(
    prediction_class: Type[ClassificationPredictionBase],
    expected_outputs: Callable[[], list[int]],
    prediction_logits: torch.Tensor,
):
    predictions = prediction_class(prediction_logits)

    list_of_integers = predictions.integer

    expected_output_integers = expected_outputs()
    assert isinstance(list_of_integers, list)
    assert all(isinstance(x, int) for x in list_of_integers)
    assert all(
        [
            predicted_id == expected_id
            for predicted_id, expected_id in zip(list_of_integers, expected_output_integers)
        ]
    ), f"Results integers \n{list_of_integers}\nDoes not equal expected\n{expected_output_integers}"


@pytest.mark.parametrize(
    "prediction_class, expected_outputs",
    [
        (PetBreedPrediction, prediction_breed_names),
        (PetSpeciesPrediction, prediction_species_names),
    ],
)
def test_prediction_class_string(
    prediction_class: Type[ClassificationPredictionBase],
    expected_outputs: Callable[[], list[str]],
    prediction_logits: torch.Tensor,
):
    predictions = prediction_class(prediction_logits)

    list_of_outputs = predictions.names

    expected_names = expected_outputs()
    assert isinstance(list_of_outputs, list)
    assert all(isinstance(x, str) for x in list_of_outputs)
    assert all(
        pred_name == expected_name
        for pred_name, expected_name in zip(list_of_outputs, expected_names)
    ), f"Names mismatch, expected:\n{expected_names}\nPredicted:\n{list_of_outputs}"
