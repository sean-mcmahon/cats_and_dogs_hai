import torch

from cats_and_dogs_hai.models.classification_model import create_classification_model
from cats_and_dogs_hai.labels.pet_classes import number_pet_breed_classes

def test_create_without_checkpoint():
    model = create_classification_model(number_pet_breed_classes)
    assert isinstance(model, torch.nn.Module)
    assert model.fc.out_features == number_pet_breed_classes

def test_create_with_checkpoint():
    model_path = 'tests/test_data/epoch=1-step=16.ckpt'
    model = create_classification_model(number_pet_breed_classes, weights_path=model_path)
    assert isinstance(model, torch.nn.Module)
    assert model.fc.out_features == number_pet_breed_classes