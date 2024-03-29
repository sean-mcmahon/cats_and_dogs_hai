import numpy as np

from cats_and_dogs_hai.labels import cat_breed_names
from cats_and_dogs_hai.labels import dog_breed_names
from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.labels import pet_breed_id_to_species_id
from cats_and_dogs_hai.labels import PetSpeciesIds


def test_breed_labels_length():
    total_number_classes = len(cat_breed_names) + len(dog_breed_names)
    assert total_number_classes == len(pet_breeds_to_id)


def test_breed_label_ids_valid():
    unique_ids = np.unique(list(pet_breeds_to_id.values()))
    assert unique_ids.shape[0] == len(pet_breeds_to_id)
    assert max(unique_ids) == len(pet_breeds_to_id) - 1
    assert min(unique_ids) == 0

def test_breed_to_species_length():
    total_number_classes = len(cat_breed_names) + len(dog_breed_names)
    assert total_number_classes == len(pet_breed_id_to_species_id)

def test_breed_to_species_has_all_cats():
    for catbreed in cat_breed_names:
        cat_id = pet_breeds_to_id[catbreed]
        assert pet_breed_id_to_species_id[cat_id] == PetSpeciesIds.Cat

def test_breed_to_species_has_all_dogs():
    for dogbreed in dog_breed_names:
        dogbreed_id = pet_breeds_to_id[dogbreed]
        assert pet_breed_id_to_species_id[dogbreed_id] == PetSpeciesIds.Dog