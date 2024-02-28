from typing import Dict
from enum import IntEnum
from .pets_breed_names import cat_breed_names
from .pets_breed_names import dog_breed_names


pet_breeds_to_id: Dict[str, int] = {}
index = 0
for names_enum in [cat_breed_names, dog_breed_names]:
    for name in names_enum:
        pet_breeds_to_id[name] = index
        index += 1
pet_ids_to_breed: Dict[int, str] = {value: key for key, value in pet_breeds_to_id.items()}


class SpeciesIds(IntEnum):
    Cat = 0
    Dog = 1


pet_breed_to_species_id: Dict[str, int] = {}
for catbreed in cat_breed_names:
    pet_breed_to_species_id[catbreed] = SpeciesIds.Cat
for dogbreed in dog_breed_names:
    pet_breed_to_species_id[dogbreed] = SpeciesIds.Dog
