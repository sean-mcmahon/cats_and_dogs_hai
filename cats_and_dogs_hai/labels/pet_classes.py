from enum import IntEnum
from .pets_breed_names import cat_breed_names
from .pets_breed_names import dog_breed_names


pet_breeds_to_id = {}
index = 0
for names_enum in [cat_breed_names, dog_breed_names]:
    for name in names_enum:
        pet_breeds_to_id[name] = index
        index += 1


class SpeciesIds(IntEnum):
    Cat = 0
    Dog = 1


breed_to_species_id = {}
for catbreed in cat_breed_names:
    breed_to_species_id[catbreed] = SpeciesIds.Cat
for dogbreed in dog_breed_names:
    breed_to_species_id[dogbreed] = SpeciesIds.Dog
