from cats_and_dogs_hai.data_paths import dataset_directory
from cats_and_dogs_hai.data_paths import dataset_info_filename
from cats_and_dogs_hai.data_paths import dataset_images_directory

def test_data_directory():
    assert dataset_directory.is_dir()

def test_dataset_info():
    assert dataset_info_filename.is_file()
    assert str(dataset_info_filename).endswith('csv')

def test_dataset_images():
    assert dataset_images_directory.is_dir()