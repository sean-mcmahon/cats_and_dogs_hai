
from cats_and_dogs_hai.labels import pet_breeds_to_id
from cats_and_dogs_hai.data_loading.create_dataloaders import create_classification_dataloaders
from cats_and_dogs_hai.training.training_module_classification import ClassificationTrainModule


def test_train_forward_pass():
    number_of_classes = len(pet_breeds_to_id)
    resnet_module = ClassificationTrainModule(number_classes=number_of_classes)
    train_dls = create_classification_dataloaders(debug=True)

    image, label = next(iter(train_dls.train_dl))
    model = resnet_module.model
    model.train(True)
    output = model(image)

    assert output.shape == label.shape == (1, number_of_classes)

def test_val_forward_pass():
    number_of_classes = len(pet_breeds_to_id)
    resnet_module = ClassificationTrainModule(number_classes=number_of_classes)
    train_dls = create_classification_dataloaders(debug=True)

    image, label = next(iter(train_dls.val_dl))
    model = resnet_module.model
    model.eval()
    output = model(image)

    assert output.shape == label.shape == (1, number_of_classes)