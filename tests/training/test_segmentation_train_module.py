from cats_and_dogs_hai.training.training_module_segmentation import SegmentationTrainModule
from cats_and_dogs_hai.data_loading.create_dataloaders import create_segmentation_dataloaders


def test_train_forward_pass():
    segmentation_module = SegmentationTrainModule()
    train_dls = create_segmentation_dataloaders(debug=True)

    image, label = next(iter(train_dls.train_dl))
    model = segmentation_module.model
    model.train(True)
    output = model(image)

    prediction = output["out"]
    loss = segmentation_module.criterion(prediction, label)
    assert prediction.shape == label.shape


def test_val_forward_pass():
    segmentation_module = SegmentationTrainModule()
    train_dls = create_segmentation_dataloaders(debug=True)

    image, label = next(iter(train_dls.val_dl))
    model = segmentation_module.model
    model.eval()
    output = model(image)

    prediction = output["out"]

    assert prediction.shape == label.shape


def test_train_step():
    segmentation_module = SegmentationTrainModule()
    train_dls = create_segmentation_dataloaders(debug=True)

    image, label = next(iter(train_dls.val_dl))

    loss = segmentation_module.training_step((image, label), 0)

def test_val_step():

    segmentation_module = SegmentationTrainModule()
    train_dls = create_segmentation_dataloaders(debug=True)

    image, label = next(iter(train_dls.val_dl))

    loss = segmentation_module.validation_step((image, label), 0)
    segmentation_module.on_validation_epoch_end()