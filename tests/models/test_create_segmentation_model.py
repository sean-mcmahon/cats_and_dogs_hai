import torch

from cats_and_dogs_hai.models.segmentation_model import create_segmentation_model

def test_create_without_checkpoint():
    model = create_segmentation_model(weights_path=None)
    assert isinstance(model, torch.nn.Module)
    assert model.classifier.high_classifier.out_channels == 1
    assert model.classifier.low_classifier.out_channels == 1

# def test_create_with_checkpoint():
#     model_path = 'tests/test_data/epoch=1-step=16.ckpt'
#     model = create_segmentation_model(weights_path=model_path)
#     assert isinstance(model, torch.nn.Module)
#     assert model.classifier.high_classifier.out_channels == 1
#     assert model.classifier.low_classifier.out_channels == 1
