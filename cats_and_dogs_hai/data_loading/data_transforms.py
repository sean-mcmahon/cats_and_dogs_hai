from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torch

resnet_preprocessing_ = [
    # v2.Resize(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToTensor(),
]

segmentation_preprocessing = [
    # v2.Resize(520, interpolation=InterpolationMode.BILINEAR, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToTensor(),
]

classification_preprocessing_transforms = v2.Compose(resnet_preprocessing_)

image_classification_training_transforms = v2.Compose(
    resnet_preprocessing_
    + [
        v2.RandomRotation(degrees=(0, 30)),
        v2.RandomHorizontalFlip(p=0.5),
    ]
)


image_segmentation_preprocessing_transforms = v2.Compose(segmentation_preprocessing)

image_segmentation_training_transforms = v2.Compose(
    segmentation_preprocessing
    + [
        v2.RandomRotation(degrees=(0, 30)),
        v2.RandomHorizontalFlip(p=0.5),
    ]
)
