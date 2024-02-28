
from torchvision.transforms import v2
import torch

resnet_preprocessing_transforms = v2.Compose([
    # v2.Resize(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToTensor()
])

image_classification_transforms = v2.Compose([
    v2.RandomRotation(degrees=(0, 30)),
    v2.RandomHorizontalFlip(p=0.5),
    # v2.Resize(256),
    v2.ToDtype(torch.float32, scale=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToTensor()
])