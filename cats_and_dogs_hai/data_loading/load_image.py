from pathlib import Path

from torchvision.io import read_image
import torch


def load_image(image_path: Path) -> torch.Tensor:
    if not image_path.is_file():
        raise FileNotFoundError(f"Cannot find image {str(image_path)}")
    return read_image(str(image_path))