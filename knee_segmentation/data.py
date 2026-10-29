from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class KneeSegmentationDataset(Dataset):
    """Dataset for paired knee X-ray images and binary segmentation masks."""

    def __init__(self, images_dir: str | Path, masks_dir: str | Path, image_size: tuple[int, int] = (256, 256)) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size

        self.image_files = sorted(path for path in self.images_dir.iterdir() if path.is_file())
        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        missing_masks = [path.name for path in self.image_files if not (self.masks_dir / path.name).exists()]
        if missing_masks:
            raise ValueError(f"Missing mask files for: {', '.join(missing_masks[:5])}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _to_tensor(self, image: Image.Image, is_mask: bool) -> torch.Tensor:
        if is_mask:
            array = np.array(image.resize(self.image_size, Image.Resampling.NEAREST), dtype=np.float32)
            array = (array > 127).astype(np.float32)
        else:
            array = np.array(image.resize(self.image_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        if array.ndim == 2:
            array = array[None, ...]
        else:
            array = array.transpose(2, 0, 1)

        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[index]
        mask_path = self.masks_dir / image_path.name

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        return self._to_tensor(image, is_mask=False), self._to_tensor(mask, is_mask=True)
