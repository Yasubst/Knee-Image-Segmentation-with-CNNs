
from torch.utils.data import Dataset
import cv2
import numpy as np


def read_xray(dir):
    xray = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

    if xray is None:
        raise FileNotFoundError(f"X-ray not found: {dir}")

    # Normalize X-ray to [0, 1]
    xray = xray.astype(np.float32) / 255.0

    # Shape: (1, H, W)
    xray = xray.reshape(1, *xray.shape)

    return xray


def read_mask(dir):
    mask = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Mask not found: {dir}")

    # White tibia = 1, black background = 0
    mask = (mask > 127).astype(np.float32)

    # Shape: (1, H, W)
    mask = mask.reshape(1, *mask.shape)

    return mask


class KneeDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray_path = self.df.iloc[index]["xrays"]
        mask_path = self.df.iloc[index]["masks"]

        xray = read_xray(xray_path)
        mask = read_mask(mask_path)

        # Apply augmentation to training images only
        if self.augment:
            # Horizontal flip
            if np.random.rand() < 0.5:
                xray = np.flip(xray, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()

            # Small rotation, same for image and mask
            if np.random.rand() < 0.5:
                h, w = xray.shape[1:]

                angle = np.random.uniform(-10, 10)

                matrix = cv2.getRotationMatrix2D(
                    (w / 2, h / 2), angle, 1.0
                )

                xray_rotated = cv2.warpAffine(
                    xray[0],
                    matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                mask_rotated = cv2.warpAffine(
                    mask[0],
                    matrix,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                xray = xray_rotated[np.newaxis, ...]
                mask = mask_rotated[np.newaxis, ...]

        return {
            "xray": np.ascontiguousarray(xray, dtype=np.float32),
            "mask": np.ascontiguousarray(mask, dtype=np.float32)
        }
