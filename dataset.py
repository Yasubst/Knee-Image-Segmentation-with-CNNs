from torch.utils import Dataset
import cv2

import numpy as np



def read_xray(dir):
    xray = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    xray = xray.astype(np.float32) / 255.0
    xray = xray.reshape(1, *xray.shape)  # defining the dimension 1 H W

    return xray

def read_mask(dir):
    mask = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.float32)
    mask = mask.reshape(1, *mask.shape)
    return mask

class KneeDataset(Dataset):
    def __init__(self, df):
        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        xray_path = self.df.iloc[index]["xrays"]
        mask_path = self.df.iloc[index]["masks"]

        xray = read_xray(xray_path)
        mask = read_mask(mask_path)

        return {
            "xray": xray,
            "mask": mask
        }






