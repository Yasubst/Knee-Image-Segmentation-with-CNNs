
from args import get_args
import pandas as pd
import os
import torch

from dataset import KneeDataset
from torch.utils.data import DataLoader
from model import UNet
from trainer import train_model


# 1: Get arguments
args = get_args()

# 2: Load CSV files
train_set = pd.read_csv(
    os.path.join(args.csv_dir, "train.csv")
)

val_set = pd.read_csv(
    os.path.join(args.csv_dir, "val.csv")
)

# 3: Create datasets

# Enable augmentation for training only
train_dataset = KneeDataset(
    train_set,
    augment=True
)

# No augmentation for validation
val_dataset = KneeDataset(
    val_set,
    augment=False
)

# 4: Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=int(args.bs),
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=int(args.bs),
    shuffle=False,
    num_workers=0
)

# 5: Initialize U-Net model
model = UNet()

# 6: Define device (CPU or GPU)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

print(
    "Total model parameters:",
    sum(p.numel() for p in model.parameters())
)

# 7: Start training
train_model(
    model,
    train_loader,
    val_loader,
    device
)
