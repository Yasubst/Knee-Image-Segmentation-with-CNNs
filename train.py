from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from knee_segmentation.data import KneeSegmentationDataset
from knee_segmentation.model import UNet


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = KneeSegmentationDataset(args.images_dir, args.masks_dir, (args.image_size, args.image_size))

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_size else None

    model = UNet(in_channels=1, out_channels=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / max(len(train_loader), 1)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    val_loss += (bce(logits, masks) + dice_loss(logits, masks)).item()
            avg_val = val_loss / max(len(val_loader), 1)
            print(f"epoch={epoch} train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
        else:
            print(f"epoch={epoch} train_loss={avg_train:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved checkpoint to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net for knee image segmentation")
    parser.add_argument("--images-dir", required=True, help="Directory containing input knee images")
    parser.add_argument("--masks-dir", required=True, help="Directory containing binary mask images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", default="checkpoints/unet_knee.pt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
