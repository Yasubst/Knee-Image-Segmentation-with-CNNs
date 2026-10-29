from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from knee_segmentation.model import UNet


def load_image(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, None, ...])


def save_mask(mask_tensor: torch.Tensor, output_path: str) -> None:
    mask = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def infer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image = load_image(args.input, args.image_size).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    save_mask(probs, args.output)
    print(f"Saved predicted mask to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate knee segmentation mask using a trained U-Net")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input knee image path")
    parser.add_argument("--output", required=True, help="Output mask image path")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
