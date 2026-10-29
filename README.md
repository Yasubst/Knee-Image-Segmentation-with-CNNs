# Knee-Image-Segmentation-with-CNNs

Minimal PyTorch implementation for knee image segmentation with a U-Net CNN.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Use matching file names in image and mask folders:

```text
data/
  images/
    sample_001.png
  masks/
    sample_001.png
```

## Train

```bash
python train.py \
  --images-dir data/images \
  --masks-dir data/masks \
  --epochs 20 \
  --batch-size 8 \
  --output checkpoints/unet_knee.pt
```

## Predict

```bash
python predict.py \
  --checkpoint checkpoints/unet_knee.pt \
  --input data/images/sample_001.png \
  --output outputs/sample_001_mask.png
```
