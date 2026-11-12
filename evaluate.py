import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from model import UNet
from dataset import KneeDataset

os.makedirs("demo_images", exist_ok=True)
# Load validation data
val_df = pd.read_csv("data/CSVs/val.csv")
val_dataset = KneeDataset(val_df)

# Load the best trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)
model.load_state_dict(
    torch.load(
        "sessions/improved_model.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

from torch.utils.data import DataLoader

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]

for threshold in thresholds:
    total_dice = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in val_loader:
            xrays = batch["xray"].to(device)
            masks = batch["mask"].to(device)

            probs = torch.sigmoid(model(xrays))
            preds = (probs > threshold).float()

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            dice = (2 * intersection + 1e-6) / (union + 1e-6)

            total_dice += dice.sum().item()
            total_images += xrays.size(0)

    print(f"Threshold {threshold:.2f} | Validation Dice: {total_dice / total_images:.4f}")

# Select one validation image
sample = val_dataset[0]

xray = torch.from_numpy(sample["xray"]).unsqueeze(0).to(device)
actual_mask = sample["mask"][0]

# Predict segmentation mask
with torch.no_grad():
    outputs = model(xray)
    probs = torch.sigmoid(outputs)
    predicted_mask = (probs > 0.40).float()

# Convert tensors to images
xray_image = sample["xray"][0]
predicted_mask = predicted_mask[0, 0].cpu().numpy()

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(xray_image, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(actual_mask, cmap="gray")
plt.title("Actual Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig("demo_images/validation_prediction.png", dpi=200, bbox_inches="tight")
plt.show()

# Evaluate on unlabeled test images
from dataset import read_xray

test_df = pd.read_csv("data/CSVs/test.csv")

plt.figure(figsize=(10, 8))

with torch.no_grad():
    for i in range(min(3, len(test_df))):
        image = read_xray(test_df.iloc[i]["xrays"])
        xray = torch.from_numpy(image).unsqueeze(0).to(device)

        probs = torch.sigmoid(model(xray))
        predicted = (probs > 0.40).float()[0, 0].cpu().numpy()

        plt.subplot(3, 2, 2 * i + 1)
        plt.imshow(image[0], cmap="gray")
        plt.title("Test X-ray")
        plt.axis("off")

        plt.subplot(3, 2, 2 * i + 2)
        plt.imshow(predicted, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

plt.tight_layout()
plt.savefig("demo_images/test_predictions.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"Test set contains {len(test_df)} unlabeled images.")
print("Displayed predictions for 3 test images.")