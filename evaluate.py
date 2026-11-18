
import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from args import get_args
from dataset import KneeDataset
from model import UNet
from utils import dice_score_from_logits
from torch.utils.data import DataLoader



# 1. Configuration


args = get_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

THRESHOLD = 0.5

os.makedirs(args.out_dir, exist_ok=True)

comparison_dir = os.path.join(
    args.out_dir,
    "validation_comparisons_cleaned"
)

raw_prediction_dir = os.path.join(
    args.out_dir,
    "test_predictions_raw"
)

clean_prediction_dir = os.path.join(
    args.out_dir,
    "test_predictions_cleaned"
)

os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(raw_prediction_dir, exist_ok=True)
os.makedirs(clean_prediction_dir, exist_ok=True)



# 2. Post-processing function

def postprocess_mask(mask):
    """
    Clean a binary segmentation mask.

    Steps:
    1. Keep largest connected foreground component.
    2. Fill enclosed holes.
    3. Apply mild morphological closing.

    Input:
        mask: uint8 array containing 0 and 255.

    Output:
        cleaned uint8 mask containing 0 and 255.
    """

    binary = (mask > 0).astype(np.uint8)

    # Identify connected white regions
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    # Handle empty prediction
    if num_labels <= 1:
        return np.zeros_like(
            mask,
            dtype=np.uint8
        )

    # Ignore background label 0
    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    # Keep largest connected component
    largest_component = (
        labels == largest_label
    ).astype(np.uint8)

    # Fill holes using external contour
    contours, _ = cv2.findContours(
        largest_component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(
        largest_component
    )

    cv2.drawContours(
        filled_mask,
        contours,
        -1,
        color=1,
        thickness=cv2.FILLED
    )

    # Mild morphological closing
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        filled_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return (
        cleaned.astype(np.uint8) * 255
    )



# 3. Dice calculation for binary masks

def calculate_binary_dice(pred_mask, true_mask, eps=1e-6):

    pred = (pred_mask > 0).astype(np.float32)
    true = (true_mask > 0).astype(np.float32)

    intersection = np.sum(pred * true)

    denominator = (
        np.sum(pred) + np.sum(true)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return float(dice)



# 4. Load best trained U-Net model

model = UNet().to(device)

model_path = os.path.join(
    args.out_dir,
    "demo_model.pth"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model checkpoint not found: {model_path}"
    )

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Best model loaded successfully!")



# 5. Load validation dataset

val_df = pd.read_csv(
    os.path.join(
        args.csv_dir,
        "val.csv"
    )
)

val_dataset = KneeDataset(
    val_df,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# 6. Evaluate raw and cleaned validation predictions

validation_results = []

print("\nStarting validation evaluation...")

with torch.no_grad():

    for i, batch in enumerate(val_loader):

        xray = batch["xray"].to(device)
        mask = batch["mask"].to(device)

        logits = model(xray)

        # Raw model prediction
        probabilities = torch.sigmoid(logits)

        raw_prediction = (
            probabilities >= THRESHOLD
        ).float()

        # Original Dice using existing utility
        raw_dice = dice_score_from_logits(
            logits,
            mask,
            threshold=THRESHOLD
        )

        # Convert tensors to NumPy
        xray_img = (
            xray[0, 0].cpu().numpy()
        )

        true_mask = (
            mask[0, 0].cpu().numpy()
        )

        raw_mask = (
            raw_prediction[0, 0]
            .cpu()
            .numpy()
            .astype(np.uint8) * 255
        )

        # Apply post-processing
        cleaned_mask = postprocess_mask(
            raw_mask
        )

        # Calculate cleaned Dice
        cleaned_dice = calculate_binary_dice(
            cleaned_mask,
            true_mask
        )

        improvement = cleaned_dice - raw_dice

        filename = os.path.basename(
            val_df.iloc[i]["xrays"]
        )

        validation_results.append({
            "image": filename,
            "raw_dice": raw_dice,
            "cleaned_dice": cleaned_dice,
            "dice_change": improvement
        })

        # Create comparison figure
        fig, axes = plt.subplots(
            1, 4,
            figsize=(16, 4)
        )

        axes[0].imshow(
            xray_img,
            cmap="gray"
        )
        axes[0].set_title("Original Knee X-ray")

        axes[1].imshow(
            true_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[1].set_title("Ground Truth Tibia")

        axes[2].imshow(
            raw_mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[2].set_title(
            f"Raw Prediction\nDice: {raw_dice:.4f}"
        )

        axes[3].imshow(
            cleaned_mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[3].set_title(
            f"Cleaned Prediction\nDice: {cleaned_dice:.4f}"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                comparison_dir,
                f"validation_{i + 1:03d}.png"
            ),
            dpi=200
        )

        plt.close(fig)

        print(
            f"Image {i + 1:02d}/{len(val_dataset)} | "
            f"Raw Dice: {raw_dice:.4f} | "
            f"Cleaned Dice: {cleaned_dice:.4f} | "
            f"Change: {improvement:+.4f}"
        )



# 7. Save validation metrics

results_df = pd.DataFrame(
    validation_results
)

results_path = os.path.join(
    args.out_dir,
    "postprocessing_dice_scores.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

mean_raw_dice = results_df["raw_dice"].mean()

mean_cleaned_dice = (
    results_df["cleaned_dice"].mean()
)

mean_improvement = (
    mean_cleaned_dice - mean_raw_dice
)

improved_images = (
    results_df["dice_change"] > 0
).sum()

worsened_images = (
    results_df["dice_change"] < 0
).sum()

print("\n====================================")
print("POST-PROCESSING EVALUATION RESULTS")
print("====================================")

print(
    f"Mean Raw Dice:     {mean_raw_dice:.4f}"
)

print(
    f"Mean Cleaned Dice: {mean_cleaned_dice:.4f}"
)

print(
    f"Mean Dice Change:  {mean_improvement:+.4f}"
)

print(
    f"Improved images:   {improved_images}"
)

print(
    f"Worsened images:   {worsened_images}"
)


# 8. Plot raw vs cleaned Dice scores


image_numbers = np.arange(
    1,
    len(results_df) + 1
)

plt.figure(figsize=(13, 6))

plt.plot(
    image_numbers,
    results_df["raw_dice"],
    marker="o",
    label="Raw Dice",
    color="royalblue"
)

plt.plot(
    image_numbers,
    results_df["cleaned_dice"],
    marker="s",
    label="Cleaned Dice",
    color="forestgreen"
)

plt.axhline(
    mean_raw_dice,
    color="royalblue",
    linestyle="--",
    alpha=0.5,
    label=f"Mean Raw: {mean_raw_dice:.4f}"
)

plt.axhline(
    mean_cleaned_dice,
    color="forestgreen",
    linestyle="--",
    alpha=0.5,
    label=f"Mean Cleaned: {mean_cleaned_dice:.4f}"
)

plt.title(
    "Validation Dice: Raw vs Post-processed Masks"
)

plt.xlabel("Validation Image")
plt.ylabel("Dice Score")
plt.ylim(0, 1)
plt.xticks(image_numbers)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        args.out_dir,
        "postprocessing_dice_comparison.png"
    ),
    dpi=200
)

plt.close()



# 9. Load test images

test_df = pd.read_csv(
    os.path.join(
        args.csv_dir,
        "test.csv"
    )
)

print("\nStarting test predictions...")


# 10. Generate raw and cleaned test masks

with torch.no_grad():

    for i, row in test_df.iterrows():

        image_path = row["xrays"]

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Cannot read X-ray: {image_path}"
            )

        original_height, original_width = image.shape

        # Use the same resolution as training
        resized = cv2.resize(
            image,
            (256, 256),
            interpolation=cv2.INTER_AREA
        )

        normalized = (
            resized.astype(np.float32) / 255.0
        )

        input_tensor = torch.from_numpy(
            normalized
        ).unsqueeze(0).unsqueeze(0).to(device)

        logits = model(input_tensor)

        probabilities = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

        # Raw binary prediction
        raw_mask = (
            probabilities >= THRESHOLD
        ).astype(np.uint8) * 255

        # Clean prediction at model resolution
        cleaned_mask = postprocess_mask(
            raw_mask
        )

        # Restore original image dimensions
        raw_mask = cv2.resize(
            raw_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        cleaned_mask = cv2.resize(
            cleaned_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        filename = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        # Save raw mask
        cv2.imwrite(
            os.path.join(
                raw_prediction_dir,
                f"{filename}_raw.png"
            ),
            raw_mask
        )

        # Save cleaned mask
        cv2.imwrite(
            os.path.join(
                clean_prediction_dir,
                f"{filename}_cleaned.png"
            ),
            cleaned_mask
        )

        print(
            f"Test image {i + 1}/{len(test_df)} "
            f"completed: {filename}"
        )


print("\nAll evaluation tasks completed!")
