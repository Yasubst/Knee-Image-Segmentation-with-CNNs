
import os
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. Define dataset directories
data_list = []

xrays_dir = "data/xrays"
masks_dir = "data/masks"
csv_dir = "data/CSVs"

os.makedirs(csv_dir, exist_ok=True)

if not os.path.isdir(xrays_dir):
    raise FileNotFoundError(f"Missing folder: {xrays_dir}")

if not os.path.isdir(masks_dir):
    raise FileNotFoundError(f"Missing folder: {masks_dir}")


# 2. Match X-ray images with masks
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

for xray in sorted(os.listdir(xrays_dir)):

    if not xray.lower().endswith(valid_extensions):
        continue

    xray_path = os.path.join(xrays_dir, xray)
    mask_path = os.path.join(masks_dir, xray)

    # Check whether the corresponding mask exists
    if not os.path.isfile(mask_path):
        mask_path = None

    else:
        # Check X-ray and mask dimensions
        image = cv2.imread(
            xray_path,
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            raise ValueError(
                f"Cannot read image or mask: {xray}"
            )

        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {xray}: "
                f"X-ray {image.shape}, "
                f"Mask {mask.shape}"
            )

    data_list.append((xray_path, mask_path))


# 3. Create complete dataset CSV
data_df = pd.DataFrame(
    data_list,
    columns=["xrays", "masks"]
)

data_df.to_csv(
    os.path.join(csv_dir, "dataset.csv"),
    index=False
)

print("Total X-rays:", len(data_df))
print(
    "Images with masks:",
    data_df["masks"].notna().sum()
)
print(
    "Images without masks:",
    data_df["masks"].isna().sum()
)


# 4. Separate labeled and unlabeled images
labeled_df = data_df[
    data_df["masks"].notna()
].copy()

test_df = data_df[
    data_df["masks"].isna()
].copy()


# 5. Validate dataset size
if len(labeled_df) != 100:
    raise ValueError(
        f"Expected 100 labeled images, "
        f"found {len(labeled_df)}."
    )

if len(test_df) != 50:
    raise ValueError(
        f"Expected 50 unlabeled test images, "
        f"found {len(test_df)}."
    )


# 6. Split labeled images
train_df, val_df = train_test_split(
    labeled_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# 7. Save CSV files
train_df.to_csv(
    os.path.join(csv_dir, "train.csv"),
    index=False
)

val_df.to_csv(
    os.path.join(csv_dir, "val.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(csv_dir, "test.csv"),
    index=False
)


print("\nDataset split completed!")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))


# 8. Plot dataset distribution
csv_names = [
    "Training",
    "Validation",
    "Testing"
]

csv_counts = [
    len(train_df),
    len(val_df),
    len(test_df)
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    csv_names,
    csv_counts,
    color=["steelblue", "orange", "green"]
)

plt.title("Knee X-ray Dataset Distribution")
plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")

for bar, count in zip(bars, csv_counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        str(count),
        ha="center"
    )

plt.ylim(0, max(csv_counts) + 15)
plt.tight_layout()

plt.savefig(
    os.path.join(
        csv_dir,
        "csv_distribution.png"
    ),
    dpi=300
)

plt.show()
