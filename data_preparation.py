import os
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_list = []
xrays_dir = "data/xrays"

for xray in sorted(os.listdir(xrays_dir)):
    xray_path = os.path.join(xrays_dir, xray)
    mask_path = os.path.join("data/masks", xray)

    if not os.path.exists(mask_path):
        mask_path = None

    data_list.append((xray_path, mask_path))

data_df = pd.DataFrame(data_list, columns=["xrays", "masks"])
data_df.to_csv("data/CSVs/dataset.csv", index=False)

print("Total X-rays:", len(data_df))
print("Images with masks:", data_df["masks"].notna().sum())
print("Images without masks:", data_df["masks"].isna().sum())

labeled_df = data_df[data_df["masks"].notna()]
test_df = data_df[data_df["masks"].isna()]

train_df, val_df = train_test_split(
    labeled_df,
    test_size=0.2,
    random_state=42
)

train_df.to_csv("data/CSVs/train.csv", index=False)
val_df.to_csv("data/CSVs/val.csv", index=False)
test_df.to_csv("data/CSVs/test.csv", index=False)

print("training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

csv_names = ["Dataset", "Training", "Validation", "Test"]
csv_counts = [len(data_df), len(train_df), len(val_df), len(test_df)]

plt.bar(csv_names, csv_counts)
plt.title("Dataset Splits Distribution")
plt.xlabel("CSV Files")
plt.ylabel("Number of Images")

for i, count in enumerate(csv_counts):
    plt.text(i, count + 2, str(count), ha="center")

plt.tight_layout()
plt.savefig("data/CSVs/csv_distribution.png", dpi=300)

plt.show()
