
import argparse


def get_args():

    parser = argparse.ArgumentParser(
        description="Knee_Segmentation project"
    )

    # Dataset CSV directory
    parser.add_argument(
        "-csv_dir",
        default="data/CSVs"
    )

    # Model output directory
    parser.add_argument(
        "-out_dir",
        default="sessions"
    )

    # Number of training epochs
    parser.add_argument(
        "-epochs",
        type=int,
        default=100
    )

    # Batch size
    parser.add_argument(
        "-bs",
        type=int,
        default=4
    )

    # Learning rate
    parser.add_argument(
        "-lr",
        type=float,
        default=1e-4,
        choices=[1e-2, 1e-3, 1e-4, 1e-5]
    )

    # Weight decay
    parser.add_argument(
        "-wd",
        type=float,
        default=1e-5,
        choices=[1e-5, 1e-7]
    )

    args = parser.parse_args()

    return args


# Test arguments
if __name__ == "__main__":

    args = get_args()

    print("CSV directory:", args.csv_dir)
    print("Output directory:", args.out_dir)
    print("Epochs:", args.epochs)
    print("Batch size:", args.bs)
    print("Learning rate:", args.lr)
    print("Weight decay:", args.wd)
