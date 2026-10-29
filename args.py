import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Knee_Segmentation project")

    parser.add_argument('-csv_dir', default='data/CSVs')

    parser.add_argument('-out_dir', default='sessions')

    parser.add_argument('-epochs', default=10)

    parser.add_argument('-bs', default=16) # bs = batch size

    parser.add_argument('-lr', default=1e-3,
                        choices=[1e-2, 1e-3, 1e-4, 1e-5])

    parser.add_argument('-wd', default=1e-5,
                        choices=[1e-5, 1e-7])

    args = parser.parse_args()
    return args