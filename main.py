from args import get_args
import pandas as pd
import os

# 1: we need arguments
args = get_args()

# 2: loading the csv files
train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv')) # we need to have this csv file in the csv directory
val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

