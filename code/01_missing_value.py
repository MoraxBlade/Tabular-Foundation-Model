# code/06_missing_value.py
import pandas as pd
import numpy as np
import os
from utils import SEED

np.random.seed(SEED)
PROCESSED_DATA_DIR = "data/processed"
MISSING_DATA_DIR = "data/missing"
MISSING_RATIOS = [0, 0.1, 0.2, 0.3]
os.makedirs(MISSING_DATA_DIR, exist_ok=True)

def add_missing_values(df, missing_ratio):
    df_missing = df.copy()
    feature_cols = df_missing.columns[:-1]
    mask = np.random.rand(*df_missing[feature_cols].shape) < missing_ratio
    df_missing[feature_cols] = df_missing[feature_cols].mask(mask)
    return df_missing

def generate_missing_datasets():
    for train_file in os.listdir(PROCESSED_DATA_DIR):
        if not train_file.endswith("_train.csv"):
            continue
        dataset_name = train_file.replace("_train.csv", "")
        print("处理数据集：", dataset_name)
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, train_file))
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}_test.csv"))
        for ratio in MISSING_RATIOS:
            ratio_str = f"{int(ratio*100)}missing"
            train_miss = add_missing_values(train_df, ratio)
            test_miss = add_missing_values(test_df, ratio)
            train_save = os.path.join(MISSING_DATA_DIR, f"{dataset_name}_train_{ratio_str}.csv")
            test_save = os.path.join(MISSING_DATA_DIR, f"{dataset_name}_test_{ratio_str}.csv")
            train_miss.to_csv(train_save, index=False)
            test_miss.to_csv(test_save, index=False)
            print("已生成", ratio*100, "%缺失：", train_save)

if __name__ == "__main__":
    generate_missing_datasets()