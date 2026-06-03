# code/09_scalability_dataset.py
import pandas as pd
import numpy as np
import os
from utils import SEED

np.random.seed(SEED)

PROCESSED_DATA_DIR = "data/processed"
SCALABILITY_DATA_DIR = "data/scalability"
os.makedirs(SCALABILITY_DATA_DIR, exist_ok=True)

# 目标样本量阶梯（单位：条）
TARGET_SIZES = [1000, 10000, 100000]

def generate_scalability_datasets():
    for train_file in os.listdir(PROCESSED_DATA_DIR):
        if not train_file.endswith("_train.csv"):
            continue
        dataset_name = train_file.replace("_train.csv", "")
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, train_file))
        total = len(train_df)
        print(f"\n=== {dataset_name}  总样本数: {total} ===")

        # 动态选择可用的样本量（不超过总样本数，且去重）
        valid_sizes = sorted(set([s for s in TARGET_SIZES if s <= total]))
        if not valid_sizes:
            # 如果连最小的1000都抽不了，就使用总样本数本身（例如credit-g的700）
            valid_sizes = [total]
            print(f"目标样本量均超过总样本数，将使用全量 {total} 作为样本量")

        for size in valid_sizes:
            sample_df = train_df.sample(n=size, random_state=SEED, replace=False)
            save_path = os.path.join(SCALABILITY_DATA_DIR, f"{dataset_name}_train_{size}samples.csv")
            sample_df.to_csv(save_path, index=False)
            print(f"生成 {size} 样本量: {save_path}")

if __name__ == "__main__":
    generate_scalability_datasets()
    print(f"\n所有可扩展性样本集生成完成，保存至：{SCALABILITY_DATA_DIR}")