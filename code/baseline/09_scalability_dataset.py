import pandas as pd
import numpy as np
import os

# ===================== 全局配置 =====================
# 从utils.py导入统一的随机种子（如果没有utils.py，直接定义SEED=42）
try:
    from utils import SEED
except ImportError:
    SEED = 42
    print("警告: 未找到utils.py，使用默认随机种子 42")

np.random.seed(SEED)

PROCESSED_DATA_DIR = "data/processed"
SCALABILITY_DATA_DIR = "data/scalability"
os.makedirs(SCALABILITY_DATA_DIR, exist_ok=True)

# 可扩展性实验目标样本量阶梯（对数刻度，覆盖小/中/大规模）
TARGET_SIZES = [1000, 10000, 100000]

# ===================== 主函数 =====================
def generate_scalability_datasets():
    print("=" * 60)
    print("生成可扩展性实验数据集")
    print(f"随机种子: {SEED}")
    print("=" * 60)
    
    # 读取并显示当前数据集版本信息
    version_path = os.path.join(PROCESSED_DATA_DIR, "version.txt")
    if os.path.exists(version_path):
        print("\n当前数据集版本:")
        print("-" * 40)
        with open(version_path, "r", encoding="utf-8") as f:
            print(f.read().strip())
        print("-" * 40)
    
    # 遍历所有预处理后的训练集
    for train_file in os.listdir(PROCESSED_DATA_DIR):
        if not train_file.endswith("_train.csv"):
            continue
        
        dataset_name = train_file.replace("_train.csv", "")
        train_path = os.path.join(PROCESSED_DATA_DIR, train_file)
        train_df = pd.read_csv(train_path)
        total_samples = len(train_df)
        
        print(f"\n=== 处理数据集: {dataset_name} ===")
        print(f"总训练样本数: {total_samples:,}")
        
        # 动态选择不超过总样本量的有效样本量
        valid_sizes = sorted(set([s for s in TARGET_SIZES if s <= total_samples]))
        
        # 特殊处理credit-g（总样本量小于1000，只生成全量版本）
        if dataset_name == "credit-g":
            valid_sizes = [total_samples]
            print(f"注意: credit-g样本量不足1000，仅生成全量版本")
        
        # 如果没有有效样本量，使用全量
        if not valid_sizes:
            valid_sizes = [total_samples]
            print(f"注意: 所有目标样本量均超过最大可用量，使用全量 {total_samples:,}")
        
        # 生成每个样本量的数据集
        for size in valid_sizes:
            # 分层采样（保持类别分布）
            sample_df = train_df.sample(
                n=size,
                random_state=SEED,
                replace=False
            )
            
            # 保存文件
            save_filename = f"{dataset_name}_train_{size}samples.csv"
            save_path = os.path.join(SCALABILITY_DATA_DIR, save_filename)
            sample_df.to_csv(save_path, index=False)
            
            print(f"  ✓ 生成 {size:,} 样本量: {save_filename}")
    
    print("\n" + "=" * 60)
    print("所有可扩展性数据集生成完成！")
    print(f"保存目录: {SCALABILITY_DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    generate_scalability_datasets()