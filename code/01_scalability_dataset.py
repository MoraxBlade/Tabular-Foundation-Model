import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import StratifiedShuffleSplit

# ===================== 全局配置 =====================
try:
    from utils import SEED
except ImportError:
    SEED = 42
np.random.seed(SEED)

PROCESSED_DATA_DIR = "data/processed"
SCALABILITY_DATA_DIR = "data/scalability"
os.makedirs(SCALABILITY_DATA_DIR, exist_ok=True)

# ===================== 辅助函数 =====================
def generate_log_spaced_sizes(total_train_samples, min_size=1000, max_size=None, num=5):
    """
    生成对数间隔的样本量列表（不超过 total_train_samples）
    min_size: 最小样本量（默认1000）
    max_size: 最大样本量（默认为 total_train_samples）
    num: 生成几个样本量阶梯
    """
    if max_size is None:
        max_size = total_train_samples
    # 限制最大不超过总样本量
    max_size = min(max_size, total_train_samples)
    # 生成对数间隔，并取整
    sizes = np.logspace(np.log10(min_size), np.log10(max_size), num=num, dtype=int)
    # 去重并确保从小到大排序
    sizes = np.unique(sizes)
    # 过滤掉超出总样本量的值（理论上已处理，但安全起见）
    sizes = [s for s in sizes if s <= total_train_samples]
    return sizes

def stratified_sample(df, n_samples, random_state):
    """分层采样（按最后一列的标签），支持 n_samples == len(df) 的情况"""
    if n_samples == len(df):
        return df.copy()  # 全量时直接返回副本，无需分割
    y = df.iloc[:, -1]
    # 使用 StratifiedShuffleSplit 保证采样后各类比例与原始一致
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    for train_idx, _ in sss.split(df, y):
        sampled_df = df.iloc[train_idx].reset_index(drop=True)
        return sampled_df
    # 理论上不会到这里，但作为后备
    return df.sample(n=n_samples, random_state=random_state)

# ===================== 主函数 =====================
def generate_scalability_datasets(
    target_sizes=None,          # 手动指定样本量列表，如 [1000, 5000, 10000]
    repeat=1,                   # 重复次数（生成多份数据，后缀 _r0, _r1...）
    auto_sizes=True,            # 是否自动生成对数间隔样本量（若 target_sizes 为 None 则启用）
    min_size=1000,              # 自动生成时的最小样本量
    num_sizes=5,                # 自动生成时的阶梯数量
    overwrite=False,            # 是否覆盖已存在的文件
    seed=SEED                   # 基础随机种子（每次重复会偏移）
):
    print("=" * 60)
    print("生成可扩展性实验数据集")
    print(f"基础随机种子: {seed}, 重复次数: {repeat}")
    print("=" * 60)

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

        # 确定样本量列表
        if target_sizes is not None:
            sizes = [s for s in target_sizes if s <= total_samples]
            if not sizes:
                print(f"  警告: 指定的样本量 {target_sizes} 均大于总样本量 {total_samples}，跳过")
                continue
        elif auto_sizes:
            sizes = generate_log_spaced_sizes(total_samples, min_size=min_size, num=num_sizes)
            print(f"  自动生成样本量: {sizes}")
        else:
            # 如果既未指定 target_sizes 也未开启 auto_sizes，则默认只生成全量
            sizes = [total_samples]
            print(f"  使用全量样本: {sizes}")

        # 生成每个样本量的数据集（支持重复）
        for size in sizes:
            for rep in range(repeat):
                # 每次重复使用不同种子，确保采样独立
                current_seed = seed + rep
                np.random.seed(current_seed)

                # 确定文件名
                if repeat == 1:
                    save_filename = f"{dataset_name}_train_{size}samples.csv"
                else:
                    save_filename = f"{dataset_name}_train_{size}samples_r{rep}.csv"
                save_path = os.path.join(SCALABILITY_DATA_DIR, save_filename)

                if os.path.exists(save_path) and not overwrite:
                    print(f"  ⏭ 跳过已存在: {save_filename}")
                    continue

                # 分层采样（已处理 size == total_samples 的情况）
                sampled_df = stratified_sample(train_df, size, random_state=current_seed)
                sampled_df.to_csv(save_path, index=False)
                print(f"  ✓ 生成 {size:,} 样本量 (重复 {rep}): {save_filename}")

    print("\n" + "=" * 60)
    print("所有可扩展性数据集生成完成！")
    print(f"保存目录: {SCALABILITY_DATA_DIR}")
    print("=" * 60)

# ===================== 命令行入口 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成可扩展性实验数据集")
    parser.add_argument("--sizes", nargs="+", type=int, help="手动指定样本量列表，如 1000 5000 10000")
    parser.add_argument("--repeat", type=int, default=1, help="重复采样次数（生成多份数据）")
    parser.add_argument("--auto", action="store_true", default=True, help="自动生成对数间隔样本量（默认启用）")
    parser.add_argument("--min-size", type=int, default=1000, help="自动生成时的最小样本量")
    parser.add_argument("--num-sizes", type=int, default=5, help="自动生成时的阶梯数量")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--seed", type=int, default=SEED, help="基础随机种子")
    args = parser.parse_args()

    generate_scalability_datasets(
        target_sizes=args.sizes,
        repeat=args.repeat,
        auto_sizes=args.auto,
        min_size=args.min_size,
        num_sizes=args.num_sizes,
        overwrite=args.overwrite,
        seed=args.seed
    )