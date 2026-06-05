import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ===================== 全局配置（绝对不要修改） =====================
SEED = 42
np.random.seed(SEED)
TEST_SIZE = 0.3

# 标准数据集原始行数（仅用于校验原始数据完整性）
EXPECTED_ROWS = {
    "adult": 32561,
    "credit-g": 1000,
    "covtype": 581012
}

# ===================== 降采样配置 =====================
DOWNSAMPLE_ADULT = True          # 是否对 adult 进行降采样
TARGET_TOTAL_ADULT = 10000       # 降采样后的总样本量（划分前）

DOWNSAMPLE_COVTYPE = True        # 是否对 covtype 进行降采样
TARGET_TOTAL_COVTYPE = 50000     # 降采样后的总样本量（划分前）

# ===================== 数据集预处理函数 =====================
def preprocess_adult():
    """处理 UCI 经典 adult 收入预测数据集，支持降采样到 TARGET_TOTAL_ADULT 行"""
    df = pd.read_csv(
        "data/raw/adult.csv",
        na_values=["?"],
        skipinitialspace=True
    )
    original_len = len(df)
    if original_len != EXPECTED_ROWS["adult"]:
        print(f"警告：adult 数据集行数异常！预期 {EXPECTED_ROWS['adult']} 行，实际 {original_len} 行")

    print(f"adult 数据集加载成功，原始数据行数：{original_len:,}")
    
    # 标签转换
    df.iloc[:, -1] = df.iloc[:, -1].map({"<=50K": 0, ">50K": 1}).astype(int)

    # 分类特征编码
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"分类特征数量：{len(cat_cols)}，数值特征数量：{df.shape[1]-1-len(cat_cols)}")
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    # 填充缺失值
    missing_before = df.isnull().sum().sum()
    df = df.fillna(-1)
    missing_after = df.isnull().sum().sum()
    print(f"缺失值处理：填充前 {missing_before} 个，填充后 {missing_after} 个")

    # ----- Adult 降采样逻辑 -----
    if DOWNSAMPLE_ADULT and original_len > TARGET_TOTAL_ADULT:
        print(f"\n adult 原始数据 {original_len:,} 行，需降采样至 {TARGET_TOTAL_ADULT:,} 行（分层采样）")
        y = df.iloc[:, -1]
        # 分层采样得到 TARGET_TOTAL_ADULT 行总样本
        downsampled_df, _ = train_test_split(
            df,
            train_size=TARGET_TOTAL_ADULT,
            random_state=SEED,
            stratify=y
        )
        df = downsampled_df.reset_index(drop=True)
        print(f"   降采样完成，总样本数：{len(df):,}")
    elif DOWNSAMPLE_ADULT and original_len <= TARGET_TOTAL_ADULT:
        print(f"\n  adult 原始数据 {original_len:,} 行不足 {TARGET_TOTAL_ADULT:,}，保留全部数据")
    else:
        print(f"\nadult 使用全部原始数据，行数：{original_len:,}")

    return df

def preprocess_credit_g():
    """处理德国信用数据集"""
    df = pd.read_csv("data/raw/credit-g.csv")
    original_len = len(df)
    if original_len != EXPECTED_ROWS["credit-g"]:
        print(f"警告：credit-g 数据集行数异常！预期 {EXPECTED_ROWS['credit-g']} 行，实际 {original_len} 行")
    # 标签转换
    df.iloc[:, -1] = df.iloc[:, -1].map({"bad": 0, "good": 1}).astype(int)
    # 分类特征编码
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    df = df.fillna(-1)
    return df

def preprocess_covtype():
    """处理森林覆盖类型数据集，支持降采样到 TARGET_TOTAL_COVTYPE 行"""
    df = pd.read_csv("data/raw/covtype_raw.csv")
    original_len = len(df)
    if original_len != EXPECTED_ROWS["covtype"]:
        print(f"警告：covtype 数据集行数异常！预期 {EXPECTED_ROWS['covtype']} 行，实际 {original_len} 行")

    # 标签转换：原始标签 1~7 → 0~6
    df.iloc[:, -1] = df.iloc[:, -1] - 1
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)

    # ----- Covtype 降采样逻辑 -----
    if DOWNSAMPLE_COVTYPE and original_len > TARGET_TOTAL_COVTYPE:
        print(f"\n covtype 原始数据 {original_len:,} 行，需降采样至 {TARGET_TOTAL_COVTYPE:,} 行（分层采样）")
        y = df.iloc[:, -1]
        downsampled_df, _ = train_test_split(
            df,
            train_size=TARGET_TOTAL_COVTYPE,
            random_state=SEED,
            stratify=y
        )
        df = downsampled_df.reset_index(drop=True)
        print(f"   降采样完成，总样本数：{len(df):,}")
    elif DOWNSAMPLE_COVTYPE and original_len <= TARGET_TOTAL_COVTYPE:
        print(f"\n  covtype 原始数据 {original_len:,} 行不足 {TARGET_TOTAL_COVTYPE:,}，保留全部数据")
    else:
        print(f"\ncovtype 使用全部原始数据，行数：{original_len:,}")

    df = df.fillna(-1)
    return df

# ===================== 统一划分与保存函数 =====================
def split_and_save(df, dataset_name):
    """7:3 分层划分训练/测试集，并保存为 CSV 文件"""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train_path = f"data/processed/{dataset_name}_train.csv"
    test_path = f"data/processed/{dataset_name}_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"\n{dataset_name} 预处理与划分完成")
    print(f"   总样本数：{len(df):,} | 训练集：{len(train):,} (70%) | 测试集：{len(test):,} (30%)")
    print(f"   标签类别数：{len(y.unique())}")
    print(f"   训练集标签分布：{y_train.value_counts().sort_index().tolist()}")
    print(f"   测试集标签分布：{y_test.value_counts().sort_index().tolist()}")
    print("-" * 60)

# ===================== 主执行函数 =====================
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    print("开始预处理数据集...")
    print("=" * 60)

    split_and_save(preprocess_adult(), "adult")
    split_and_save(preprocess_credit_g(), "credit-g")
    split_and_save(preprocess_covtype(), "covtype")

    print("\n所有数据集预处理完成！")
    print("结果保存路径：data/processed/")
    print("生成文件：")
    print("   - adult_train.csv / adult_test.csv")
    print("   - credit-g_train.csv / credit-g_test.csv")
    print("   - covtype_train.csv / covtype_test.csv")