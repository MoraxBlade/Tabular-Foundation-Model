import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import os

# ===================== 全局配置 =====================
SEED = 42
np.random.seed(SEED)
TEST_SIZE = 0.3

# 针对表格大模型的数据集降采样配置
ADULT_SAMPLE_SIZE = 10000      # adult 降采样后的总样本量（划分前）
COVTYPE_SAMPLE_SIZE = 50000    # covtype 降采样后的总样本量（划分前）

# ===================== 1. 基础数据清洗 =====================
def get_clean_adult():
    """处理 UCI 经典 adult 数据集，支持分层降采样至 ADULT_SAMPLE_SIZE 行"""
    print("正在从 ucimlrepo 在线获取 adult (ID: 2) 数据集...")
    adult = fetch_ucirepo(id=2)
    
    X = adult.data.features
    y = adult.data.targets
    df = pd.concat([X, y], axis=1)
    
    # 清理所有字符列前后的多余空格，并将 '?' 转换为真正的缺失值 np.nan
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('?', np.nan)
        df[col] = df[col].replace('nan', np.nan)
        
    # 标签转换：训练集含有 '<=50K'，测试集含有 '<=50K.'，统一归一化
    df.iloc[:, -1] = df.iloc[:, -1].map({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})
    
    # 过滤掉无法解析标签的异常行
    df = df.dropna(subset=[df.columns[-1]])
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    
    # ----- Adult 降采样逻辑（分层采样）-----
    original_len = len(df)
    if original_len > ADULT_SAMPLE_SIZE:
        print(f"[-] adult 原始数据 {original_len:,} 行，需降采样至 {ADULT_SAMPLE_SIZE:,} 行（分层采样）")
        y_label = df.iloc[:, -1]
        # 使用 train_test_split 进行分层采样，保留目标行数作为总样本
        downsampled_df, _ = train_test_split(
            df,
            train_size=ADULT_SAMPLE_SIZE,
            random_state=SEED,
            stratify=y_label
        )
        df = downsampled_df.reset_index(drop=True)
        print(f"   降采样完成，总样本数：{len(df):,}")
    else:
        print(f"[-] adult 原始数据 {original_len:,} 行，未超过 {ADULT_SAMPLE_SIZE}，保留全部数据")
    
    return df

def get_clean_credit_g():
    """处理德国信用数据集，直接从 ucimlrepo 获取（无降采样）"""
    print("正在从 ucimlrepo 在线获取 credit-g (Statlog, ID: 144) 数据集...")
    credit = fetch_ucirepo(id=144)
    
    X = credit.data.features
    y = credit.data.targets
    df = pd.concat([X, y], axis=1)
    
    # 标签转换：原数据中 1->Good (赋值为 1), 2->Bad (赋值为 0)
    df.iloc[:, -1] = df.iloc[:, -1].map({1: 1, 2: 0}).astype(int)
    return df

def get_clean_covtype():
    """处理森林覆盖类型数据集，支持分层降采样至 COVTYPE_SAMPLE_SIZE 行"""
    print("正在从 ucimlrepo 在线获取 covtype (ID: 31) 数据集，这可能需要一点时间...")
    covertype = fetch_ucirepo(id=31)
    
    # 提取特征和标签并拼接，标签放在最后一列
    X = covertype.data.features
    y = covertype.data.targets
    df = pd.concat([X, y], axis=1)
    
    # 标签转换：原始标签 1~7 → 0~6
    df.iloc[:, -1] = df.iloc[:, -1] - 1
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    
    # ----- Covtype 降采样逻辑（分层采样）-----
    original_len = len(df)
    if original_len > COVTYPE_SAMPLE_SIZE:
        print(f"[-] covtype 原始数据 {original_len:,} 行，需降采样至 {COVTYPE_SAMPLE_SIZE:,} 行（分层采样）")
        y_label = df.iloc[:, -1]
        downsampled_df, _ = train_test_split(
            df,
            train_size=COVTYPE_SAMPLE_SIZE,
            random_state=SEED,
            stratify=y_label
        )
        df = downsampled_df.reset_index(drop=True)
        print(f"   降采样完成，总样本数：{len(df):,}")
    else:
        print(f"[-] covtype 原始数据 {original_len:,} 行，未超过 {COVTYPE_SAMPLE_SIZE}，保留全部数据")
    
    return df

# ===================== 2. 基线模型特征转换 =====================
def generate_baseline_features(df):
    """将清洗后的纯净数据转换为树模型适用的粗糙格式 (Label Encoding + -1填充)"""
    df_base = df.copy()
    # 分类特征标签编码（所有 object 类型列）
    cat_cols = df_base.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        # 注: pandas 的 cat.codes 会自动把 NaN 映射为 -1
        df_base[col] = df_base[col].astype("category").cat.codes
    
    # 提取剩下的缺失值（原数值列），统一填充 -1
    df_base = df_base.fillna(-1)
    return df_base

# ===================== 3. 统一划分与双轨保存 =====================
def split_and_save(df_clean, dataset_name):
    """同时切分 baseline 和 tfm 的特征集合，保证行列完全一致"""
    # 优先生成 baseline 专用的粗糙特征集
    df_base = generate_baseline_features(df_clean)
    
    # 将目标标签 y 分开（由于 clean 和 base 的 y 是完全一致的）
    y = df_clean.iloc[:, -1]
    
    # 核心：将 df_clean 和 df_base 放在同一个 train_test_split 里面，
    # 这样相同的随机种子切分出来的行索引绝对一样，保证 100% 公平对比。
    train_clean, test_clean, train_base, test_base = train_test_split(
        df_clean, df_base, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    # -- 保存为 Baseline 树模型使用 (保持在原目录，以免改动原有基线脚本) --
    base_dir = "data/processed"
    os.makedirs(base_dir, exist_ok=True)
    train_base.to_csv(f"{base_dir}/{dataset_name}_train.csv", index=False)
    test_base.to_csv(f"{base_dir}/{dataset_name}_test.csv", index=False)
    
    # -- 保存为 TFM 表格大模型使用 (新建带语义信息的 tfm 目录) --
    tfm_dir = "data/processed/tfm"
    os.makedirs(tfm_dir, exist_ok=True)
    train_clean.to_csv(f"{tfm_dir}/{dataset_name}_train.csv", index=False)
    test_clean.to_csv(f"{tfm_dir}/{dataset_name}_test.csv", index=False)
    
    print(f"\n[OK] {dataset_name} 双轨数据处理完成")
    print(f"   总样本数：{len(df_clean):,} | 训练集：{len(train_clean):,} | 测试集：{len(test_clean):,}")
    print(f"   保存 Baseline 版本至: {base_dir}/...")
    print(f"   保存 TFM 版本至    : {tfm_dir}/...")

# ===================== 主执行函数 =====================
if __name__ == "__main__":
    print("开始预处理数据集 (Baseline & TFM 双轨制)...")
    print("=" * 60)
    
    split_and_save(get_clean_adult(), "adult")
    split_and_save(get_clean_credit_g(), "credit-g")
    split_and_save(get_clean_covtype(), "covtype")
    
    print("\n所有数据集双轨制数据生成完毕！")