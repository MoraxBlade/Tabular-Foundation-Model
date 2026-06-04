import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ===================== 全局配置（绝对不要修改） =====================
SEED = 42
np.random.seed(SEED)
TEST_SIZE = 0.3

# 标准数据集样本量（用于完整性校验）
# 注意：adult 为 UCI 经典训练集的数据行数（不含表头），共 32,561 条数据
EXPECTED_ROWS = {
    "adult": 32561,      # 数据行数，文件含表头共 32562 行
    "credit-g": 1000,    # 数据行数，文件含表头共 1001 行
    "covtype": 581012    # 数据行数，文件含表头共 581013 行
}

# ===================== 数据集预处理函数 =====================
def preprocess_adult():
    """处理 UCI 经典 adult 收入预测数据集（32,561 条数据）
    修复：1. 正确识别 `?` 缺失值  2. 标签前后的空格问题
    """
    df = pd.read_csv(
        "data/raw/adult.csv",
        na_values=["?"],
        skipinitialspace=True   # 自动去除字段前后的空格（解决标签空格问题）
    )
    
    # 完整性校验
    if len(df) != EXPECTED_ROWS["adult"]:
        raise ValueError(
            f"Adult 数据集行数异常！预期 {EXPECTED_ROWS['adult']} 行（数据行），"
            f"实际 {len(df)} 行。请检查原始文件是否完整，或是否包含多余的表头行。"
        )
    
    print(f"adult 数据集加载成功，数据行数：{len(df):,}")
    print(f"特征列数：{df.shape[1]-1}，标签列：{df.columns[-1]}")
    
    # 标签转换：<=50K → 0，>50K → 1
    df.iloc[:, -1] = df.iloc[:, -1].map({"<=50K": 0, ">50K": 1}).astype(int)
    
    # 分类特征标签编码（所有 object 类型列）
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"分类特征数量：{len(cat_cols)}，数值特征数量：{df.shape[1]-1-len(cat_cols)}")
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    
    # 统一填充缺失值（原始数据中 `?` 已转为 NaN）
    missing_before = df.isnull().sum().sum()
    df = df.fillna(-1)
    missing_after = df.isnull().sum().sum()
    print(f"缺失值处理：填充前 {missing_before} 个，填充后 {missing_after} 个")
    
    return df

def preprocess_credit_g():
    """处理德国信用数据集（OpenML ID: 31）"""
    df = pd.read_csv("data/raw/credit-g.csv")
    
    if len(df) != EXPECTED_ROWS["credit-g"]:
        print(f"警告：credit-g 数据集行数异常！预期 {EXPECTED_ROWS['credit-g']} 行，实际 {len(df)} 行")
        print("请从 https://www.openml.org/d/31 重新下载完整版本")
    
    # 标签转换：bad → 0，good → 1
    df.iloc[:, -1] = df.iloc[:, -1].map({"bad": 0, "good": 1}).astype(int)
    
    # 分类特征编码
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    
    df = df.fillna(-1)
    return df

def preprocess_covtype():
    """处理森林覆盖类型数据集（OpenML ID: 150）"""
    # 注意：下载脚本保存的文件名为 covtype_raw.csv
    df = pd.read_csv("data/raw/covtype_raw.csv")
    
    if len(df) != EXPECTED_ROWS["covtype"]:
        print(f"警告：covtype 数据集行数异常！预期 {EXPECTED_ROWS['covtype']} 行，实际 {len(df)} 行")
        print("请从 https://www.openml.org/d/150 重新下载完整版本")
    
    # 标签转换：原始标签 1~7 → 0~6
    df.iloc[:, -1] = df.iloc[:, -1] - 1
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    
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
    
    # 处理三个数据集
    split_and_save(preprocess_adult(), "adult")
    split_and_save(preprocess_credit_g(), "credit-g")
    split_and_save(preprocess_covtype(), "covtype")
    
    print("\n所有数据集预处理完成！")
    print("结果保存路径：data/processed/")
    print("生成文件：")
    print("   - adult_train.csv / adult_test.csv")
    print("   - credit-g_train.csv / credit-g_test.csv")
    print("   - covtype_train.csv / covtype_test.csv")