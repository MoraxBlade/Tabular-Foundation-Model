import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ===================== 全局配置（绝对不要修改） =====================
SEED = 42
np.random.seed(SEED)
TEST_SIZE = 0.3

# 标准数据集样本量（用于完整性校验）
# 注意：adult此处为UCI经典训练集总行数（32561条数据 + 1行表头 = 32563）
EXPECTED_ROWS = {
    "adult": 32561,
    "credit-g": 1000,
    "covtype": 581012
}

# ===================== 数据集预处理函数 =====================
def preprocess_adult():
    """处理UCI经典adult收入预测数据集（32563行版本）
    修复：1. 标签前的空格问题 2. 正确识别`?`缺失值
    """
    # 关键修复：指定na_values为["?"]，正确识别原始数据中的缺失值
    df = pd.read_csv(
        "data/raw/adult.csv",
        na_values=["?"],
        skipinitialspace=True  # 自动去除字段前后的空格（解决标签空格问题）
    )
    
    # 完整性校验（精准拦截你之前遇到的CSV解析异常）
    if len(df) != EXPECTED_ROWS["adult"]:
        print(f"    警告：adult数据集行数异常！")
        print(f"   预期：{EXPECTED_ROWS['adult']} 行（含表头）")
        print(f"   实际：{len(df)} 行")
        print("   排查建议：")
        print("   1. 用VS Code打开原始CSV，确认右下角总行数是否为32563")
        print("   2. 检查是否存在字段内换行符（双引号包裹的多行文本）")
        print("   3. 确认下载的是UCI训练集而非测试集或合并集")
        exit(1)  # 行数异常直接退出，避免生成错误数据
    
    print(f"adult数据集加载成功，总行数：{len(df)}（含表头）")
    print(f"数据列数：{len(df.columns)} | 标签列：{df.columns[-1]}")
    
    # 标签转换：<=50K → 0，>50K → 1
    # 已通过skipinitialspace自动去除空格，无需再处理" <=50K"这种情况
    df.iloc[:, -1] = df.iloc[:, -1].map({"<=50K": 0, ">50K": 1}).astype(int)
    
    # 分类特征标签编码（所有object类型列）
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"   分类特征数量：{len(cat_cols)} | 数值特征数量：{len(df.columns)-1-len(cat_cols)}")
    
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    
    # 填充缺失值（原始数据中`?`已转为NaN，此处统一填充-1）
    missing_before = df.isnull().sum().sum()
    df = df.fillna(-1)
    missing_after = df.isnull().sum().sum()
    print(f"   缺失值处理：填充前 {missing_before} 个，填充后 {missing_after} 个")
    
    return df

def preprocess_credit_g():
    """处理德国信用数据集（OpenML ID: 31）"""
    df = pd.read_csv("data/raw/credit-g.csv")
    
    if len(df) != EXPECTED_ROWS["credit-g"]:
        print(f"警告：credit-g数据集行数异常！预期{EXPECTED_ROWS['credit-g']}行，实际{len(df)}行")
        print("   请从 https://www.openml.org/d/31 重新下载完整版本")
    
    df.iloc[:, -1] = df.iloc[:, -1].map({"bad": 0, "good": 1}).astype(int)
    
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    
    df = df.fillna(-1)
    return df

def preprocess_covtype():
    """处理森林覆盖类型数据集（OpenML ID: 150）"""
    df = pd.read_csv("data/raw/covtype.csv")
    
    if len(df) != EXPECTED_ROWS["covtype"]:
        print(f"警告：covtype数据集行数异常！预期{EXPECTED_ROWS['covtype']}行，实际{len(df)}行")
        print("   请从 https://www.openml.org/d/150 重新下载完整版本")
    
    df.iloc[:, -1] = df.iloc[:, -1] - 1
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    
    df = df.fillna(-1)
    return df

# ===================== 统一划分与保存函数 =====================
def split_and_save(df, dataset_name):
    """7:3分层划分训练测试集并保存"""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 分层划分（保证训练测试集标签分布一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    
    # 保存文件（index=False确保不生成额外的行号列）
    train.to_csv(f"data/processed/{dataset_name}_train.csv", index=False)
    test.to_csv(f"data/processed/{dataset_name}_test.csv", index=False)
    
    # 打印详细结果（方便你验证划分是否正确）
    print(f"\n{dataset_name} 预处理与划分完成")
    print(f"   总样本数：{len(df)} | 训练集：{len(train)} | 测试集：{len(test)}")
    print(f"   标签类别数：{len(y.unique())}")
    print(f"   训练集标签分布：{y_train.value_counts().sort_index().tolist()}")
    print(f"   测试集标签分布：{y_test.value_counts().sort_index().tolist()}")
    print("-" * 60)

# ===================== 主执行函数 =====================
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("data/processed", exist_ok=True)
    print("开始预处理数据集...")
    print("=" * 60)
    
    # 仅运行adult数据集（注释掉其他不用的）
    split_and_save(preprocess_adult(), "adult")
    
    split_and_save(preprocess_credit_g(), "credit-g")
    split_and_save(preprocess_covtype(), "covtype")
    
    print("所有数据集预处理完成！")
    print("结果保存路径：data/processed/")
    print("生成文件：")
    print("   - adult_train.csv (22794行，70%训练集)")
    print("   - adult_test.csv (9769行，30%测试集)")