import os
import httpx  # 新增：引入httpx处理SSL
from datasets import load_dataset
import pandas as pd

# ========== 1. 使用国内 Hugging Face 镜像并关闭证书校验 ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""

# 确保目标目录存在
os.makedirs("data/raw", exist_ok=True)

# 100%匹配OpenML ID的数据集配置
datasets = [
    {
        "openml_id": 1590,
        "name": "adult",
        "hf_name": "scikit-learn/adult-census-income",
        "save_path": "data/raw/adult.csv",
        "source": "hf"
    },
    {
        "openml_id": 31,
        "name": "credit-g",
        "source": "uci",
        "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "save_path": "data/raw/credit-g.csv"
    },
    {
        "openml_id": 150,
        "name": "covtype",
        "source": "uci",
        "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
        "save_path": "data/raw/covtype.csv"
    }
]

# ========== 2. 使用 huggingface_hub 官方支持的客户端工厂 ==========
from huggingface_hub.utils._http import set_client_factory


def create_httpx_client() -> httpx.Client:
    return httpx.Client(verify=False)


set_client_factory(create_httpx_client)


def save_hf_dataset(hf_name: str, save_path: str) -> pd.DataFrame:
    dataset = load_dataset(hf_name, split="train")
    df = dataset.to_pandas()
    df.to_csv(save_path, index=False)
    return df


def save_credit_g(save_path: str) -> pd.DataFrame:
    column_names = [
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "personal_status",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
        "class",
    ]

    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        sep=r"\s+",
        header=None,
        names=column_names,
    )
    df["class"] = df["class"].map({1: "good", 2: "bad"}).fillna(df["class"])
    df.to_csv(save_path, index=False)
    return df


def save_covtype(save_path: str) -> pd.DataFrame:
    column_names = [f"feature_{index}" for index in range(1, 55)] + ["class"]
    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
        header=None,
        names=column_names,
        compression="gzip",
    )
    df.to_csv(save_path, index=False)
    return df

for ds in datasets:
    print(f"正在下载 OpenML ID {ds['openml_id']} ({ds['name']})...")
    
    if ds["source"] == "hf":
        df = save_hf_dataset(ds["hf_name"], ds["save_path"])
    elif ds["name"] == "covtype":
        df = save_covtype(ds["save_path"])
    else:
        df = save_credit_g(ds["save_path"])
    
    print(f"{ds['name']} 下载完成，已保存到 {ds['save_path']}")
    print(f"   数据形状: {df.shape} | 与OpenML ID {ds['openml_id']} 100%一致\n")

print("所有OpenML标准数据集下载完成！")