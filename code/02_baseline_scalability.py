# code/10_scalability_baseline.py
import os
import glob
import re
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import evaluate_model, save_result, SEED

SAVE_DIR = "results/baseline_scalability"
os.makedirs(SAVE_DIR, exist_ok=True)

def discover_experiments():
    """自动扫描 data/scalability/ 中所有可扩展性数据集"""
    exps = []
    for filepath in glob.glob("data/scalability/*_train_*samples.csv"):
        basename = os.path.basename(filepath)
        match = re.match(r"(.+)_train_(\d+)samples\.csv", basename)
        if match:
            dataset = match.group(1)
            size = int(match.group(2))
            exps.append((dataset, size))
    # 按数据集和样本量排序
    return sorted(set(exps))

def run_scalability_experiment():
    experiments = discover_experiments()
    if not experiments:
        print("未找到任何可扩展性数据，请先运行 01_scalability_dataset.py")
        return

    print(f"发现 {len(experiments)} 个可扩展性实验：")
    for ds, sz in experiments:
        print(f"  {ds} - {sz} samples")

    for dataset, size in experiments:
        print(f"\n{'='*60}\n数据集：{dataset}, 样本量：{size}\n{'='*60}")
        train_path = f"data/scalability/{dataset}_train_{size}samples.csv"
        test_path = f"data/processed/{dataset}_test.csv"

        if not os.path.exists(test_path):
            print(f"  测试集不存在：{test_path}，跳过")
            continue

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        except Exception as e:
            print(f"  读取数据失败：{e}，跳过")
            continue

        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

        print(f"  训练集大小：{len(X_train)}，测试集大小：{len(X_test)}")

        # XGBoost
        xgb = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric="logloss")
        res_xgb = evaluate_model(xgb, X_train, X_test, y_train, y_test,
                                 model_name=f"XGBoost_{size}samples", dataset_name=dataset)
        save_result(res_xgb, os.path.join(SAVE_DIR, "xgboost_scalability.csv"))

        # LightGBM
        lgb = LGBMClassifier(random_state=SEED, verbose=-1)
        res_lgb = evaluate_model(lgb, X_train, X_test, y_train, y_test,
                                 model_name=f"LightGBM_{size}samples", dataset_name=dataset)
        save_result(res_lgb, os.path.join(SAVE_DIR, "lightgbm_scalability.csv"))

if __name__ == "__main__":
    run_scalability_experiment()
    print(f"\n基线可扩展性实验完成，结果保存至：{SAVE_DIR}")