# code/10_scalability_baseline.py
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import load_scalability_dataset, evaluate_model, save_result, SEED

# 每个数据集实际可用的样本量（与生成的数据集一致）
DATASET_SIZES = {
    "adult": [1000, 10000],
    "covtype": [1000, 10000, 100000],
    "credit-g": [700]   # credit-g 全量 700
}

SAVE_DIR = "results/baseline_scalability"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_scalability_experiment():
    for dataset, sizes in DATASET_SIZES.items():
        print(f"\n{'='*60}\n数据集：{dataset}\n{'='*60}")
        for size in sizes:
            try:
                X_train, X_test, y_train, y_test = load_scalability_dataset(dataset, size)
            except FileNotFoundError as e:
                print(f"跳过 {dataset} 样本量 {size}：{e}")
                continue

            print(f"\n--- 样本量：{size} ---")

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