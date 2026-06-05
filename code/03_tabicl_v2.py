import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from tabpfn_client import TabPFNClassifier, set_access_token
import warnings

warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
API_TOKEN = "tabpfn_sk_Fu7kHdLwAbFV6-HsaTz26DW-EGjZJ5Eo3kHctr5gq7A"
set_access_token(API_TOKEN)

DATASETS = ["credit-g", "adult", "covtype"]

TFM_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/processed/tfm")
)

# ✅ 结果保存目录
RESULT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../results/tabpfn")
)
os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_tabpfn(dataset_name):
    print(f"\n{'='*20} 正在评估数据集: {dataset_name} {'='*20}")

    train_path = os.path.join(TFM_DATA_DIR, f"{dataset_name}_train.csv")
    test_path = os.path.join(TFM_DATA_DIR, f"{dataset_name}_test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"错误: 找不到 {dataset_name} 的 TFM 数据")
        return

    # 1. 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 2. 训练
    print("开始训练 (发送至 TabPFN API)...")
    model = TabPFNClassifier(n_estimators=8)
    model.fit(X_train, y_train)

    # 3. 预测
    print("开始预测...")
    predictions = model.predict(X_test)

    # 4. 指标计算
    acc = accuracy_score(y_test, predictions)
    unique_classes = len(y_train.unique())

    result_dict = {
        "Dataset": dataset_name,
        "Accuracy": round(acc, 4),
    }

    if unique_classes == 2:
        pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        result_dict.update({
            "AUC": round(auc, 4),
            "F1": round(f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
        })

        print(
            f"[结果 - {dataset_name}] "
            f"Acc: {acc:.4f} | AUC: {auc:.4f} | "
            f"F1: {f1:.4f}"
        )
    else:
        # 多分类：macro / weighted
        f1_macro = f1_score(y_test, predictions, average="macro")
        f1_weighted = f1_score(y_test, predictions, average="weighted")

        result_dict.update({
            "F1-macro": round(f1_macro, 4),
            "F1-weighted": round(f1_weighted, 4),
        })

        print(
            f"[结果 - {dataset_name}] "
            f"Acc: {acc:.4f} | "
            f"F1-macro: {f1_macro:.4f} | "
            f"F1-weighted: {f1_weighted:.4f}"
        )

    # 5. 保存结果表格
    result_df = pd.DataFrame([result_dict])
    save_path = os.path.join(RESULT_DIR, f"{dataset_name}_tabpfn_result.csv")
    result_df.to_csv(save_path, index=False)

    print(f"✅ 结果已保存至: {save_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    for ds in DATASETS:
        evaluate_tabpfn(ds)