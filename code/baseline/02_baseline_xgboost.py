from xgboost import XGBClassifier
from utils import load_dataset, evaluate_model, save_result
import os

SEED = 42

def run_xgboost(dataset_name):
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    model = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric="logloss")
    result = evaluate_model(model, X_train, X_test, y_train, y_test, "XGBoost", dataset_name)
    save_result(result, "results/baseline/xgboost_results.csv")

if __name__ == "__main__":
    os.makedirs("results/baseline", exist_ok=True)
    run_xgboost("adult")
    run_xgboost("credit-g")
    run_xgboost("covtype")  # 最后跑