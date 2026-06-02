from lightgbm import LGBMClassifier
from utils import load_dataset, evaluate_model, save_result
import os

SEED = 42

def run_lightgbm(dataset_name):
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    model = LGBMClassifier(random_state=SEED, verbose=-1)
    result = evaluate_model(model, X_train, X_test, y_train, y_test, "LightGBM", dataset_name)
    save_result(result, "results/baseline/lightgbm_results.csv")

if __name__ == "__main__":
    os.makedirs("results/baseline", exist_ok=True)
    run_lightgbm("adult")
    run_lightgbm("credit-g")
    run_lightgbm("covtype")