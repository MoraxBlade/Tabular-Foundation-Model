import os
from utils import load_dataset, load_missing_dataset, load_scalability_dataset, evaluate_model, save_result, SEED
from tabicl import TabICLClassifier

MODEL_NAME = "TabICL"
SAVE_DIR = "results/tabicl"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_basic():
    for ds in ["adult", "credit-g", "covtype"]:
        X_train, X_test, y_train, y_test = load_dataset(ds)
        model = TabICLClassifier(device="cpu", random_state=
                                 SEED)
        result = evaluate_model(model, X_train, X_test, y_train, y_test, MODEL_NAME, ds)
        save_result(result, os.path.join(SAVE_DIR, "basic.csv"))

def run_missing():
    ratios = [0, 0.1, 0.2, 0.3]
    for ds in ["adult", "credit-g", "covtype"]:
        for r in ratios:
            r_str = f"{int(r*100)}missing"
            X_train, X_test, y_train, y_test = load_missing_dataset(ds, r_str)
            model = TabICLClassifier(device="cpu", random_state=SEED)
            result = evaluate_model(model, X_train, X_test, y_train, y_test, f"{MODEL_NAME}_{r_str}", ds)
            save_result(result, os.path.join(SAVE_DIR, "missing.csv"))

def run_scalability():
    sizes_map = {     
        "adult": [1000, 1626, 2645, 4303, 7000],  # 只包含已经存在的 CSV
        "covtype": [1000, 2432, 5916, 14389, 35000],
        "credit-g": [700]
         }
        
    for ds, sizes in sizes_map.items():
        for size in sizes:
            X_train, X_test, y_train, y_test = load_scalability_dataset(ds, size)
            model = TabICLClassifier(device="cpu", random_state=SEED)
            result = evaluate_model(model, X_train, X_test, y_train, y_test, f"{MODEL_NAME}_{size}samples", ds)
            save_result(result, os.path.join(SAVE_DIR, "scalability.csv"))

if __name__ == "__main__":
     run_basic()
     run_missing()
     run_scalability()