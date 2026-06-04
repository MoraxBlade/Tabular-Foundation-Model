# code/07_baseline_missing_experiment.py
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import load_missing_dataset, evaluate_model, save_result, SEED

MISSING_RATIOS = [0,0.1,0.2,0.3]
DATA_LIST = ["adult","credit-g","covtype"]
SAVE_FOLDER = "results/baseline_missing"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def run_all_missing_exp():
    for ds in DATA_LIST:
        print("开始数据集：", ds)
        for r in MISSING_RATIOS:
            r_str = f"{int(r*100)}missing"
            print("缺失比例：",r_str)
            X_train,X_test,y_train,y_test = load_missing_dataset(ds, r_str)
            
            # XGB
            xgb = XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric="logloss")
            res_xgb = evaluate_model(xgb,X_train,X_test,y_train,y_test,"XGBoost_"+r_str,ds)
            save_result(res_xgb, os.path.join(SAVE_FOLDER,"xgboost_missing.csv"))
            
            # LGB
            lgb = LGBMClassifier(random_state=SEED, verbose=-1)
            res_lgb = evaluate_model(lgb,X_train,X_test,y_train,y_test,"LightGBM_"+r_str,ds)
            save_result(res_lgb, os.path.join(SAVE_FOLDER,"lightgbm_missing.csv"))

if __name__ == "__main__":
    run_all_missing_exp()