# code/08_summary_results.py
"""
实验结果汇总与可视化脚本（修复版 + TabPFN 显式支持）
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import re

# ==================== 全局配置 ====================
RESULT_ROOT = "results"
SAVE_SUMMARY = "results/summary.csv"
FIG_SAVE = "report/figures"
os.makedirs(FIG_SAVE, exist_ok=True)

BASELINE_MODELS = ["XGBoost", "LightGBM"]

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

LABEL_ACC = "Accuracy"
LABEL_DATASET = "Dataset"
LABEL_MISS_PCT = "Missing Ratio (%)"
TITLE_BASE = "Baseline Model Accuracy Comparison"
TITLE_MISS = "Robustness to Missing Values"

# ==================== 加载所有结果（显式支持 TabPFN） ====================
def load_all_results():
    all_dfs = []
    required_cols = {"dataset", "model", "accuracy", "f1", "auc", "train_time", "infer_time_ms", "peak_memory_mb"}
    
    # 1. 处理普通结果（基线、缺失值、可扩展性）
    for csv_path in glob(f"{RESULT_ROOT}/**/*.csv", recursive=True):
        if "tabpfn" in csv_path.lower():
            continue
        try:
            df = pd.read_csv(csv_path)
            if not required_cols.issubset(df.columns):
                print(f"Skipped {csv_path}: missing required columns")
                continue
            df["model"] = df["model"].astype(str).str.strip()
            df = df[(df["accuracy"] >= 0) & (df["accuracy"] <= 1)]
            all_dfs.append(df)
            print(f"Loaded: {csv_path} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
    
    # 2. 单独处理 TabPFN 结果
    tabpfn_files = glob("results/tabpfn/*.csv")
    for csv_path in tabpfn_files:
        try:
            df = pd.read_csv(csv_path)
            # 列名映射
            col_map = {
                "Dataset": "dataset",
                "Accuracy": "accuracy",
                "AUC": "auc",
                "F1": "f1",
                "F1-macro": "f1",
                "F1-weighted": "f1"
            }
            df = df.rename(columns=col_map)
            if "dataset" not in df.columns or "accuracy" not in df.columns:
                print(f"Skipped {csv_path}: missing dataset or accuracy column")
                continue
            df["model"] = "TabPFN"
            df["train_time"] = -1.0
            df["infer_time_ms"] = -1.0
            df["peak_memory_mb"] = -1.0
            if "auc" not in df.columns:
                df["auc"] = -1.0
            if "f1" not in df.columns:
                df["f1"] = -1.0
            df = df[["dataset", "model", "accuracy", "f1", "auc", "train_time", "infer_time_ms", "peak_memory_mb"]]
            df = df[(df["accuracy"] >= 0) & (df["accuracy"] <= 1)]
            all_dfs.append(df)
            print(f"Loaded TabPFN: {csv_path} ({len(df)} rows)")
        except Exception as e:
            print(f"Failed to load TabPFN {csv_path}: {e}")
    
    if not all_dfs:
        return None
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["dataset", "model"]).reset_index(drop=True)
    return full

# ==================== 基线准确率柱状图 ====================
def draw_baseline_accuracy(df):
    base = df[~df["model"].str.contains("missing", case=False, na=False)].copy()
    base = base[base["model"].isin(BASELINE_MODELS)]
    if base.empty:
        print("No baseline data, skip bar chart")
        return
    plt.figure(figsize=(11, 5))
    sns.barplot(data=base, x="dataset", y="accuracy", hue="model", palette="Blues_d")
    plt.title(TITLE_BASE, fontsize=14)
    plt.xlabel(LABEL_DATASET)
    plt.ylabel(LABEL_ACC)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE, "base_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Baseline accuracy chart saved")

# ==================== 缺失值趋势图 ====================
def draw_missing_trend_separate(df):
    miss_df = df[df["model"].str.contains(r"_\d+missing$", regex=True, na=False)].copy()
    if miss_df.empty:
        print("No missing value experiment results, skip trend plots")
        return

    def extract_miss_percent(s):
        m = re.search(r"_(\d+)missing$", s)
        return int(m.group(1)) if m else -1

    miss_df["miss_percent"] = miss_df["model"].apply(extract_miss_percent)
    miss_df = miss_df[miss_df["miss_percent"] >= 0]
    miss_df["base_model"] = miss_df["model"].str.replace(r"_\d+missing$", "", regex=True)

    datasets = miss_df["dataset"].unique()
    for ds in datasets:
        sub = miss_df[miss_df["dataset"] == ds].copy()
        if sub.empty:
            continue

        print(f"\nMissing trend for {ds}:")
        for bm in sub["base_model"].unique():
            pts = sub[sub["base_model"] == bm][["miss_percent", "accuracy"]].sort_values("miss_percent")
            print(f"  {bm}: {pts.values.tolist()}")

        plt.figure(figsize=(6, 4))
        for bm in sub["base_model"].unique():
            bm_sub = sub[sub["base_model"] == bm].sort_values("miss_percent")
            if len(bm_sub) < 2:
                plt.plot(bm_sub["miss_percent"], bm_sub["accuracy"],
                         marker='o', linestyle='', label=bm)
                print(f"Warning: {ds} - {bm} has less than 2 points, only markers")
            else:
                plt.plot(bm_sub["miss_percent"], bm_sub["accuracy"],
                         marker='o', label=bm, linewidth=2)

        plt.title(f"{ds} - {TITLE_MISS}")
        plt.xlabel(LABEL_MISS_PCT)
        plt.ylabel(LABEL_ACC)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Model")
        plt.tight_layout()
        save_path = os.path.join(FIG_SAVE, f"missing_trend_{ds}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

# ==================== 可扩展性曲线 ====================
def draw_scalability(df):
    scal_df = df[df["model"].str.contains(r"_\d+samples$", regex=True, na=False)].copy()
    if scal_df.empty:
        print("No scalability results found, skip")
        return

    def extract_sample_size(s):
        m = re.search(r"_(\d+)samples$", s)
        return int(m.group(1)) if m else -1

    scal_df["sample_size"] = scal_df["model"].apply(extract_sample_size)
    scal_df = scal_df[scal_df["sample_size"] > 0]
    scal_df["base_model"] = scal_df["model"].str.replace(r"_\d+samples$", "", regex=True)

    datasets = scal_df["dataset"].unique()
    for ds in datasets:
        sub = scal_df[scal_df["dataset"] == ds].sort_values("sample_size")
        if sub.empty:
            continue

        print(f"\nScalability for {ds}:")
        for bm in sub["base_model"].unique():
            pts = sub[sub["base_model"] == bm][["sample_size", "accuracy"]].sort_values("sample_size")
            print(f"  {bm}: {pts.values.tolist()}")

        plt.figure(figsize=(6, 4))
        for bm in sub["base_model"].unique():
            bm_sub = sub[sub["base_model"] == bm].sort_values("sample_size")
            plt.plot(bm_sub["sample_size"], bm_sub["accuracy"],
                     marker='s', label=bm, linewidth=2)
        plt.title(f"{ds} - Scalability")
        plt.xlabel("Sample Size")
        plt.ylabel(LABEL_ACC)
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(FIG_SAVE, f"scalability_{ds}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    data = load_all_results()
    if data is None:
        print("No valid result data. Please run experiments first.")
    else:
        data.to_csv(SAVE_SUMMARY, index=False, encoding="utf-8")
        print(f"Summary saved to {SAVE_SUMMARY}, total {len(data)} records")
        draw_baseline_accuracy(data)
        draw_missing_trend_separate(data)
        draw_scalability(data)
        print(f"All figures saved to {FIG_SAVE}")