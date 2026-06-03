# code/08_summary_results.py
import pandas as pd
import numpy as np
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

# 指定基线对比的模型（排除测试模型如 LogisticRegression）
BASELINE_MODELS = ["XGBoost", "LightGBM"]

# 强制使用英文字体，避免中文方框
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 所有标签统一英文
LABEL_ACC = "Accuracy"
LABEL_DATASET = "Dataset"
LABEL_MISS_PCT = "Missing Ratio (%)"
TITLE_BASE = "Baseline Model Accuracy Comparison"
TITLE_MISS = "Robustness to Missing Values"

# ==================== 加载所有结果 ====================
def load_all_results():
    all_dfs = []
    required_cols = {"dataset", "model", "accuracy", "f1", "auc", "train_time", "infer_time_ms", "peak_memory_mb"}
    for csv_path in glob(f"{RESULT_ROOT}/**/*.csv", recursive=True):
        try:
            df = pd.read_csv(csv_path)
            if not required_cols.issubset(df.columns):
                print(f"Skipped {csv_path}: missing required columns")
                continue
            df["model"] = df["model"].astype(str).str.strip()
            all_dfs.append(df)
            print(f"Loaded: {csv_path}")
        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
    if not all_dfs:
        return None
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["dataset", "model"]).reset_index(drop=True)
    return full

# ==================== Baseline bar chart ====================
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

# ==================== Missing value trend (separate per dataset) ====================
def draw_missing_trend_separate(df):
    miss_df = df[df["model"].str.contains("missing", case=False, na=False)].copy()
    if miss_df.empty:
        print("No missing value experiment results, skip trend plots")
        return

    def extract_miss_percent(s):
        m = re.search(r"(\d+)missing", s)
        return int(m.group(1)) if m else -1
    miss_df["miss_percent"] = miss_df["model"].apply(extract_miss_percent)
    miss_df = miss_df[miss_df["miss_percent"] >= 0]
    miss_df["base_model"] = miss_df["model"].str.replace(r"_\d+missing", "", regex=True)

    datasets = miss_df["dataset"].unique()
    for ds in datasets:
        sub = miss_df[miss_df["dataset"] == ds].copy()
        if sub.empty:
            continue
        
        plt.figure(figsize=(6, 4))
        for bm in sub["base_model"].unique():
            bm_sub = sub[sub["base_model"] == bm].sort_values("miss_percent")
            if len(bm_sub) < 2:
                plt.plot(bm_sub["miss_percent"], bm_sub["accuracy"], marker='o', linestyle='', label=bm)
                print(f"Warning: {ds} - {bm} has less than 2 points, only markers")
            else:
                plt.plot(bm_sub["miss_percent"], bm_sub["accuracy"], marker='o', label=bm, linewidth=2)
        
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

# ==================== Scalability curves (optional) ====================
def draw_scalability(df):
    scal_df = df[df["model"].str.contains(r"_\d+k$", regex=True, na=False)].copy()
    if scal_df.empty:
        print("No scalability results found, skip")
        return
    scal_df["sample_size"] = scal_df["model"].str.extract(r"_(\d+)k$").astype(int)
    scal_df["base_model"] = scal_df["model"].str.replace(r"_\d+k$", "", regex=True)
    datasets = sorted(scal_df["dataset"].unique())
    for ds in datasets:
        sub = scal_df[scal_df["dataset"] == ds].sort_values("sample_size")
        if sub.empty:
            continue
        plt.figure(figsize=(6, 4))
        for bm in sub["base_model"].unique():
            bm_sub = sub[sub["base_model"] == bm].sort_values("sample_size")
            plt.plot(bm_sub["sample_size"], bm_sub["accuracy"], marker='s', label=bm, linewidth=2)
        plt.title(f"{ds} - Scalability")
        plt.xlabel("Sample Size (k)")
        plt.ylabel(LABEL_ACC)
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(FIG_SAVE, f"scalability_{ds}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

# ==================== Main ====================
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