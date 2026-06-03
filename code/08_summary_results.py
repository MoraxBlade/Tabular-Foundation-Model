# code/08_summary_results.py
"""
实验结果汇总与可视化脚本

使用说明：
1. 本脚本自动扫描 results/ 下所有子目录中的 CSV 文件，
   要求每个 CSV 必须包含以下列：
   dataset, model, accuracy, f1, auc, train_time, infer_time_ms, peak_memory_mb

2. 模型命名规范（重要！）：
   - 基础实验（原始数据集）：
       模型名直接写，例如 "XGBoost", "LightGBM", "TabPFN", "TabICL"
   - 缺失值实验：
       模型名必须包含 "_数字missing" 后缀，例如 "XGBoost_0missing", "TabPFN_20missing"
       脚本会自动提取缺失百分比并绘制折线图。
   - 可扩展性实验（不同样本量）：
       模型名必须包含 "_数字k" 后缀，例如 "LightGBM_1k", "TabICL_10k"
       脚本会自动提取样本量（千）并绘制曲线。

3. 输出：
   - results/summary.csv：所有结果合并文件
   - report/figures/base_accuracy.png：基线模型准确率柱状图（仅显示 XGBoost/LightGBM）
   - report/figures/missing_trend_{dataset}.png：每个数据集的缺失值鲁棒性折线图
   - report/figures/scalability_{dataset}.png：每个数据集的可扩展性曲线（若有数据）

4. 如果某些实验未运行（例如缺失值或可扩展性），脚本会自动跳过对应图表。
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

# 基线对比的模型列表（可根据需要修改，例如加入 "TabPFN", "TabICL"）
BASELINE_MODELS = ["XGBoost", "LightGBM"]

# 强制使用英文字体（避免中文显示方框）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 英文标签
LABEL_ACC = "Accuracy"
LABEL_DATASET = "Dataset"
LABEL_MISS_PCT = "Missing Ratio (%)"
TITLE_BASE = "Baseline Model Accuracy Comparison"
TITLE_MISS = "Robustness to Missing Values"

# ==================== 加载所有结果 ====================
def load_all_results():
    """递归加载 results/ 下所有符合列要求的 CSV"""
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

# ==================== 基线准确率柱状图 ====================
def draw_baseline_accuracy(df):
    """绘制指定基线模型的准确率柱状图"""
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

# ==================== 缺失值趋势图（每个数据集一张） ====================
def draw_missing_trend_separate(df):
    """为每个数据集单独绘制缺失值准确率折线图"""
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

# ==================== 可扩展性曲线（每个数据集一张） ====================
def draw_scalability(df):
    """为每个数据集绘制可扩展性曲线（样本量 vs 准确率）"""
    scal_df = df[df["model"].str.contains(r"_\d+k$", regex=True, na=False)].copy()
    if scal_df.empty:
        print("No scalability results found, skip")
        return
    scal_df["sample_size"] = scal_df["model"].str.extract(r"_(\d+)k$").astype(int)
    scal_df["base_model"] = scal_df["model"].str.replace(r"_\d+k$", "", regex=True)

    datasets = scal_df["dataset"].unique()
    for ds in datasets:
        sub = scal_df[scal_df["dataset"] == ds].sort_values("sample_size")
        if sub.empty:
            continue
        plt.figure(figsize=(6, 4))
        for bm in sub["base_model"].unique():
            bm_sub = sub[sub["base_model"] == bm].sort_values("sample_size")
            plt.plot(bm_sub["sample_size"], bm_sub["accuracy"],
                     marker='s', label=bm, linewidth=2)
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