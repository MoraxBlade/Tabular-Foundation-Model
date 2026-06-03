# code/08_summary_results.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# 全局配置
RESULT_ROOT = "results"
SAVE_SUMMARY = "results/summary.csv"
FIG_SAVE = "report/figures"
os.makedirs(FIG_SAVE, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

def load_all_results():
    """递归加载所有结果 CSV，统一列名并清洗 model 字段"""
    all_dfs = []
    required_cols = {"dataset", "model", "accuracy", "f1", "auc", "train_time", "infer_time_ms", "peak_memory_mb"}

    for csv_path in glob(f"{RESULT_ROOT}/**/*.csv", recursive=True):
        try:
            df = pd.read_csv(csv_path)
            if not required_cols.issubset(df.columns):
                print(f"跳过 {csv_path}：缺少必要列")
                continue
            # 统一 model 命名：去除可能的后缀空格，且将 "XGBoost_0missing" 保留为原始名
            df["model"] = df["model"].astype(str).str.strip()
            all_dfs.append(df)
            print(f"加载成功：{csv_path}")
        except Exception as e:
            print(f"读取失败 {csv_path}: {e}")

    if not all_dfs:
        return None
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["dataset", "model"]).reset_index(drop=True)
    return full

def draw_baseline_accuracy(df):
    """原始数据集（无缺失）准确率柱状图"""
    base = df[~df["model"].str.contains("missing", case=False, na=False)].copy()
    if base.empty:
        return
    plt.figure(figsize=(11, 5))
    sns.barplot(data=base, x="dataset", y="accuracy", hue="model", palette="Blues_d")
    plt.title("各模型在原始数据集上的准确率对比", fontsize=14)
    plt.xlabel("数据集")
    plt.ylabel("准确率")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE, "base_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()

def draw_missing_trend(df):
    """改进的缺失值准确率折线图（强制数值 x 轴，保证连线）"""
    miss_df = df[df["model"].str.contains("missing", case=False, na=False)].copy()
    if miss_df.empty:
        print("警告：没有缺失值实验结果，跳过缺失趋势图")
        return

    # 提取缺失百分比数值（如 "XGBoost_10missing" -> 10）
    def extract_miss_percent(s):
        import re
        match = re.search(r"(\d+)missing", s)
        return int(match.group(1)) if match else -1
    miss_df["miss_percent"] = miss_df["model"].apply(extract_miss_percent)
    # 去掉无法解析的行
    miss_df = miss_df[miss_df["miss_percent"] >= 0]

    # 同时提取基础模型名（去掉 _数字missing 后缀）用于 hue 分组
    miss_df["base_model"] = miss_df["model"].str.replace(r"_\d+missing", "", regex=True)

    datasets = sorted(miss_df["dataset"].unique())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = miss_df[miss_df["dataset"] == ds].copy()
        # 每个模型单独画线，确保 x 为数值，marker='o' 连线
        for model_name in sub["base_model"].unique():
            model_sub = sub[sub["base_model"] == model_name].sort_values("miss_percent")
            if len(model_sub) < 2:
                print(f"警告：{ds} 中 {model_name} 缺失比例数据点不足2，无法连线")
                ax.plot(model_sub["miss_percent"], model_sub["accuracy"],
                        marker='o', label=model_name, linewidth=2)
            else:
                ax.plot(model_sub["miss_percent"], model_sub["accuracy"],
                        marker='o', label=model_name, linewidth=2)
        ax.set_title(f"{ds} 缺失值鲁棒性")
        ax.set_xlabel("缺失百分比 (%)")
        ax.grid(True, alpha=0.3)
        if ax != axes[0]:
            ax.get_legend().remove()
    axes[0].set_ylabel("准确率")
    axes[0].legend(title="模型", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE, "missing_trend.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("缺失值趋势图已保存")

def draw_scalability(df):
    """可扩展性曲线（需要结果文件中存在 sample_size 列或从 model 名解析）"""
    # 这里假设可扩展性实验的结果中 model 列包含类似 "XGBoost_1k" 的字样
    scal_df = df[df["model"].str.contains(r"_\d+k$", regex=True, na=False)].copy()
    if scal_df.empty:
        print("未找到可扩展性实验结果，跳过该图")
        return
    # 提取样本量（1k, 10k, 100k -> 数值）
    scal_df["sample_size"] = scal_df["model"].str.extract(r"_(\d+)k$").astype(int)
    scal_df["base_model"] = scal_df["model"].str.replace(r"_\d+k$", "", regex=True)

    datasets = sorted(scal_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = scal_df[scal_df["dataset"] == ds].sort_values("sample_size")
        for bm in sub["base_model"].unique():
            bm_sub = sub[sub["base_model"] == bm].sort_values("sample_size")
            ax.plot(bm_sub["sample_size"], bm_sub["accuracy"], marker='s', label=bm, linewidth=2)
        ax.set_title(f"{ds} 可扩展性")
        ax.set_xlabel("样本量 (千)")
        ax.set_xscale("log")  # 样本量跨度大，使用对数坐标
        ax.grid(True, alpha=0.3)
        if ax != axes[0]:
            ax.get_legend().remove()
    axes[0].set_ylabel("准确率")
    axes[0].legend(title="模型")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_SAVE, "scalability.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("可扩展性图已保存")

if __name__ == "__main__":
    data = load_all_results()
    if data is None:
        print("无有效结果数据，请先运行实验脚本。")
    else:
        data.to_csv(SAVE_SUMMARY, index=False, encoding="utf-8")
        print(f"汇总表已保存至 {SAVE_SUMMARY}，共 {len(data)} 条记录")
        draw_baseline_accuracy(data)
        draw_missing_trend(data)
        draw_scalability(data)
        print(f"所有图表已保存至 {FIG_SAVE}")