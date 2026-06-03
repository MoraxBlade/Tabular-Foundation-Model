# Tabular-Foundation-Model
表格基础⼤模型属于预训练模型，依托上下⽂学习技术在结构化表格数据上完成分类与回归任务，⽆需针对特 定数据集微调训练，即可实现效果优异的零样本预测。该⽅向由 TabPFN 率先落地，后续⼜衍⽣出 TabICL 、 TabDPT 等改进模型，正快速动摇梯度提升树⻓期以来在各类实测表格基准数据集上的统治地位。
## 快速开始
1. 克隆仓库：`git clone https://github.com/MoraxBlade/Tabular-Foundation-Model.git`
2. 创建虚拟环境：`conda create -n tabfm python=3.10.14`
    - 这里需要下载conda
        - 初始化：`conda init powershell`
        - 关掉现在的终端
        - 执行激活命令：`conda activate tabfm`
    - 好处是删掉conda就能把所有为了项目安的东西删掉了
3. 激活环境：`conda activate tabfm`
4. 安装依赖：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
    - 用了镜像源，会快很多
5. 验证：`pip list`
5. 下载数据集：见`download_datasets.py`
    - openML本身太不稳定了，直接api下载
    - 一定要在conda环境里用命令直接运行
6. 运行数据预处理：`python code/01_data_preprocess.py`

## 分工
- chr：（环境+数据+基线+标准化）
- wzx：TabPFN v2
- gxy：TabICL v2
- cy & gcy：报告+PPT

## 代码规范
 - 结果保存为CSV格式，列名统一：dataset, model, accuracy, f1, auc, train_time, infer_time_ms, peak_memory_mb

## 文件结构（已根据当前仓库实际内容更新，括号内为简短注释）
```bash
Tabular-Foundation-Model/
├── 豆包写的逐日实验步骤.md     # 日常实验记录与步骤说明（成员个人笔记）
├── download_datasets.py       # 脚本：批量下载/更新数据集（使用 API）
├── requirements.txt           # Python 依赖清单，用于 `pip install -r requirements.txt`
├── README.md                  # 项目说明文档（本文件）
├── code/                      # 所有实验脚本与工具代码
│   ├── 01_data_preprocess.py  # 数据预处理：划分、编码、标准化
│   ├── 02_baseline_xgboost.py # XGBoost 基线训练与评估
│   ├── 03_baseline_lightgbm.py# LightGBM 基线训练与评估
│   ├── 04_tabpfn_v2.py        # TabPFN v2 实验脚本（负责人：wzx）
│   ├── 05_tabicl_v2.py        # TabICL v2 实验脚本（负责人：gxy）
│   ├── 06_missing_value.py    # 生成缺失值数据集的脚本（用于缺失值实验）
│   ├── 07_baseline_missing_experiment.py # 缺失值基线实验（命名与用途示例）
│   ├── 08_summary_results.py  # 汇总并生成 `results/summary.csv` 的脚本
│   ├── 09_scalability_dataset.py  # 可扩展性数据准备脚本（生成样本量列表）
│   ├── 10_scalability_baseline.py # 可扩展性基线实验脚本（性能对比）
│   └── utils.py               # 通用工具（数据加载、评估指标、结果保存）
├── data/                      # 保存数据集（原始/预处理/含缺失版本）
│   ├── raw/                   # 原始数据或下载链接（大文件建议不直接入仓）
│   │   ├── adult.csv
│   │   ├── covtype.csv
│   │   ├── credit-g.csv
│   │   └── download_links.txt # 原始数据下载链接或说明
│   ├── processed/             # 预处理后用于训练/测试的 CSV（统一格式）
│   │   ├── adult_train.csv
│   │   ├── adult_test.csv
│   │   └── ...
│   └── missing/               # 按缺失比例生成的实验数据集（用于缺失值实验）
│       ├── adult_train_0missing.csv
│       ├── adult_train_10missing.csv
│       ├── ...                # 命名模式：<dataset>_{train|test}_{N}missing.csv
├── results/                   # 所有实验结果（CSV 格式，按子目录区分）
│   ├── baseline/              # 基线模型结果（XGBoost / LightGBM）
│   │   ├── xgboost_results.csv
│   │   └── lightgbm_results.csv
│   ├── baseline_missing/      # 缺失值实验结果（脚本会递归查找）
│   │   ├── xgboost_missing.csv
│   │   └── lightgbm_missing.csv
│   ├── tabpfn/                # TabPFN 实验结果
│   ├── tabicl/                # TabICL 实验结果
│   ├── test/                  # 临时/测试结果（例如 quick-check）
│   └── summary.csv            # 汇总表：合并各实验结果供 `08_summary_results.py` 使用
├── report/                    # 报告与答辩材料（草稿、图表、最终稿）
│   ├── drafts/                # 各成员草稿与分工提交（按文件命名规则）
│   │   ├── 1_baseline.md
│   │   ├── daily_update.md    # 同步更新
│   │   └── ...
│   ├── figures/               # 导出的图表（统一格式、300 DPI PNG）
│   └── final/                 # 最终整合材料（main.md / main.pdf / PPT）
│       ├── main.md
│       ├── main.pdf
│       └── presentation.pptx
└── .gitignore                 # Git 忽略规则（不跟踪大数据 / 虚拟环境等）
```

## 协作规范
1. 所有代码提交至`code/`目录，遵循现有脚本格式
2. 个人实验报告提交至`report/drafts/`，命名格式：`成员编号_负责模块.md`
3. 所有图表导出为300DPI PNG格式，提交至`report/figures/`
4. 实验结果保存至`results/`对应子目录，使用统一CSV格式
5. 最终报告和PPT由报告负责人统一整合至`report/final/`目录

## 实验结果文件命名与保存规范（主要是汇总和图标生成方面）

为保证 `08_summary_results.py` 能够正确识别并汇总所有实验数据，请各成员严格遵守以下命名规范：

### 1. 结果文件保存位置
- 基线模型（XGBoost、LightGBM）：保存到 `results/baseline/`
- TabPFN v2 实验：保存到 `results/tabpfn/`
- TabICL v2 实验：保存到 `results/tabicl/`
- 缺失值实验：统一保存到 `results/baseline_missing/`（或其他子目录，脚本会递归查找）

### 2. CSV 文件必须包含的列（顺序不限）
```csv
dataset,model,accuracy,f1,auc,train_time,infer_time_ms,peak_memory_mb
```

## SOTA 模型（TabPFN / TabICL）实验统一要求

为确保所有模型结果可公平对比并自动被 `08_summary_results.py` 汇总，请 TabPFN v2 和 TabICL v2 的负责人严格按照以下规范编写实验脚本。

### 1. 必须使用的统一工具函数
所有数据加载、评估、结果保存必须调用 `utils.py` 中已实现的函数，**禁止重新实现**：
- 数据加载：
  - 基础实验：`load_dataset(dataset_name)` → 返回 `X_train, X_test, y_train, y_test`
  - 缺失值实验：`load_missing_dataset(dataset_name, ratio_str)`  
    `ratio_str` 格式为 `"0missing"`, `"10missing"`, `"20missing"`, `"30missing"`
  - 可扩展性实验：`load_scalability_dataset(dataset_name, sample_size)`  
    `sample_size` 为整数（1000, 10000, 100000, 700 等）
- 评估与保存：
  - `evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name)`  
    返回包含 accuracy, f1, auc, train_time, infer_time_ms, peak_memory_mb 的字典
  - `save_result(result_dict, save_path)` 追加保存为 CSV

### 2. 模型命名规范（与汇总脚本完全匹配）

| 实验类型 | 模型名格式 | 示例 | 说明 |
|---------|-----------|------|------|
| 基础实验 | `"TabPFN"` 或 `"TabICL"` | `model_name="TabPFN"` | 不含任何后缀 |
| 缺失值实验 | `"{模型名}_{X}missing"` | `"TabPFN_10missing"`, `"TabICL_30missing"` | X 为缺失百分比（0,10,20,30） |
| 可扩展性实验 | `"{模型名}_{size}samples"` | `"TabPFN_10000samples"`, `"TabICL_700samples"` | size 为样本量整数 |

> ⚠️ 注意：可扩展性实验的模型名必须使用 `_数字samples` 后缀（不要用 `_数字k`），因为 `08_summary_results.py` 已同时支持两种格式。

### 3. 结果保存路径
- TabPFN v2 的所有结果统一保存至 `results/tabpfn/` 目录
- TabICL v2 的所有结果统一保存至 `results/tabicl/` 目录
- 建议每个实验类型单独一个 CSV 文件，例如：
  - `results/tabpfn/tabpfn_basic.csv`
  - `results/tabpfn/tabpfn_missing.csv`
  - `results/tabpfn/tabpfn_scalability.csv`

### 4. 必须完成的实验内容

#### 4.1 基础分类实验（必做）
对三个数据集 `["adult", "credit-g", "covtype"]` 分别运行模型，使用 **完整训练集**（`load_dataset`）。

#### 4.2 缺失值鲁棒性实验（必做）
对 `adult`, `credit-g`, `covtype` 三个数据集，分别运行缺失比例 `[0, 0.1, 0.2, 0.3]` 的版本。  
调用 `load_missing_dataset(dataset_name, f"{int(ratio*100)}missing")`。

#### 4.3 可扩展性实验（必做）
针对每个数据集实际可用的样本量（参考 `09_scalability_dataset.py` 生成的列表）运行：

| 数据集 | 可用样本量 |
|--------|-----------|
| adult | 1000, 10000 |
| covtype | 1000, 10000, 100000 |
| credit-g | 700 |

调用 `load_scalability_dataset(dataset_name, sample_size)`。

### 5. 固定配置（与基线严格一致）
- 随机种子：`SEED = 42`（已从 `utils` 导入）
- 数据集划分：训练集 70% / 测试集 30%（预处理时已固定）
- 模型参数：全部使用官方推荐默认参数，**不做超参数调优**
- 设备：统一使用 `device="cpu"`（确保内存监控稳定）

### 6. 脚本结构建议（参考基线脚本）
可参照 `02_baseline_xgboost.py` 和 `10_scalability_baseline.py` 的结构，每个实验类型定义一个函数，在 `if __name__ == "__main__"` 中调用。  
示例框架：
```python
import os
from utils import load_dataset, load_missing_dataset, load_scalability_dataset, evaluate_model, save_result, SEED
from tabpfn import TabPFNClassifier   # 或 from tabicl import TabICLClassifier

MODEL_NAME = "TabPFN"   # 或 "TabICL"
SAVE_DIR = "results/tabpfn"   # 或 "results/tabicl"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_basic():
    for ds in ["adult", "credit-g", "covtype"]:
        X_train, X_test, y_train, y_test = load_dataset(ds)
        model = TabPFNClassifier(device="cpu", version="2.5", n_estimators=100, random_state=SEED)
        result = evaluate_model(model, X_train, X_test, y_train, y_test, MODEL_NAME, ds)
        save_result(result, os.path.join(SAVE_DIR, "basic.csv"))

def run_missing():
    ratios = [0, 0.1, 0.2, 0.3]
    for ds in ["adult", "credit-g", "covtype"]:
        for r in ratios:
            r_str = f"{int(r*100)}missing"
            X_train, X_test, y_train, y_test = load_missing_dataset(ds, r_str)
            model = TabPFNClassifier(device="cpu", version="2.5", n_estimators=100, random_state=SEED)
            result = evaluate_model(model, X_train, X_test, y_train, y_test, f"{MODEL_NAME}_{r_str}", ds)
            save_result(result, os.path.join(SAVE_DIR, "missing.csv"))

def run_scalability():
    sizes_map = {"adult": [1000,10000], "covtype": [1000,10000,100000], "credit-g": [700]}
    for ds, sizes in sizes_map.items():
        for size in sizes:
            X_train, X_test, y_train, y_test = load_scalability_dataset(ds, size)
            model = TabPFNClassifier(device="cpu", version="2.5", n_estimators=100, random_state=SEED)
            result = evaluate_model(model, X_train, X_test, y_train, y_test, f"{MODEL_NAME}_{size}samples", ds)
            save_result(result, os.path.join(SAVE_DIR, "scalability.csv"))

if __name__ == "__main__":
    run_basic()
    run_missing()
    run_scalability()