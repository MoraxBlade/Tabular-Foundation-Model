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

## 文件结构
### code文件夹说明
#### utils 通用工具
- `utils.py`：通用工具函数，包含数据加载函数（`load_dataset`、`load_missing_dataset`、`load_scalability_dataset`）、评估与保存函数（`evaluate_model`、`save_result`）、常量（如 `SEED`）与辅助工具（内存/时间测量、日志）。

#### 00 汇总和图表输出
- `00_summary_results.py`：汇总各类实验结果并生成 `results/summary.csv`，按需汇总子目录下的 CSV，导出对比图表到 `report/figures/`。

#### 01 数据预处理
- `01_data_preprocess.py`：原始数据预处理主脚本，负责清洗、训练/测试划分（70/30）、类别编码、数值标准化，并输出到 `data/processed/` 和 `data/tfm/`。
- `01_table_preprocess.py`：表格数据统一预处理辅助脚本，将不同数据集转换为表格基础模型（TFM）要求的统一格式（例如列顺序、元数据、CSV 格式）。
- `01_missing_value.py`：生成含缺失值的实验数据集（按 0%,10%,20%,30%），将生成文件保存到 `data/missing/`，文件名遵循 `<dataset>_{train|test}_{N}missing.csv`。
- `01_scalability_dataset.py`：为可扩展性实验生成不同样本量的子集并保存到 `data/scalability/`（如 1000、10000、100000 等）。

#### 02 基线模型与实验
- `02_baseline_xgboost.py`：XGBoost 基线训练与评估脚本，使用 `utils` 接口加载数据与评估，结果追加保存到 `results/baseline/xgboost_results.csv`。
- `02_baseline_lightgbm.py`：LightGBM 基线训练与评估脚本，保存结果到 `results/baseline/lightgbm_results.csv`。
- `02_baseline_missing_experiment.py`：在含缺失数据集上的基线鲁棒性实验，批量运行不同缺失比例的数据并保存结果到 `results/baseline_missing/`。
- `02_baseline_scalability.py`：可扩展性基线实验，测试不同训练样本规模下模型表现并保存到 `results/baseline_scalability/`。

#### 03 TabICL
- `03_tabicl_v2.py`：TabICL / TabPFN 等表格基础模型实验脚本，封装与第三方库（或 API）的调用，统一使用 `utils` 的加载与评估接口，结果保存到 `results/tabicl/` 或 `results/tabpfn/`
#### 04 TabPFN





### 文件树
```bash
Tabular-Foundation-Model/
├── README.md                  # 项目说明文档（本文件）
├── requirements.txt           # Python 依赖清单，用于 `pip install -r requirements.txt`
├── download_datasets.py       # 批量下载/更新数据集的脚本（走公开 API）
├── project2机翻中文.pdf       # 项目相关参考资料/翻译文档
├── code/                      # 所有实验脚本与通用工具代码
│   ├── 00_summary_results.py  # 汇总各类实验结果并生成 `results/summary.csv`
│   ├── 01_data_preprocess.py  # 原始数据预处理：清洗、划分、编码、标准化
│   ├── 01_table_preprocess.py # 表格数据统一预处理辅助脚本
│   ├── 01_missing_value.py    # 生成缺失值版本数据集
│   ├── 01_scalability_dataset.py # 生成可扩展性实验所需的样本子集
│   ├── 02_baseline_xgboost.py # XGBoost 基线训练与评估
│   ├── 02_baseline_lightgbm.py # LightGBM 基线训练与评估
│   ├── 02_baseline_missing_experiment.py # 缺失值基线实验脚本
│   ├── 02_baseline_scalability.py # 可扩展性基线实验脚本
│   ├── 03_tabicl_v2.py        # TabICL / TabPFN API 相关实验脚本
│   └── utils.py               # 通用工具：数据加载、评估指标、结果保存
├── data/                      # 数据文件（原始 / 预处理 / 含缺失 / 可扩展性）
│   ├── raw/                   # 原始数据文件
│   │   ├── adult.csv
│   │   ├── covtype_raw.csv
│   │   ├── credit-g.csv
│   │   └── ...
│   ├── processed/             # 预处理后用于训练/测试的统一 CSV
│   │   ├── adult_train.csv
│   │   ├── adult_test.csv
│   │   ├── credit-g_train.csv
│   │   ├── credit-g_test.csv
│   │   ├── covtype_train.csv
│   │   ├── covtype_test.csv
│   │   └── tfm/               # 面向表格基础模型的统一格式数据
│   ├── missing/               # 按缺失比例生成的实验数据集
│   │   ├── adult_train_0missing.csv
│   │   ├── adult_train_10missing.csv
│   │   ├── ...                # 命名模式：<dataset>_{train|test}_{N}missing.csv
│   └── scalability/           # 不同样本量的可扩展性实验数据
│       ├── adult_train_1000samples.csv
│       ├── adult_train_10000samples.csv
│       ├── covtype_train_1000samples.csv
│       ├── covtype_train_10000samples.csv
│       ├── covtype_train_100000samples.csv
│       └── credit-g_train_700samples.csv
├── results/                   # 所有实验结果（CSV 格式，按子目录区分）
│   ├── baseline/              # 基线模型结果（XGBoost / LightGBM）
│   │   ├── xgboost_results.csv
│   │   └── lightgbm_results.csv
│   ├── baseline_missing/      # 缺失值实验结果
│   │   ├── xgboost_missing.csv
│   │   └── lightgbm_missing.csv
│   ├── baseline_scalability/  # 可扩展性基线实验结果
│   │   ├── xgboost_scalability.csv
│   │   └── lightgbm_scalability.csv
│   ├── tabpfn/                # TabPFN 结果输出目录
│   │   ├── adult_tabpfn_result.csv
│   │   ├── covtype_tabpfn_result.csv
│   │   └── credit-g_tabpfn_result.csv
│   ├── tabicl/                # TabICL 结果输出目录
│   ├── test/                  # 临时/测试结果（例如 quick-check）
│   │   └── test_result.csv
│   └── summary.csv            # 汇总表：合并各实验结果供 `00_summary_results.py` 使用
├── report/                    # 报告与答辩材料（草稿、图表、最终稿）
│   ├── drafts/                # 各成员草稿与阶段记录
│   │   ├── 1_baseline.md
│   │   └── daily_update.md
│   ├── figures/               # 导出的图表（PNG）
│   │   ├── base_accuracy.png
│   │   ├── missing_trend_adult.png
│   │   ├── missing_trend_credit-g.png
│   │   ├── missing_trend_covtype.png
│   │   ├── scalability_adult.png
│   │   ├── scalability_credit-g.png
│   │   └── scalability_covtype.png
│   └── final/                 # 最终整合材料（主文档、PDF、PPT）
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
所有数据加载、评估、结果保存必须调用 `utils.py` 中已实现的函数
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

> 注意：可扩展性实验的模型名可用 `_数字samples` 后缀也用 `_数字k`，因为 `08_summary_results.py` 已同时支持两种格式。

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
- 模型参数：全部使用官方推荐默认参数
- 设备：统一使用 `device="cpu"`（确保内存监控稳定）
