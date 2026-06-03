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
5. 验证：`pit list`
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
- 所有实验使用固定随机种子42
- 数据集划分7:3
- 结果保存为CSV格式，列名统一：dataset, model, accuracy, f1, auc, train_time, infer_time, memory_usage

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
    ├── 09_scalability_dataset.py  # 可扩展性脚本
    ├── 10_scalability_baseline.py #
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
- TabICL v2 实验：保存到 `results/tabic/`
- 缺失值实验：统一保存到 `results/baseline_missing/`（或其他子目录，脚本会递归查找）

### 2. CSV 文件必须包含的列（顺序不限）
```csv
dataset,model,accuracy,f1,auc,train_time,infer_time_ms,peak_memory_mb
```