# Tabular-Foundation-Model
表格基础⼤模型属于预训练模型，依托上下⽂学习技术在结构化表格数据上完成分类与回归任务，⽆需针对特 定数据集微调训练，即可实现效果优异的零样本预测。该⽅向由 TabPFN 率先落地，后续⼜衍⽣出 TabICL 、 TabDPT 等改进模型，正快速动摇梯度提升树⻓期以来在各类实测表格基准数据集上的统治地位。
## 快速开始
1. 克隆仓库：`git clone https://github.com/MoraxBlade/Tabular-Foundation-Model.git`
2. 创建虚拟环境：`conda create -n tabfm python=3.10.14`
    - 这里需要下载conda
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
- 成员3：TabICL v2
- 成员5：报告+PPT

## 代码规范
- 所有实验使用固定随机种子42
- 数据集划分7:3
- 结果保存为CSV格式，列名统一：dataset, model, accuracy, f1, auc, train_time, infer_time, memory_usage

## 文件结构
```bash
Tabular-Foundation-Model/
├── code/                     # 所有代码文件
│   ├── 01_data_preprocess.py # 数据集预处理脚本
│   ├── 02_baseline_xgboost.py # XGBoost基线实验脚本
│   ├── 03_baseline_lightgbm.py # LightGBM基线实验脚本
│   ├── 04_tabpfn_v2.py       # TabPFN v2实验脚本（成员2负责）
│   ├── 05_tabicl_v2.py       # TabICL v2实验脚本（成员3负责）
│   ├── 06_missing_value.py   # 缺失值实验数据集生成脚本
│   ├── 07_scalability.py     # 可扩展性实验脚本
│   ├── 08_summary_results.py # 实验结果自动汇总脚本
│   └── utils.py              # 统一实验工具函数（数据加载、评估、保存）
├── data/                     # 数据集目录
│   ├── raw/                  # 原始数据集（仅放下载链接，大文件不上传Git）
│   │   └── download_links.txt
│   ├── processed/            # 预处理后的训练/测试集
│   └── missing/              # 缺失值实验数据集
├── results/                  # 所有实验结果（CSV格式）
│   ├── baseline/             # 基线模型结果
│   ├── tabpfn/               # TabPFN v2结果
│   ├── tabicl/               # TabICL v2结果
│   └── summary.csv           # 所有实验结果汇总表
├── report/                   # 报告与答辩材料
│   ├── drafts/               # 个人初步报告/草稿（按分工提交）
│   │   ├── 1_baseline.md     # 基线实验报告
│   │   ├── 2_tabpfn.md       # TabPFN实验报告
│   │   ├── 3_tabicl.md       # TabICL实验报告
│   │   ├── 4_efficiency.md   # 效率分析报告
│   │   └── daily_update/     # 每日进度更新
│   ├── figures/              # 所有实验图表（统一存放，共享使用）
│   └── final/                # 最终提交材料
│       ├── main.md           # 最终完整报告
│       ├── main.pdf          # 导出的PDF版本
│       └── presentation.pptx # 答辩PPT
├── .gitignore                # Git忽略文件配置
├── requirements.txt          # 依赖库版本清单
└── README.md                 # 项目说明文档
```

## 协作规范
1. 所有代码提交至`code/`目录，遵循现有脚本格式
2. 个人实验报告提交至`report/drafts/`，命名格式：`成员编号_负责模块.md`
3. 所有图表导出为300DPI PNG格式，提交至`report/figures/`
4. 实验结果保存至`results/`对应子目录，使用统一CSV格式
5. 最终报告和PPT由报告负责人统一整合至`report/final/`目录