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