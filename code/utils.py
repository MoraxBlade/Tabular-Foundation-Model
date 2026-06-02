import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from memory_profiler import memory_usage
import pandas as pd
import os

SEED = 42

def load_dataset(dataset_name):
    """加载预处理后的数据集（增加路径校验和异常捕获）"""
    train_path = f"data/processed/{dataset_name}_train.csv"
    test_path = f"data/processed/{dataset_name}_test.csv"
    
    # 校验文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在：{train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试集文件不存在：{test_path}")
    
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
    except Exception as e:
        raise RuntimeError(f"加载{dataset_name}数据集失败：{str(e)}")
    
    # 分离特征和标签（兼容所有预处理后的数据集）
    X_train = train.drop(train.columns[-1], axis=1)
    y_train = train.iloc[:, -1]
    X_test = test.drop(test.columns[-1], axis=1)
    y_test = test.iloc[:, -1]
    
    # 校验数据完整性
    if X_train.empty or X_test.empty:
        raise ValueError(f"{dataset_name}数据集加载后为空，请检查预处理脚本")
    
    print(f" 成功加载{dataset_name}数据集 | 训练集：{len(X_train)}条 | 测试集：{len(X_test)}条")
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    """统一评估函数（优化内存监控、AUC计算鲁棒性）"""
    try:
        # 记录训练时间和内存（增加timeout避免卡住，interval适配大数据集）
        start_time = time.time()
        # 内存监控：interval=0.5减少日志量，timeout=3600避免无限等待
        mem_usage = memory_usage(
            (model.fit, (X_train, y_train)), 
            interval=0.5, 
            timeout=3600,
            include_children=True  # 包含子进程内存（适配部分模型）
        )
        train_time = time.time() - start_time
        peak_memory = max(mem_usage) if mem_usage else 0.0
        
        # 记录推理时间
        start_time = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - start_time
        single_infer_time = infer_time / len(X_test) * 1000  # 毫秒/样本
        
        # 计算核心指标（处理空预测的情况）
        accuracy = accuracy_score(y_test, y_pred) if len(y_pred) > 0 else 0.0
        # F1：处理单类别情况（避免macro平均报错）
        try:
            f1 = f1_score(y_test, y_pred, average="macro")
        except ValueError:
            f1 = f1_score(y_test, y_pred, average="binary" if len(np.unique(y_test))==2 else "micro")
        
        # 计算AUC（增强鲁棒性）
        auc = -1.0
        try:
            y_pred_proba = model.predict_proba(X_test)
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            elif n_classes > 2:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovo")
        except (AttributeError, ValueError):
            # 模型无predict_proba/标签类别数异常时，AUC记为-1
            auc = -1.0
        
        # 整理结果（统一格式）
        result = {
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
            "train_time": round(train_time, 2),
            "infer_time_ms": round(single_infer_time, 4),
            "peak_memory_mb": round(peak_memory, 2)
        }
        
        # 打印结果（可视化更友好）
        print(f"\n {model_name} - {dataset_name} 评估完成")
        print(f"    指标：准确率={accuracy:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
        print(f"     耗时：训练={train_time:.2f}s | 单样本推理={single_infer_time:.4f}ms")
        print(f"    内存：峰值={peak_memory:.2f}MB")
        
        return result
    
    except Exception as e:
        raise RuntimeError(f"{model_name}评估{dataset_name}失败：{str(e)}")

def save_result(result, save_path):
    """保存结果到CSV（自动创建目录，避免路径不存在报错）"""
    # 自动创建结果目录
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 保存结果（兼容追加/新建）
    df = pd.DataFrame([result])
    try:
        if os.path.exists(save_path):
            df.to_csv(save_path, mode="a", header=False, index=False)
        else:
            df.to_csv(save_path, index=False)
        print(f" 结果已保存到：{save_path}")
    except Exception as e:
        raise RuntimeError(f"保存结果失败：{str(e)}")

# 可选：测试函数（直接运行utils.py可验证基础功能）
if __name__ == "__main__":
    # 测试1：加载数据集（需先完成预处理）
    try:
        X_train, X_test, y_train, y_test = load_dataset("adult")
    except Exception as e:
        print(f"数据集加载测试失败：{e}")
    
    # 测试2：简单模型评估（快速验证评估逻辑）
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=SEED, max_iter=1000)
    try:
        result = evaluate_model(model, X_train, X_test, y_train, y_test, "LogisticRegression", "adult")
        save_result(result, "results/test/test_result.csv")
    except Exception as e:
        print(f"评估/保存测试失败：{e}")