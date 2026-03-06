from __future__ import annotations

from typing import Dict

import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    统计”真实类别“和”预测类别“的对应关系，生成混淆矩阵。
    
    假设有 3 类，那么混淆矩阵就是一个 3 x 3 的表：
    行表示真实类别
    列表示预测类别
    比如 cm[2,1] = 5 表示：
    真实是第 2 类的样本，有 5 个被预测成第 1 类了。
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    for t, p in zip(y_true[valid], y_pred[valid]):
        cm[t, p] += 1
    return cm


def oa_aa_kappa(cm: np.ndarray) -> Dict[str, float]:
    """
    从混淆矩阵中计算总体准确率（OA）、平均准确率（AA）和 Kappa 系数。
    """
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total <= 0:
        return {"oa": 0.0, "aa": 0.0, "kappa": 0.0}

    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    oa = float(diag.sum() / total)    # oa = (TP1 + TP2 + ... + TPn) / total 预测正确的样本数 / 总样本数

    class_acc = np.divide(diag, row_sum, out=np.zeros_like(diag), where=row_sum > 0)
    aa = float(class_acc.mean())     # aa = (acc1 + acc2 + ... + accn) / n  每一类的准确率的平均值

    pe = float((row_sum * col_sum).sum() / (total * total))
    if np.isclose(1.0 - pe, 0.0):
        kappa = 0.0
    else:
        kappa = float((oa - pe) / (1.0 - pe))    # kappa = (oa - pe) / (1 - pe)  OA 减去随机预测的准确率，除以 1 减去随机预测的准确率。考虑了“随机猜中”的影响。

    return {"oa": oa, "aa": aa, "kappa": kappa}
