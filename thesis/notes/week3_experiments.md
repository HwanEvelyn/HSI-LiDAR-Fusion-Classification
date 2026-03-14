# Week 3 Experiment Checklist

## 主线目标

- 固化 `HCT-BGC-v1` 主结果
- 补参数敏感性实验
- 完成论文 Experiment 部分需要的表格和图

## 待做实验

### 1. 参数敏感性

- `contrastive_weight`: `0.02 / 0.05 / 0.1`
- `temperature`: `0.07 / 0.1 / 0.2`
- `dropout`: `0.0 / 0.1 / 0.2`
- `embed_dim`: `64 / 128 / 256`

### 2. PCA 维度

- `pca_components = 20 / 30 / 40 / 50`

### 3. Patch Size

- `patch_size = 9 / 11 / 13 / 15`

### 4. 结果整理

- 导出主结果的 per-class accuracy
- 按论文格式整理 OA / AA / Kappa 表
- 生成最终 classification map 与 training curves
- 补图注和实验设置描述

## 当前不优先项

- `fusion_layers = 2 / 3`
  原因：已完成消融，结果劣于 `fusion_layers = 1`
- 继续扩展更深 Bi-CTA
  原因：当前证据不足，优先先把主线实验做完整
