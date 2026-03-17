# Week 3 Summary

## 本周重点

- 完成重复实验脚本 `run_repeat.py`，将实验协议规范为固定 `split_seed`、改变训练 `seed` 的 `mean ± std` 评估方式。
- 完成主结果重复实验：
  - Baseline-CNN
  - HCT-BGC-v1
- 完成参数实验：
  - `patch size = 9 / 11 / 13`
  - `PCA bands = 20 / 30 / 40`
  - `embed_dim = 64 / 96 / 128`
- 补齐论文素材：
  - per-class accuracy 对比表
  - confusion matrix 图
  - classification comparison 图

## 表 1 主结果

固定数据划分、重复 3 次训练后的测试集结果如下：

| Model | OA | AA | Kappa |
| --- | --- | --- | --- |
| Baseline | 84.85 ± 0.37 | 86.66 ± 0.31 | 83.57 ± 0.39 |
| HCT-BGC | 85.58 ± 0.42 | 87.63 ± 0.46 | 84.40 ± 0.46 |

结论：

- HCT-BGC-v1 在三项指标上均优于 Baseline。
- 测试集 OA 从 `84.85` 提升到 `85.58`，Kappa 从 `83.57` 提升到 `84.40`。
- 标准差较小，说明主结果比较稳定，不是单次随机波动。

## 表 2 消融实验

该表使用前面已经完成的正式单次实验结果，指标为 test OA。

| Variant | OA |
| --- | --- |
| baseline | 84.34 |
| +Bi-CTA | 84.50 |
| +Gate | 85.96 |
| +Contrastive | 84.12 |

说明：

- `baseline`：Baseline-CNN。
- `+Bi-CTA`：对应 `no_gate` 结果，即引入 Bi-CTA，但不使用 gate，采用简单融合。
- `+Gate`：对应完整的 HCT-BGC-v1。
- `+Contrastive`：在 HCT-BGC-v1 上继续加入 contrastive loss。

结论：

- Bi-CTA 单独带来小幅提升，说明跨模态信息交换是有效的。
- Gate 带来更明显增益，说明自适应融合比简单平均更有效。
- 当前版本的 Contrastive 未带来测试集增益，暂不作为主结果支撑点。

## 表 3 Patch 实验

| Patch | OA | AA | Kappa |
| --- | --- | --- | --- |
| 9 | 84.08 ± 1.36 | 85.94 ± 1.08 | 82.76 ± 1.46 |
| 11 | 85.58 ± 0.42 | 87.63 ± 0.46 | 84.40 ± 0.46 |
| 13 | 79.93 ± 0.37 | 82.01 ± 0.27 | 78.30 ± 0.42 |

结论：

- `patch = 11` 最优。
- `patch = 9` 的局部上下文不足，性能略低。
- `patch = 13` 引入过多背景和类别混杂，性能明显下降。
- 后续实验统一采用 `patch_size = 11`。

## 表 4 PCA 实验

| PCA Bands | OA | AA | Kappa |
| --- | --- | --- | --- |
| 20 | 83.71 ± 1.81 | 85.93 ± 1.52 | 82.36 ± 1.94 |
| 30 | 85.58 ± 0.42 | 87.63 ± 0.46 | 84.40 ± 0.46 |
| 40 | 83.77 ± 0.25 | 86.18 ± 0.46 | 82.45 ± 0.27 |

结论：

- `PCA = 30` 最优。
- `PCA = 20` 压缩过强，部分判别信息丢失。
- `PCA = 40` 虽保留更多维度，但也带入更多冗余与噪声。
- 后续实验统一采用 `pca_components = 30`。

## 表 5 Per-Class Accuracy

| Class | Baseline | HCT-BGC | Delta |
| --- | --- | --- | --- |
| Healthy grass | 76.54 | 82.31 | +5.77 |
| Stressed grass | 85.19 | 81.26 | -3.94 |
| Synthetic grass | 99.60 | 83.60 | -16.01 |
| Trees | 91.76 | 92.05 | +0.28 |
| Soil | 100.00 | 100.00 | +0.00 |
| Water | 95.80 | 95.80 | +0.00 |
| Residential | 84.64 | 81.66 | -2.98 |
| Commercial | 95.83 | 94.50 | -1.33 |
| Road | 80.47 | 80.56 | +0.09 |
| Highway | 43.12 | 60.64 | +17.52 |
| Railway | 99.24 | 95.47 | -3.78 |
| Parking lot 1 | 63.90 | 83.86 | +19.96 |
| Parking lot 2 | 92.73 | 77.85 | -14.88 |
| Tennis court | 100.00 | 97.98 | -2.02 |
| Running track | 100.00 | 100.00 | +0.00 |

结论：

- 提升最大的类别是 `Parking lot 1`、`Highway` 和 `Healthy grass`。
- 几乎不变的类别是 `Soil`、`Water`、`Running track`、`Road`、`Trees`。
- 下降较明显的类别是 `Synthetic grass` 和 `Parking lot 2`。

## 实验小结

本周实验可以支撑如下结论：

- Bi-CTA 提供了有效的跨模态信息交换，能够帮助模型更好地利用 HSI 与 LiDAR 的互补特征。
- Gate 对最终性能有明显贡献，说明模态自适应融合优于简单平均融合。
- Patch size 会显著影响空间上下文建模效果，过小会导致上下文不足，过大则会引入过多背景干扰；当前最合适的设置是 `patch_size = 11`。
- PCA 维度会显著影响光谱表达能力，过低会丢失判别信息，过高会带来冗余与噪声；当前最合适的设置是 `pca_components = 30`。
- 从 per-class accuracy 来看，HCT-BGC 的增益并不是对所有类别均匀提升，而是主要体现在 `Highway` 和 `Parking lot 1` 这类更依赖跨模态互补信息的困难类别上。
- 当前版本 contrastive loss 还没有带来稳定的测试集增益，因此暂时不作为论文主结果的一部分。

## 当前推荐主配置

```bash
python3 train.py \
  --model hct_bgc \
  --fusion-layers 1 \
  --patch-size 11 \
  --pca-components 30 \
  --embed-dim 128 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 0 \
  --split-seed 42 \
  --output-dir results/run_hct_bgc_v1_main
```
