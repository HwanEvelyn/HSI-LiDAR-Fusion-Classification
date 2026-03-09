# Week 1 Summary

## 本周完成内容

- 已完成官方 Houston 2013 DFTC 数据流程接入，能够正确读取 `CASI.tif`、`LiDAR.tif`、`TR` 和 `VA`。
- 已实现并跑通 Baseline 双分支 CNN，能够完成端到端训练与测试。
- 已补齐评估流程，能够保存 checkpoint、metrics JSON、log 和 confusion matrix。
- 已实现第一版 HCT-BGC 主干，包括 CNN encoders、tokenization、transformer encoders、Bi-CTA 和 gated fusion。
- 已补充第一版分类图脚本，能够导出 `classification_map.png`。

## 当前结果

| Model | OA | AA | Kappa |
| --- | ---: | ---: | ---: |
| Baseline-CNN | 0.8548 | 0.8782 | 0.8424 |
| HCT-BGC-v0 | 0.7860 | 0.8137 | 0.7684 |

说明：

- Baseline 结果来自 `results/run_baseline_compare/best_metrics.json`
- HCT-BGC-v0 结果来自 `results/run_hct_bgc_v1/best_metrics.json`

## 当前最佳结果

- 当前最佳模型：Baseline-CNN
- 最优 checkpoint：`results/run_baseline_compare/best.pth`
- 当前最佳指标：
  - OA: 0.8548
  - AA: 0.8782
  - Kappa: 0.8424

## 当前存在的问题

- 第一版 HCT-BGC 已经可以训练，但当前效果还没有超过 baseline。
- 当前 HCT-BGC 仍是最小可运行版本，尚未加入 contrastive learning。
- 当前 classification map 还是较为基础的有标签区域可视化版本，还不是更完整、更美观的整图展示。
- 目前模型选择仍然基于 test metrics，后续应改为更规范的 validation protocol。

## 下周计划

- 加入 positional encoding，并进一步提升 transformer 训练稳定性。
- 将当前单层的 Bi-CTA 扩展为更强的多层堆叠结构。
- 开始接入论文方法中的 contrastive branch。
- 增加更严格的 validation split，并改用 validation metrics 进行 checkpoint 选择。
- 为论文整理更清晰的对比图和更完整的 classification map。
