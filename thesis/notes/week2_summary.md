# Week 2 Summary

## 本周完成内容

- 将 HCT-BGC 主干补齐为 `HCT-BGC-v1`，包括：
  - learnable positional embedding
  - Bi-CTA 双向跨模态交互
  - gated fusion
  - 统一的 `forward()` 输出接口
- 接入 contrastive alignment loss（InfoNCE 风格）。
- 将训练流程改为更规范的 `train / val / test` 协议。
- 修复验证集空间泄漏问题：采用空间隔离的 `train/val` 划分，并加入 buffer。
- 完成关键消融实验：
  - Baseline-CNN
  - HCT-BGC-v1
  - HCT-BGC-v1 + Contrastive
  - `fusion_layers = 1 / 2 / 3`
  - `with gate / without gate`
- 补齐论文素材导出脚本：
  - classification map
  - gate 统计
  - training curves
  - network diagram

## 当前最佳结果

主结果配置：`run_hct_bgc_v1_main`

```bash
python3 train.py \
  --model hct_bgc \
  --fusion-layers 1 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 0 \
  --output-dir results/run_hct_bgc_v1_main
```

最佳结果：

- Best epoch: `13`
- Val OA / AA / Kappa: `0.9170 / 0.9320 / 0.9093`
- Test OA / AA / Kappa: `0.8596 / 0.8717 / 0.8480`

对比 baseline：

- Baseline-CNN Test OA / AA / Kappa: `0.8434 / 0.8726 / 0.8303`
- HCT-BGC-v1 相对 baseline：
  - OA `+0.0162`
  - Kappa `+0.0177`

## 消融结论

| 方法 | Test OA | Test AA | Test Kappa | 结论 |
| --- | ---: | ---: | ---: | --- |
| Baseline-CNN | 0.8434 | 0.8726 | 0.8303 | 作为对照基线 |
| HCT-BGC-v1 | 0.8596 | 0.8717 | 0.8480 | 当前最佳主干配置 |
| HCT-BGC-v1 + Contrastive | 0.8412 | 0.8631 | 0.8281 | 当前版本未带来泛化增益 |
| HCT-BGC-v1, `fusion_layers=2` | 0.8480 | 0.8715 | 0.8356 | 继续堆叠效果下降 |
| HCT-BGC-v1, `fusion_layers=3` | 0.8283 | 0.8510 | 0.8145 | 更深堆叠明显退化 |
| HCT-BGC-v1, `gate off` | 0.8450 | 0.8713 | 0.8323 | gate 有实际贡献 |

本周可确认结论：

- `HCT-BGC-v1` 是有效的主干改进。
- `fusion_layers=1` 最优，更多融合层没有带来收益。
- `gate` 是有贡献的。
- 当前 contrastive 模块仍不稳定，暂不作为主结果。

## 当前存在的问题

- Apple Silicon 环境下 MPS 对部分模块不稳定，训练常回退到 CPU。
- Contrastive Alignment Loss 还没有转化成稳定的测试集提升。
- 目前还缺参数敏感性实验与更系统的实验表格整理。

## 下周实验计划

- 参数敏感性：
  - `contrastive_weight`
  - `temperature`
  - `dropout`
  - `embed_dim`
- 不同 PCA 维度：
  - `20 / 30 / 40 / 50`
- 不同 patch size：
  - `9 / 11 / 13 / 15`
- 保持 `fusion_layers=1` 作为主结果，不再优先扩展到更深层。
- 补最终 per-class accuracy 表格与论文实验小节正文。
