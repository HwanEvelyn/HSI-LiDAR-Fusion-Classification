# HCT-BGC-v1 结构草图

```text
HSI patch -----------------> HSI CNN Encoder -----------------> Tokenizer + CLS + PosEmbed --+
                                                                                               |
                                                                                               v
                                                                                     HSI Transformer
                                                                                               |
                                                                                               v
                                                                                        HSI token序列

LiDAR patch --------------> LiDAR CNN Encoder ---------------> Tokenizer + CLS + PosEmbed --+
                                                                                              |
                                                                                              v
                                                                                    LiDAR Transformer
                                                                                              |
                                                                                              v
                                                                                       LiDAR token序列

HSI cls  ----查询----> LiDAR 全部 tokens
LiDAR cls ----查询----> HSI 全部 tokens
          \_________________ Bi-CTA × fusion_layers _________________/
                           残差 + FFN

更新后的 h_cls ------------------+
                                |--> Gated Fusion / Average Fusion --> fused_token --> Classifier --> logits
更新后的 l_cls ------------------+
```

## 讲解顺序

1. 先用双分支 CNN 提取 HSI 和 LiDAR 的局部特征图。
2. 每个分支把特征图转成 token 序列，加入 CLS token 和可学习位置编码。
3. 各自经过 Transformer Encoder，先完成模态内建模。
4. 进入 Bi-CTA，使用双向 CLS token 查询对侧全部 tokens，完成跨模态信息交互。
5. 取更新后的 `h_cls` 和 `l_cls` 做融合。
6. 默认用门控融合；消融时可切到简单平均融合。
7. 最终用融合表征 `fused_token` 做分类，同时保留 `h_cls/l_cls/fused_token` 供对比损失使用。
