# HCT-BGC 方法与代码对应表

| 论文方法模块 | 代码模块 | 作用说明 |
| --- | --- | --- |
| HSI CNN 编码器 | `models/hct_backbone.py::HsiCnnEncoder` | 从 HSI patch 中提取局部光谱-空间特征图。 |
| LiDAR CNN 编码器 | `models/hct_backbone.py::LidarCnnEncoder` | 从 LiDAR patch 中提取局部高程/结构特征图。 |
| Token 化 + CLS token | `models/hct_backbone.py::Tokenizer` | 将特征图投影为 token 序列，并在开头拼接可学习的 CLS token。 |
| HSI Transformer Encoder | `models/hct_bgc.py::HCT_BGC.h_te` | 建模 HSI token 的模态内长程依赖。 |
| LiDAR Transformer Encoder | `models/hct_bgc.py::HCT_BGC.l_te` | 建模 LiDAR token 的模态内长程依赖。 |
| Bi-CTA | `models/fusion_blocks.py::BiDirectionalClassTokenAttention` | 让每个模态的 CLS token 查询另一模态 token 序列，实现双向跨模态信息交互。 |
| Gated Fuse | `models/fusion_blocks.py::GatedCrossModalFusion` | 学习逐维门控权重，将 HSI 与 LiDAR 的 CLS token 融合为统一表征。 |
| 分类头 | `models/hct_bgc.py::HCT_BGC.classifier` | 基于融合 token 生成最终分类 logits。 |
| 可用于对比损失的表征输出 | `models/hct_bgc.py::HCT_BGC.forward` 输出 `h_cls`、`l_cls`、`fused_token` | 复用分支表征和融合表征，支持对比学习或表征对齐损失。 |

## Forward 输出说明

`HCT_BGC.forward(hsi, lidar)` 现在返回：

| 输出键 | 含义 | 典型用途 |
| --- | --- | --- |
| `logits` | 分类头输出的最终类别分数。 | 交叉熵分类训练。 |
| `h_cls` | HSI 分支经过 Transformer 编码和 Bi-CTA 更新后的 CLS token。 | HSI 侧对比损失或对齐损失。 |
| `l_cls` | LiDAR 分支经过 Transformer 编码和 Bi-CTA 更新后的 CLS token。 | LiDAR 侧对比损失或对齐损失。 |
| `fused_token` | 由 `h_cls` 与 `l_cls` 经过门控融合得到的联合表征。 | 联合表征学习与最终分类。 |
