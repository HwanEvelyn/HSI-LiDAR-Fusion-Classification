# HCT-BGC 最终主方案总结

本文最终采用的主模型为 `HCT-BGC`，面向 HSI-LiDAR 双模态遥感分类任务。模型以高光谱影像提供的光谱-空间判别信息和 LiDAR 提供的高程/结构信息为互补来源，通过轻量异构双分支编码、多尺度 token 表示、CLS 级跨模态交互、局部主导的跨尺度融合以及保守式门控融合，生成最终的联合分类表征。

本文的主实验协议采用多尺度输入设置：局部尺度为 `11 x 11`，上下文尺度为 `17 x 17`，上下文分支在 CNN 编码后通过自适应平均池化对齐到 `11 x 11` 的 token 网格。因此，局部分支强调中心像素邻域的判别信息，上下文分支提供更大感受野的空间补充，同时保持与局部分支一致的 token 数量。模型主配置为 `embed_dim=128`、`num_heads=4`、`num_layers=2`、`fusion_layers=1`、`encoder_variant=light_hetero`、`scale_fusion_mode=residual`，并启用 `use_conservative_fusion` 和 `use_aux_heads`。

## 1. 轻量异构双分支编码

模型首先分别对 HSI 和 LiDAR 输入进行模态专属编码。HSI 分支采用 `light_hetero` 编码器，先使用 `1 x 1` 卷积进行光谱通道混合，再使用 `3 x 3` 卷积提取局部空间纹理。LiDAR 分支同样采用 `light_hetero` 编码器，但其结构以 `3 x 3` 空间卷积为主，并加入轻量残差块，以增强对地物边界、高程变化和空间结构的表达能力。

该设计体现了异构编码思想：HSI 分支更侧重光谱-空间联合表征，LiDAR 分支更侧重空间结构与几何信息提取。两路编码器输出维度统一为 `embed_dim=128`，便于后续 token 化和跨模态交互。

## 2. 多尺度 token 表示

在多尺度分支中，模型从 `17 x 17` 输入 patch 中中心裁剪得到 `11 x 11` 的 local patch，同时保留完整 `17 x 17` patch 作为 context patch。local patch 和 context patch 都会先经过对应模态的 CNN encoder；区别在于 context 分支会在 CNN 特征图层面被自适应平均池化到 `11 x 11`，再进入 tokenizer。

每一路特征图随后通过 `Tokenizer` 转换为 Transformer token 序列。Tokenizer 使用 `1 x 1` 卷积将特征映射到统一嵌入维度，展平空间维度后拼接可学习的 `CLS token`，并加入可学习位置编码。最终每一路 token 序列包含 `1 + 11 x 11 = 122` 个 token，其中 `CLS token` 用于承载该分支的全局摘要表示。

## 3. 模态内 Transformer 编码

HSI 和 LiDAR 分别使用独立的 Transformer Encoder 进行模态内建模。对于 local 和 context 两个尺度，模型都会分别执行：

```text
HSI tokens   -> HSI Transformer Encoder
LiDAR tokens -> LiDAR Transformer Encoder
```

这一阶段的作用是先在每个模态内部建模空间 token 之间的长程依赖，使 `CLS token` 能够聚合本模态的上下文信息。由于 HSI 和 LiDAR 的物理含义不同，模型没有共享两者的 Transformer Encoder，而是保留双分支结构。

## 4. 同尺度跨模态交互：Bi-CTA

模态内编码之后，模型在 local 和 context 两个尺度上分别执行 HSI-LiDAR 跨模态交互。该交互由 `BiDirectionalClassTokenAttention` 完成，其核心思想是只更新两路的 `CLS token`，不对全部 token 进行强耦合混合。

在 local 尺度上：

```text
HSI local CLS   attends to LiDAR local tokens
LiDAR local CLS attends to HSI local tokens
```

在 context 尺度上：

```text
HSI context CLS   attends to LiDAR context tokens
LiDAR context CLS attends to HSI context tokens
```

这种 Bi-CTA 设计使每个模态的摘要 token 能够从另一模态中提取互补信息，同时避免全 token 级跨模态交互带来的计算开销和噪声传播。启用 conservative fusion 时，跨模态注意力更新会被一个可学习残差缩放因子限制，从而避免过度改写原始模态表示。

## 5. 同模态跨尺度交互

完成同尺度跨模态交互后，模型再在每个模态内部执行 local-context 跨尺度交互。具体而言：

```text
HSI local tokens   <-> HSI context tokens
LiDAR local tokens <-> LiDAR context tokens
```

该阶段让 local `CLS token` 从 context tokens 中获取大范围空间补充，同时让 context `CLS token` 受到 local tokens 的中心区域约束。这样，后续尺度融合不再只是简单地合并两个独立尺度，而是在已经发生过跨尺度信息交换的基础上形成更稳定的尺度摘要。

## 6. 局部主导的残差式尺度融合

本文最终采用 `scale_fusion_mode=residual`，对应 `LocalDominantScaleFusion`。该模块以 local `CLS` 为主干，将 context `CLS` 作为残差修正来源：

```text
gate = sigmoid(W [local_cls ; context_cls])
residual = gate * (context_cls - local_cls)
fused_cls = local_cls + strength * residual
```

其中 `gate` 是逐维尺度门控，`strength` 是可学习的上下文注入强度。该设计的核心假设是：中心像素分类主要依赖局部邻域的判别信息，但更大范围上下文能够提供辅助修正。因此，context 不直接替代 local，而是以受控残差形式补充 local 表征。

经过该模块后，模型得到两个最终模态表示：

```text
h_cls: HSI 的尺度融合表示
l_cls: LiDAR 的尺度融合表示
```

## 7. 最终保守式跨模态门控融合

在得到 `h_cls` 和 `l_cls` 后，模型通过 `GatedCrossModalFusion` 生成最终联合表征 `fused_token`。普通门控融合形式为：

```text
gate = sigmoid(W [h_cls ; l_cls])
fused = gate * h_cls + (1 - gate) * l_cls
```

在本文主方案中启用 conservative fusion，因此最终输出不是完全自由的门控结果，而是以平均融合为基线，对学习到的门控融合结果做受限偏移：

```text
base = 0.5 * (h_cls + l_cls)
fused_token = base + strength * (fused - base)
```

该机制的作用是限制跨模态融合的偏移幅度，避免模型在训练样本有限或模态噪声较强时过度依赖某一模态，从而提升训练稳定性和泛化能力。

## 8. 分类头与辅助监督

最终分类头以 `fused_token` 为输入，通过 `LayerNorm + Linear + GELU + Dropout + Linear` 输出主分类 logits：

```text
fused_token -> classifier -> logits
```

此外，主方案启用 auxiliary heads，对 `h_cls` 和 `l_cls` 分别施加辅助分类监督：

```text
h_cls -> HSI aux classifier   -> h_logits
l_cls -> LiDAR aux classifier -> l_logits
```

辅助头不作为最终决策融合使用，而是在训练阶段为单模态分支提供额外监督，使 HSI 与 LiDAR 的最终尺度融合表示都具有独立判别能力，从而稳定整体优化过程。

## 9. 最终流程概括

最终模型流程可以概括为：

```text
HSI/LiDAR 17 x 17 输入
-> local 11 x 11 与 context 17 x 17 多尺度构建
-> HSI/LiDAR light_hetero CNN 编码
-> context 特征池化对齐到 11 x 11
-> Tokenizer: CLS token + spatial tokens + positional embedding
-> HSI/LiDAR 模态内 Transformer Encoder
-> local 尺度 HSI-LiDAR Bi-CTA
-> context 尺度 HSI-LiDAR Bi-CTA
-> HSI 内部 local-context 跨尺度交互
-> LiDAR 内部 local-context 跨尺度交互
-> local-dominant residual scale fusion 得到 h_cls 和 l_cls
-> conservative gated cross-modal fusion 得到 fused_token
-> 主分类头输出 logits
```

## 10. 方法定位

本文主模型属于中后期特征级融合方法。它不是输入层 early fusion，也不是简单的决策级 late fusion；其融合发生在 token/CLS 表征层面，包含中间跨模态交互和最终跨模态门控融合。整体思路是先保留 HSI 与 LiDAR 的模态专属表达，再通过受控的 CLS 级交互和门控机制逐步引入互补信息，最终形成稳定的联合分类表征。
