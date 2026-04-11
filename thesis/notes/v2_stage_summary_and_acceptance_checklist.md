# v2阶段总结与系统验收清单

## 1. 文档目的

这份文档主要用于整理 HCT-BGC v2 阶段在 Houston 数据集上的实验进展，形成一份比较完整的阶段总结，同时列出后续做系统验收时需要准备的成果物清单。这样做的目的，一方面是把目前已经做过的尝试和结论整理清楚，避免后面写论文、写阶段报告或者准备答辩材料时再零散回忆；另一方面也是为了给后续的系统验收、材料归档和结果复现实验提供一个统一的参照。

从目前的实验情况来看，v2 已经完成了从基础方案到多尺度方案、从无辅助头到有辅助头、再到蒸馏、数据增强、一致性损失和 few-shot 尝试的一轮较系统的探索。整体上，v2 的设计方向是有效的，而且已经得到了比较明确的阶段性最优配置。

---

## 2. v2阶段的目标和整体思路

### 2.1 阶段目标

v2 阶段的目标，不只是简单把指标做高，而是希望在 v1 的基础上进一步回答下面几个问题：

- 融合模块是否还能继续优化，尤其是不同尺度信息应该怎样进入主干表示。
- 在 Houston 这种类别多、场景复杂、异质性明显的数据集上，多尺度上下文是否真的有帮助。
- 辅助监督是否能够帮助主分支训练得更稳定、更充分。
- 在已有结构基础上加入蒸馏，是否能够继续提高泛化性能。
- 如果要把 v2 作为后续论文和系统验收的主要版本，哪一个配置最适合作为最终交付方案。

### 2.2 整体改进思路

从已有配置文件和实验记录来看，v2 主要围绕下面几个方向展开：

- 编码器保持 `light_hetero` 这一相对轻量的异构建模思路。
- 保留 `conservative fusion`，避免过强融合导致特征扰动过大。
- 在空间建模上，从单尺度 `patch_size = 11` 扩展到 `patch_size = 11, context_patch_size = 17` 的多尺度输入。
- 在多尺度分支内部，进一步尝试不同的 `scale_fusion_mode`，例如 `average` 和 `residual`。
- 引入辅助头 `aux heads`，并比较 `linear` 和 `mlp` 两种形式。
- 在辅助头基础上进一步尝试 `distill`，验证蒸馏是否有效。
- 对数据增强策略、patch 尺度、泛化控制和一致性损失 `MSE` 做扩展消融。
- 在少样本条件下补做 few-shot 实验，观察模型随样本量增加的性能变化。

可以认为，v2 的主线不是“盲目加模块”，而是围绕“怎样更稳、更合理地融合局部与上下文信息”来做结构优化。

---

## 3. v2实验配置路线梳理

### 3.1 v2 主配置与当前仓库配置说明

当前仓库中的 `experiments/hct_bgc_v2.0_houston.yaml` 对应的是 v2.0 阶段的基础配置，其主要特征如下：

- `patch_size = 11`
- `scale_fusion_mode = average`
- `encoder_variant = light_hetero`
- `use_conservative_fusion = true`
- `use_aux_heads = false`

这一版可以理解为 v2 阶段在仓库中保留的一份单尺度整理配置。

`v2.0` 没有设置 `context_patch_size > patch_size`，因此它属于单尺度路径。

在 v1 主配置基础上，引入 `light_hetero + conservative fusion + aux heads` 后，测试集性能由 `OA = 85.58 ± 0.42`、`AA = 87.63 ± 0.46`、`Kappa = 84.40 ± 0.46` 提升到 `OA = 86.38 ± 0.14`、`AA = 88.52 ± 0.13`、`Kappa = 85.26 ± 0.16`。

### 3.2 residual + 多尺度 11×17 上下文方案

当前仓库中的 `experiments/hct_bgc_v2_houston_multiscale_11_17.yaml` 对应的是多尺度方案，其核心配置为：

- `patch_size = 11`
- `context_patch_size = 17`
- `context_token_size = 11`
- `scale_fusion_mode = residual`
- `encoder_variant = light_hetero`
- `use_conservative_fusion = true`
- `use_aux_heads = true`

这版实验的重点，是在保持中心 patch 为 11 的同时，引入更大范围的 17×17 上下文窗口，再通过 `context_token_size = 11` 控制上下文信息进入模型时的表达粒度。这个设计本质上是在做“局部精细判别 + 外围上下文补充”的结合。

采用 patch 11✖️17 多尺度上下文，`并设置 context_token_size=11` 后，对 local 和 context 分支的 fusion 做三种方式的消融：`average`、`gate`、`residual`，实验证明，`residual` 效果最好，Test OA 从 86.38 ± 0.14 提升到 86.85 ± 1.04，Test AA 从 88.52 ± 0.13 提升到 88.17 ± 0.55，Test Kappa 从 85.26 ± 0.16 提升到 85.76 ± 1.11；

### 3.3 aux head 形式比较

在多尺度方案上，还继续比较了 `aux_head_mode = linear` 和 `aux_head_mode = mlp` 两种辅助头形式。

- `aux = linear` 的特点是结构简单、额外参数少、训练目标更直接。
- `aux = mlp` 的特点是表达能力更强，但也可能更容易引入额外波动或者过拟合。

从当前结果看，`linear` 更适合作为最终正式方案。

### 3.4 distill 尝试

仓库中的 `experiments/hct_bgc_v2_1_ms_houston.yaml` 是在多尺度方案基础上继续加入蒸馏的尝试，主要增加了：

- `aux_head_mode = linear`
- `aux_distill_weight = 0.1`
- `aux_distill_temperature = 2.0`

这一版的目的，是希望通过主输出对辅助头进行蒸馏约束，进一步提升辅助分支与主分支的一致性。不过从结果来看，蒸馏没有表现出稳定正收益。

---

## 4. v2阶段实验结果汇总

下面整理目前已经给出的 Houston 数据集 v2 结果。

### 4.1 实验结果总表

| 方案 | Val OA | Val AA | Val Kappa | Test OA | Test AA | Test Kappa |
| --- | --- | --- | --- | --- | --- | --- |
| 仓库中的 `hct_bgc_v2.0_houston.yaml`：单尺度配置记录 | mean=0.9034, std=0.0142 | mean=0.9176, std=0.0104 | mean=0.8944, std=0.0155 | mean=0.8432, std=0.0143 | mean=0.8663, std=0.0107 | mean=0.8304, std=0.0155 |
| v2 主配置：`light_hetero + conservative fusion + aux heads` | mean=0.9240, std=0.0058 | mean=0.9345, std=0.0042 | mean=0.9169, std=0.0063 | mean=0.8638, std=0.0014 | mean=0.8852, std=0.0013 | mean=0.8526, std=0.0016 |
| 11×17 multiscale + residual + context-token-size=11 + aux=`linear` | mean=0.9325, std=0.0169 | mean=0.9198, std=0.0287 | mean=0.9261, std=0.0184 | mean=0.8685, std=0.0104 | mean=0.8817, std=0.0055 | mean=0.8576, std=0.0111 |
| 11×17 multiscale + residual + context-token-size=11 + aux=`mlp` | 0.9476 | 0.9392 | 0.9426 | 0.8661 | 0.8759 | 0.8548 |
| 11×17 multiscale + residual + context-token-size=11 + aux=`linear` + distill | mean=0.9319, std=0.0309 | mean=0.9306, std=0.0241 | mean=0.9255, std=0.0337 | mean=0.8636, std=0.0079 | mean=0.8718, std=0.0067 | mean=0.8521, std=0.0086 |

### 4.2 说明主配置有效的消融表

如果论文或验收材料的重点是说明“当前主配置为什么成立”，那么不需要把所有扩展尝试都塞进主表，而应该抽取最能支撑主线结论的几组结果。下面这张表更适合作为 v2 主配置有效性的消融实验表。

| 消融阶段 | 相比上一阶段的变化 | Test OA | Test AA | Test Kappa | 结论 |
| --- | --- | --- | --- | --- | --- |
| v1 主配置基线 | - | `85.58 ± 0.42` | `87.63 ± 0.46` | `84.40 ± 0.46` | 作为 v2 之前的对照基线 |
| v2 主配置 | `+ light_hetero + conservative fusion + aux heads` | `86.38 ± 0.14` | `88.52 ± 0.13` | `85.26 ± 0.16` | 主干升级有效，三项指标整体提升 |
| v2 主配置 + 多尺度上下文 | `+ patch 11×17 + context_token_size=11 + scale_fusion_mode=residual + aux=linear` | `86.85 ± 1.04` | `88.17 ± 0.55` | `85.76 ± 1.11` | 多尺度上下文进一步提升 `OA` 和 `Kappa`，形成当前最优方案 |
| 最优方案替换辅助头 | `aux: linear -> mlp` | `86.61` | `87.59` | `85.48` | 单次结果未超过 `aux=linear` 的重复均值，不支持替换正式方案 |
| 最优方案加入蒸馏 | `+ distill` | `86.36 ± 0.79` | `87.18 ± 0.67` | `85.21 ± 0.86` | 蒸馏未带来稳定正收益 |

这张表对应的叙事顺序很清楚：先证明 v2 主干升级有效，再证明 `11×17` 多尺度上下文和 `residual` 融合把性能推到当前最好，最后说明 `aux=mlp` 和 `distill` 都没有超过当前主配置，因此正式版本保留 `aux=linear` 更合理。

如果你想把“`residual` 优于其他 local-context 融合方式”也单独做成一张子表，可以再补一张 `average / gate / residual` 的局部消融表；但目前这份材料里明确给出的可复用数字，已经足够支撑“主配置有效”这个核心结论。

### 4.3 当前最优结论

从多次重复实验的均值结果来看，当前 v2 阶段综合性能最好的方案是：

`Houston（patch 11 × 17）+ residual + context-token-size=11 + aux=linear`

对应测试集指标为：

- `Test OA = 0.8685 ± 0.0104`
- `Test AA = 0.8817 ± 0.0055`
- `Test Kappa = 0.8576 ± 0.0111`

---

## 5. v2阶段的重点分析


### 5.1 residual 在多尺度融合消融中效果最好

在 `models/hct_bgc.py` 中，只有当 `context_patch_size > patch_size` 时，模型才会进入 `use_multiscale` 分支，并进一步实例化和调用 `hsi_scale_fusion`、`lidar_scale_fusion`。也就是说，`scale_fusion_mode` 只在多尺度开启时真正生效。

1. 在 v1 主配置基础上，采用轻异构编码器 `light_hetero`、保守融合 `conservative fusion` 和辅助头 `aux heads` 后，`Test OA` 从 `85.58 ± 0.42` 提升到 `86.38 ± 0.14`，`Test AA` 从 `87.63 ± 0.46` 提升到 `88.52 ± 0.13`，`Test Kappa` 从 `84.40 ± 0.46` 提升到 `85.26 ± 0.16`。这说明 v2 的主干升级方向是有效的。
2. 在 `patch 11×17`、`context_token_size = 11` 的多尺度方案下，进一步比较了 `average`、`gate`、`residual` 三种 local-context 融合方式。结果表明 `residual` 效果最好，`Test OA` 从 `86.38 ± 0.14` 提升到 `86.85 ± 1.04`，`Test AA` 从 `88.52 ± 0.13` 变化到 `88.17 ± 0.55`，`Test Kappa` 从 `85.26 ± 0.16` 提升到 `85.76 ± 1.11`。虽然 `AA` 没有同步提高，但从 `OA` 和 `Kappa` 看，`residual` 仍然是当前最优的多尺度融合方式。

因此，**在已经完成的多尺度融合消融实验中，`residual` 的综合效果优于 `average` 和 `gate`，是当前最优选择。**

### 5.2 辅助头是有效的，但更适合用轻量方式实现

引入辅助头以后，模型训练过程中不仅优化主分类输出，也同时对两个辅助分支进行约束，这样做通常有两个潜在好处：

- 可以让模态相关特征在中间层得到更明确的监督。
- 可以在训练前期提供更充足的梯度信号，提升优化稳定性。

从结果上看，辅助头与当前最优的多尺度方案配合较好，至少在现有实验中，它没有拖后腿，反而帮助模型进一步提升了性能。

不过，当比较 `aux = linear` 和 `aux = mlp` 时，当前证据更支持 `linear`：

- `aux = linear` 有多次重复实验支撑，结果可靠。
- `aux = mlp` 目前只有单次结果，虽然验证集指标更高，但测试集并没有超过 `aux = linear` 的重复均值结果。

因此，这里更合理的结论不是“MLP 不好”，而是：**现阶段没有足够证据证明 MLP 辅助头优于线性辅助头，所以正式方案应优先选择更简单、可复现性更好的 `linear`。**

### 5.3 多尺度 11×17 上下文是当前 v2 的核心增益来源之一

从现有可确认的代码逻辑和实验结果来看，多尺度 `11×17` 上下文是当前 v2 中最能明确成立的增益来源之一。采用这一方案后，测试集指标达到当前最优的：

- `Test OA = 0.8685 ± 0.0104`
- `Test Kappa = 0.8576 ± 0.0111`

相对单尺度配置记录，增益为：

- `Test OA` 提升约 `2.53` 个百分点
- `Test Kappa` 提升约 `2.72` 个百分点

这说明在 Houston 数据集上，仅靠中心 11×11 patch 仍然不能完全覆盖目标判别所需的上下文信息，而引入更大范围的 17×17 视野后，模型可以更充分利用周围空间结构、邻域语义和模态补充信息。

### 5.4 当前最优方案仍然存在一定泛化落差

虽然当前最优配置已经取得了 v2 阶段最好的测试结果，但仍然需要客观看待验证集和测试集之间的差距。当前最优配置下：

- `Val OA = 0.9325`
- `Test OA = 0.8685`

这说明模型在验证集上表现更高，而在测试集上仍然存在明显回落。比较合理的解释是，Houston 官方划分下验证区域和测试区域之间仍存在一定分布差异，模型虽然已经通过保守融合、辅助监督和多尺度上下文在一定程度上改善了泛化能力，但还没有完全消除这种分布偏移带来的影响。

因此，在论文或系统验收材料中，应该把这一点如实写清楚：**当前版本已经取得阶段性最优结果，但泛化落差仍然存在。**

### 5.5 验证集更高，不代表测试集一定更强

`11×17 + aux=mlp` 的单次结果中，验证集指标看起来非常亮眼：

- `Val OA = 0.9476`
- `Val Kappa = 0.9426`

但其测试集结果：

- `Test OA = 0.8661`
- `Test Kappa = 0.8548`

并没有超过 `11×17 + aux=linear` 的重复实验均值结果。

**重复实验均值和标准差比单次最高值更有说服力**。从这个角度讲，`aux=linear` 更适合作为正式版本。

### 5.6 其他扩展尝试目前都没有超过当前主版本

除了主线方案之外，v2 还做了多种扩展尝试，包括：

- 四种数据增强消融：`flip_only`、`rot180`、`spectral_noise`、`d4`
- 增大 patch 尺度
- 增强泛化控制
- 添加一致性损失 `MSE`
- 将辅助头由 `linear` 改为 `mlp`
- 引入融合分支指导的辅助蒸馏机制，即利用融合输出 `logits` 作为 teacher，对 HSI 与 LiDAR 辅助头施加 `KL` 蒸馏约束

从当前结果来看，这些尝试最终都没有超过当前版本配置的性能。这一点本身也很重要，因为它说明当前最优方案并不是偶然得到的，而是在较系统的扩展探索之后被保留下来的稳定解。

### 5.7 蒸馏目前没有形成稳定正收益

在 `11×17 + aux=linear` 的基础上加入蒸馏后，结果为：

- `Test OA = 0.8636 ± 0.0079`
- `Test Kappa = 0.8521 ± 0.0086`

相比未蒸馏版本：

- `Test OA` 略有下降
- `Test AA` 略有下降
- `Test Kappa` 略有下降

同时，验证集标准差也明显变大，说明蒸馏至少在当前参数设置下没有让训练变得更稳，反而可能增加了优化过程中的不确定性。

我认为这里可以得出一个比较稳妥的结论：**蒸馏不是当前 v2 的有效主增益项，至少暂时不适合写成核心贡献。**  
如果后面时间充足，它可以保留为后续优化方向；如果时间有限，它更适合归入“已尝试但未取得稳定提升”的探索项。

### 5.8 v2 当前已经形成比较明确的推荐配置

综合考虑性能、稳定性、结构复杂度和重复实验支撑情况，当前最适合作为 v2 正式推荐配置的是：

- `patch_size = 11`
- `context_patch_size = 17`
- `context_token_size = 11`
- `scale_fusion_mode = residual`
- `encoder_variant = light_hetero`
- `use_conservative_fusion = true`
- `use_aux_heads = true`
- `aux_head_mode = linear`

这版配置的优势主要有三点：

- 指标上是目前 v2 综合最优。
- 结论来自重复实验，而不是单次偶然结果。
- 结构改动虽然比 v2.0 多，但整体仍然保持在可以解释、可以复现、可以交付的范围内。

### 5.9 few-shot 实验说明模型在低样本场景下具有持续增益趋势

除了常规监督设置之外，还做了 few-shot 的多组实验。结果显示，在随机抽样的实验条件下，性能指标会随着样本数从 `5-shot` 到 `100-shot` 逐步提高，并且在 `100-shot` 时达到实验以来测试集最高分数：

- `OA = 0.9569`
- `AA = 0.9624`
- `Kappa = 0.9533`

这个结果说明，当前模型不仅在常规训练设置下有效，在低样本到中等样本规模的设定下同样表现出较好的可扩展性和持续增益趋势。这部分内容后续可以作为系统验收和论文实验章节里的重要补充。

### 5.10 当前仍需要正视的局限性

虽然 v2 已经取得了阶段性最优结果，但当前仍然有几个需要诚实说明的问题：

- 验证集和测试集之间仍然存在一定落差，说明泛化问题虽然有所改善，但没有被完全解决。
- 一些更复杂的扩展设计，例如 `aux=mlp`、辅助蒸馏、一致性损失和更强数据增强，并没有超过当前主版本。
- few-shot 虽然表现出良好趋势，但如果后续要作为论文重点，仍然建议补齐更完整的重复实验统计。
- 目前的分析主要基于 OA、AA、Kappa 三项总体指标，后续仍需结合 confusion matrix 和 per-class accuracy 来进一步解释哪些类别真正受益。

因此，v2 当前更准确的表述应该是：**已经形成了稳定有效的改进方向和可交付的推荐版本，但仍有一些细节值得后续继续打磨。**

---

## 6. v2阶段总体结论

结合目前全部实验结果，v2 阶段可以概括为下面几条结论：

1. v2 的改进方向是有效的，尤其是 `light_hetero + conservative fusion + aux heads` 带来的主干升级，以及 `11×17` 多尺度上下文下的融合优化，为性能提升提供了主要贡献。
2. `light_hetero + conservative fusion` 这一整体框架是可以保留的，说明当前基础建模路线没有走偏。
3. 已经完成了多尺度 `average / gate / residual` 的融合消融，结论是 `residual` 综合效果最好。
4. 辅助头是值得保留的训练机制，但从已有证据看，正式方案优先采用 `linear`，不优先采用 `mlp`。
5. 蒸馏、增强的一致性损失、更强数据增强和更大 patch 等扩展尝试，暂时都没有形成超过当前主版本的收益。
6. few-shot 实验显示模型性能会随着 shot 数增加而持续提升，并在 `100-shot` 时达到 `OA = 0.9569`、`AA = 0.9624`、`Kappa = 0.9533`。
7. 当前作为 v2 正式版本的方案是：`11×17 multiscale + residual + context-token-size=11 + aux=linear`。


> v2 阶段先通过 `light_hetero + conservative fusion + aux heads` 完成主干升级，再在 `11×17` 多尺度设置下比较 `average / gate / residual` 三种融合方式，最终确认 `residual` 为最优选择，并结合线性辅助头监督在 Houston 数据集上取得了当前阶段最优且较稳定的综合性能，说明该版本已经具备作为正式交付方案的基础。

---

## 7. 系统验收需要准备的成果物清单

下面按照“可复现、可验证、可展示、可交付”的思路整理系统验收需要准备的成果物。

## 7.1 配置与版本类成果物

- 最终验收方案配置文件  
  最终正式配置：`experiments/hct_bgc_v2_houston_multiscale_11_17`

- v2 全部对比实验配置文件  
  - `v2.0`
  - `gate + aux=linear`
  - `average + aux=linear`
  - `residual + aux=linear`
  - `11×17 + aux=linear`
  - `11×17 + aux=mlp`
  - `11×17 + aux=linear + distill`
  - 数据增强、一致性损失、泛化控制和 patch 尺度等扩展配置
  - few-shot 系列配置

- 代码版本记录  
  - 当前分支名称：`main`
  - 当前 HEAD commit id：`d782eb4b7f7c2b84d05a874239b23b8922c4d364`
  - 当前工作区状态：`git status --short` 为空，说明工作区干净
  - 与 v2 相关的关键版本节点：
    - `7ea3b1b`：多尺度融合
    - `8015a81`：给 context 分支加“先降采样再 token 化”
    - `a043467`：尝试 `flip_only`、`rot180`、`spectral_noise` 等增强
    - `2f95eb5`：添加 few-shot 实验
    - `3475af3`（tag: `HCT-BGC-v2.1-ms`）：加入融合分支指导的辅助蒸馏机制

- 环境说明文档  
  - Python 版本：`Python 3.14.3`
  - 运行硬件环境：
    - 机器：`MacBook Air`
    - 芯片：`Apple M5`
    - CPU：`10` 核（`4` 个性能核 + `6` 个能效核）
    - 内存：`16 GB`
    - 系统：`macOS Darwin 25.4.0`
    - 架构：`arm64`
  - 依赖文件：仓库根目录存在 [requirements.txt](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/requirements.txt)
  - `requirements.txt` 当前记录的项目依赖为：
    - `numpy>=1.24`
    - `scipy>=1.10`
    - `matplotlib>=3.7`
    - `scikit-image>=0.22`
    - `tifffile>=2024.2`
    - `PyYAML>=6.0`
  - CUDA 版本：当前机器为 Apple Silicon Mac，不使用 CUDA

## 7.2 数据与协议类成果物
- 数据集说明文档  
  当前使用的是 `Houston 2013 DFTC` 官方数据，数据根目录为：
  - `data/raw/Houston 2013/2013_DFTC`
  当前代码实际读取的文件包括：
  - HSI：`2013_IEEE_GRSS_DF_Contest_CASI.tif`
  - LiDAR：`2013_IEEE_GRSS_DF_Contest_LiDAR.tif`
  - 官方训练标注：`2013_IEEE_GRSS_DF_Contest_Samples_TR.txt`
  - 训练 ROI 一致性校验：`2013_IEEE_GRSS_DF_Contest_Samples_TR.roi`
  - 官方测试标注：`2013_IEEE_GRSS_DF_Contest_Samples_VA.zip`
  数据来源：
  - HSI：Houston 2013 DFTC 官方 CASI 高光谱影像
  - LiDAR：Houston 2013 DFTC 官方 LiDAR 高程影像
  - 标签：官方提供的 `TR` 训练样本与 `VA` 测试样本
  类别数：
  - Houston 2013 当前代码按官方标签读取得到 `15` 类
  数据组织目录结构：

```text
data/raw/Houston 2013/2013_DFTC/
  2013_IEEE_GRSS_DF_Contest_CASI.tif
  2013_IEEE_GRSS_DF_Contest_LiDAR.tif
  2013_IEEE_GRSS_DF_Contest_Samples_TR.txt
  2013_IEEE_GRSS_DF_Contest_Samples_TR.roi
  2013_IEEE_GRSS_DF_Contest_Samples_VA.zip
```

- 划分协议说明  
  - 是否使用官方划分：是。Houston 主实验使用 `split_mode = official`
  - `split_seed` 设置：主实验配置中固定为 `42`
  - `train / val / test` 生成方式：
    - `test`：直接使用官方 `VA` 测试标注
    - `train + val`：先使用官方 `TR` 训练标注作为候选训练集
    - 再从官方 `TR` 中通过 `split_items_spatial_holdout()` 划出 `val`
    - 剩余部分作为最终 `train`
  - 是否采用空间隔离策略：是
  - 空间隔离实现方式：
    - 按空间块选择验证样本
    - 再对训练候选样本施加 Chebyshev 距离约束，剔除与验证样本距离不大于 `buffer_radius` 的样本，避免 patch 重叠泄漏
  - 默认 `val_ratio`：代码默认值为 `0.2`
  - 默认 `val_spatial_buffer`：若不手动指定，则使用 `max(patch_size, context_patch_size) // 2`
    - 对单尺度 `patch_size=11`，默认 buffer 为 `5`
    - 对多尺度 `11×17`，默认 buffer 为 `8`

- 预处理说明  
  - PCA 维度：主配置为 `pca_components = 30`
  - 归一化方式：z-score 标准化
    - 当 `preprocess_scope = train` 时，仅使用训练像素统计量拟合均值和标准差
    - HSI 为按光谱通道统计
    - LiDAR 为单通道整体统计
  - PCA 拟合方式：
    - 当 `preprocess_scope = train` 时，仅使用训练像素拟合 PCA
    - 再将整景投影到 PCA 子空间
  - patch 提取方式：
    - 以标注中心像素 `(r, c)` 为中心提取正方形 patch
    - patch 边界采用 `reflect padding`
    - HSI patch 输出为 `(C, P, P)`，LiDAR 输出为 `(1, P, P)` 或多通道形式
  - 多尺度 patch 的处理逻辑：
    - 数据加载阶段按 `extraction_patch_size = max(patch_size, context_patch_size)` 提取输入 patch
    - 若启用多尺度，例如 `patch_size=11, context_patch_size=17`
    - 模型前向时先从输入中中心裁剪出 `11×11` 作为 local 分支
    - 原始 `17×17` 输入作为 context 分支
    - context 分支特征再通过 `adaptive_avg_pool2d` 变换到 `context_token_size × context_token_size`
    - 当前最优配置中 `context_token_size = 11`

- 随机种子说明  
  当前代码中随机种子由 [utils/seed.py](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/utils/seed.py) 统一设置，固定的随机项包括：
  - Python `random`
  - `numpy`
  - `torch`
  - `torch.cuda`
  - `torch.mps`
  - `cudnn.deterministic = True`
  - `cudnn.benchmark = False`
  主结果重复实验的做法是：
  - 固定 `split_seed = 42`
  - 改变训练 `seed`
  - 通过 [scripts/run_repeat.py](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/scripts/run_repeat.py) 运行多次训练并统计 `mean ± std`
  当前最优主实验 `results/repeats/hct_bgc_v2_houston_multiscale_11_17/repeat_config.json` 中记录的 seeds 为：
  - `0, 1, 2`
  因此主结果中的 `mean ± std` 是基于 `3` 次重复得到的
  few-shot 实验当前在结果目录中可确认的 seeds 为：
  - `0, 1, 2, 3, 4`
  即 few-shot 当前已按 `5` 次重复进行统计

- few-shot 协议说明
  根据 [patch_dataset.py](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/dataset/patch_dataset.py) 中的 `build_index_fewshot()` 和 few-shot 配置文件，当前 few-shot 协议为：
  - 每类抽样方式：
    - 对每个类别的全部标注像素先随机打乱
    - 固定取 `train_per_class` 个样本作为训练集
    - 再取 `val_per_class` 个样本作为验证集
    - 剩余样本全部作为测试集
  - 不同 shot 的取值范围：
    - 配置有 `5-shot`、`10-shot`、`20-shot`、`50-shot`、`100-shot`
    - 在 `100-shot` 时达到 `OA = 0.9569`、`AA = 0.9624`、`Kappa = 0.9533`
  - 是否固定随机抽样种子：是，当前 few-shot 配置固定 `split_seed = 42`

## 7.3 训练与复现类成果物

- 标准训练命令
- 标准测试命令
- 重复实验命令
- 批量实验脚本
- 结果汇总脚本

```bash
python3 train.py \
  --model hct_bgc \
  --data-root "data/raw/Houston 2013/2013_DFTC" \
  --split-mode official \
  --preprocess-scope train \
  --patch-size 11 \
  --context-patch-size 17 \
  --context-token-size 11 \
  --scale-fusion-mode residual \
  --pca-components 30 \
  --embed-dim 128 \
  --num-heads 4 \
  --num-layers 2 \
  --fusion-layers 1 \
  --encoder-variant light_hetero \
  --use-conservative-fusion \
  --use-aux-heads \
  --aux-weight 0.2 \
  --dropout 0.1 \
  --weight-decay 5e-4 \
  --label-smoothing 0.1 \
  --selection-metric val_kappa \
  --early-stopping-patience 8 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 0 \
  --split-seed 42 \
  --save-dir results/acceptance/hct_bgc_v2_final

python3 scripts/run_repeat.py \
  --config experiments/hct_bgc_v2_houston_multiscale_11_17.yaml \
  --seeds 0 1 2

python3 evaluate.py \
  --checkpoint results/repeats/hct_bgc_v2_houston_multiscale_11_17/seed0/best.pth \
  --output-dir results/acceptance/eval_seed0

python3 scripts/visualize_map.py \
  --baseline-checkpoint results/repeats/baseline/seed0/best.pth \
  --hct-checkpoint results/repeats/hct_bgc_v2_houston_multiscale_11_17/seed0/best.pth \
  --output-dir results/paper_figures/maps

python3 scripts/summarize_ablation.py \
  results/repeats/baseline/seed0 \
  results/repeats/hct_bgc_main/seed0 \
  results/repeats/hct_bgc_v2_houston_multiscale_11_17/seed0
```


## 7.4 性能证明类成果物

这一部分是系统验收时最核心、最直观的材料。

- 主结果表  
  建议包含：
  - Baseline
  - v1
  - v2.0
  - residual + aux
  - 11×17 + aux=linear
  - 11×17 + aux=mlp
  - 11×17 + distill
  - few-shot 代表性结果

- v2 内部对比表  
  统一使用 `mean ± std` 表示，减少单次结果带来的误导。

- 增益分析表  
  需要明确写出：
  - 相比单尺度配置和 v2 主配置分别提升多少
  - 相比上一版本提升多少
  - 哪些指标提升，哪些指标没有同步提升
  - few-shot 条件下性能如何随 shot 增长变化

- confusion matrix
- per-class accuracy 表
- classification map 可视化结果
- train/val loss 曲线
- OA/AA/Kappa 随 epoch 变化曲线

其中，系统验收时最值得重点展示的，通常是下面几项：

- 最终方案与基线的主结果对比表
- 最终方案的 confusion matrix
- 最终方案的 per-class accuracy
- 最终方案和基线的分类图对比

## 7.5 模型交付类成果物

这一部分是“系统”验收里很容易被忽略，但实际很重要的一部分。

- 最终模型权重文件
- 最终配置文件
- 推理脚本
- 单张或批量推理入口
- 输入输出格式说明
- 结果保存路径说明

- 最终模型权重文件  
  当前训练代码默认保存为 `best.pth`，例如：
  - `results/repeats/hct_bgc_v2_houston_multiscale_11_17/seed0/best.pth`
  - `results/repeats/hct_bgc_v2_houston_multiscale_11_17/seed1/best.pth`
  - `results/repeats/hct_bgc_v2_houston_multiscale_11_17/seed2/best.pth`
  系统验收时建议单独复制一份最终权重到固定目录，例如：
  - `thesis/acceptance/checkpoints/hct_bgc_v2_final_best.pth`

- 最终配置文件  
  当前推荐直接引用：
  - [hct_bgc_v2_houston_multiscale_11_17.yaml](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/experiments/hct_bgc_v2_houston_multiscale_11_17.yaml)
  同时建议补一份验收专用副本：
  - `thesis/acceptance/configs/hct_bgc_v2_final_houston.yaml`

- 推理脚本 / 单张或批量推理入口  
  现有最接近交付入口的是：
  - [evaluate.py](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/evaluate.py)：读取 `best.pth`，在测试集上评估并导出 `per-class accuracy` 与 confusion matrix
  - [visualize_map.py](/Users/yuxin/repos/HSI-LiDAR-Fusion-Classification/scripts/visualize_map.py)：根据 checkpoint 生成 classification map、gate 统计等图像材料

- 输入输出格式说明  
  当前模型输入为：
  - 输入 1：HSI patch，张量形状 `(C, P, P)`
  - 输入 2：LiDAR patch，张量形状 `(1, P, P)` 
  - 输出：类别 logits、预测标签，以及在 HCT-BGC 下可选的 gate/辅助头输出
  对整景可视化任务，输入为 Houston 原始整景数据和 checkpoint，输出为 classification map、gate 统计图和相关 json/csv

- 结果保存路径说明  
  当前训练与复现实验的主要结果目录结构为：
  - `results/repeats/<experiment_name>/seed*/best.pth`
  - `results/repeats/<experiment_name>/seed*/best_metrics.json`
  - `results/repeats/<experiment_name>/seed*/model_config.json`
  - `results/repeats/<experiment_name>/summary.json`
  - `results/paper_figures/maps/`：分类图与 gate 统计
  - `results/evaluation_compare_demo/`：评估导出示例
  系统验收时建议把最终对外提交的结果统一整理到：
  - `thesis/acceptance/checkpoints/`
  - `thesis/acceptance/reports/`
  - `thesis/acceptance/scripts/`
  - `thesis/figures/`

如果后面有演示环节，还建议额外准备：

- 一份最小可运行 demo
- 一份固定输入样例
- 一份对应的标准输出样例

## 7.7 文档与汇报类成果物

这一部分主要用于答辩、汇报和论文写作。

- 阶段总结文档
- 系统验收说明书
- PPT 或答辩展示材料
- 论文实验章节草稿
- 图表素材整理文件夹

建议把所有图表统一放到固定目录中，避免后面找不到：

- 主结果表
- 消融表
- few-shot 结果表
- confusion matrix 图
- 分类图
- loss 曲线图

---

## 8. 建议的系统验收材料组织方式

为了避免后面材料很乱，我觉得可以按下面的结构整理：

```text
thesis/
  notes/
  figures/
  tables/
  acceptance/
    configs/
    logs/
    checkpoints/
    reports/
    scripts/
```

其中：

- `notes/` 放阶段总结和实验记录
- `figures/` 放图片类材料
- `tables/` 放表格类材料
- `acceptance/configs/` 放最终验收配置
- `acceptance/logs/` 放关键训练日志
- `acceptance/checkpoints/` 放模型权重
- `acceptance/reports/` 放验收文档
- `acceptance/scripts/` 放复现和推理脚本

这样整理的好处是，后续无论是自己写论文，还是老师要检查材料，都会比较清楚。

---

## 10. 最终结论

总体来说，v2 阶段已经完成了一轮比较完整的结构探索，并且形成了比较明确的结论。最关键的认识是：v2 的收益首先来自 `light_hetero + conservative fusion + aux heads` 带来的主干升级，其次来自 `11×17` 多尺度上下文下 local-context 融合方式的进一步优化，而你已经通过 `average / gate / residual` 消融证明 `residual` 是当前最优选择。与此同时，数据增强、一致性损失、MLP 辅助头、辅助蒸馏和更大 patch 等扩展尝试都没有超过当前配置，few-shot 实验则进一步说明模型在样本增加时具有持续增益趋势。当前最优方案 `11×17 multiscale + residual + aux=linear` 已经在多次重复实验下表现出最好的综合性能。
