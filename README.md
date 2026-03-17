# HSI + LiDAR Fusion Classification

## 2026/3/5 进度日志
- 完成数据 pipeline 的搭建:dataset/*、scripts/check_pipeline.py
- pipe1.读取 data_hl.mat文件，提取HSI、LiDAR和GT数据
- pipe2.Norm → PCA → Patch 
- pipe3.划分训练样本列表 + patch数据集 
- 完成数据处理 pipeline 的检查
## 2026/3/6 进度日志
- 实现双分支 CNN baseline：models/baseline_cnn.py、utils/metrics.py、train.py
- 在 train.py 中的最最简单训练闭环中使用以下命令可以成功训练个5epoch
  ```bash
  python train.py --epochs 5 --batch-size 64 --patch-size 13
  ```
  得到如下的训练结果：
  ```bash
    Using device: cpu
    Train/Val/Test batches: 127/15/95 | HSI channels: 30 | Classes: 15
    Epoch 001 | train_loss=0.7675 | val_loss=0.1270 | val_oa=0.9689 | val_aa=0.9627 | val_kappa=0.9664
    Epoch 002 | train_loss=0.0874 | val_loss=0.0962 | val_oa=0.9667 | val_aa=0.9631 | val_kappa=0.9640
    Epoch 003 | train_loss=0.0610 | val_loss=0.0894 | val_oa=0.9756 | val_aa=0.9717 | val_kappa=0.9736
    Epoch 004 | train_loss=0.0275 | val_loss=0.0274 | val_oa=0.9933 | val_aa=0.9936 | val_kappa=0.9928
    Epoch 005 | train_loss=0.0260 | val_loss=0.0687 | val_oa=0.9723 | val_aa=0.9769 | val_kappa=0.9700
    Test | loss=0.0521 | oa=0.9824 | aa=0.9857 | kappa=0.9810 
结果偏高，分析原因如下：
- build_index()在patch_dataset.py中，是在同一张大图上按照类别随机抽像素，分成train/test，会导致train、test在空间上往往离得很近
- 使用patch，而相邻patch高度重叠，导致teain\test不够独立
- 验证集是从训练集切出来的，更容易高
## 2026/3/8进度日志
- 解决 baseline 分数偏高的问题
- 调整 data 数据源为官方数据源，数据划分采用官方形式
- 得到对比结果（1个 epoch ）:
  - 官方 split + train-only preprocess: OA 0.7145
  - 官方 split + full-scene preprocess: OA 0.6831
  - 随机像素 split + full-scene preprocess: OA 0.8838
## 2026/3/9进度日志
- 把 baseline 训练和评估真正跑通
- 开始实现 HCT-BGC 的主干骨架（先不加对比学习）
- 加入 Bi-CTA 和 Gated Fusion，跑第一版主模型 
- base 环境改成虚拟环境 + cpu 跑改为 cuda 跑
- 补评估、出第一张分类图、写周报
## 2026/3/12进度日志
- 调整代码兼容mac的mps加速
- 梳理主干接口，补齐论文 Method 对应的代码开关
- 补齐 HCT-BGC 结构细节，做成真正的 v1 主干：tag v0.1.0
- 接入 Contrastive Alignment Loss
### macOS Apple Silicon 说明
- M 系列 Mac 不使用 CUDA，PyTorch 应直接安装官方默认包并使用 `mps` 加速。
- 建议命令：
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install torch torchvision torchaudio
  python3 -m pip install -r requirements.txt
  ```
- 训练时可显式指定：
  ```bash
  python3 train.py --device mps
  ```
- 也可以保持默认 `--device auto`，脚本会按 `cuda -> mps -> cpu` 顺序自动选择。
## 2026/3/13进度日志
- 整理先前代码逻辑
- 验证集选模型，测试集只报最终结果，原先只划分了 train / test ，现在加上 val
- 最佳模型的选用开始是基于 best_oa，现在改为基于 val 的验证
- 做最关键的 3 组消融实验：
  1.mac的mps加速报错，改为cpu
  2.val划分和train高度重叠导致分数虚高，修正
  3.训练结束时把最终测试集的混淆矩阵和每类精度落盘
  4.加一个汇总脚本，把各实验目录的 best_metrics.json 自动整理成 ablation markdown 表，scripts/summarize_ablation.py，直接从各实验目录生成 markdown 表
## 2026/3/14进度日志
- 导出论文素材：classification map、gate 统计、训练曲线、网络结构图草稿
- 固定第 2 周主配置：`run_hct_bgc_v1_main`
- 完成第 2 周主结果收口，整理 README、周报和第 3 周实验清单
- 开启第三周任务：实验环境稳定化 + 多次运行脚本：跑出同一协议、同一 split、3 次重复运行下的 baseline 和 hct_bgc 的可比结果，在 results/repeats/
## 第 2 周稳定版本：HCT-BGC-v1

当前推荐主实验配置命名为 `run_hct_bgc_v1_main`，对应命令：

```bash
python3 train.py \
  --model hct_bgc \
  --fusion-layers 1 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 0 \
  --output-dir results/run_hct_bgc_v1_main
```

### 模型结构

- 双分支 CNN encoder：分别提取 HSI 与 LiDAR 的局部特征。
- Tokenization：将 patch 特征图投影为 token 序列，并加入 `CLS token + learnable positional embedding`。
- 模态内 Transformer Encoder：两路 token 各自建模长程依赖。
- Bi-CTA Fusion Block：使用 `HSI cls -> LiDAR 全部 tokens`、`LiDAR cls -> HSI 全部 tokens` 的双向交互。
- Gated Fuse：`g = sigmoid(Wg[h_cls; l_cls])`，`fused = g * h_cls + (1-g) * l_cls`。
- MLP classifier：基于 `fused_token` 输出最终分类 logits。

### 主实验超参数

- `patch_size=11`
- `pca_components=30`
- `embed_dim=128`
- `num_heads=4`
- `num_layers=2`
- `fusion_layers=1`
- `dropout=0.1`
- `optimizer=Adam`
- `lr=1e-3`
- `weight_decay=1e-4`
- `batch_size=64`
- `epochs=30`
- `seed=42`

### 数据划分与预处理

- 数据集：Houston 2013 DFTC 官方数据。
- 划分协议：`official train -> spatial train/val split + buffer`，官方 `test` 仅用于最终评估。
- 验证集选择：按 `val_oa` 选择 best checkpoint，不再使用 test 指标选模型。
- 预处理：`preprocess_scope=train`，仅使用训练子集统计量做 z-score 与 PCA 拟合，避免数据泄漏。
- 默认空间隔离参数：`val_spatial_buffer = patch_size // 2 = 5`。

### 当前最优结果

当前主结果来自 `results/ablation_full/hct_bgc_v1/best_metrics.json`：

- Best epoch: `13`
- Val OA / AA / Kappa: `0.9170 / 0.9320 / 0.9093`
- Test OA / AA / Kappa: `0.8596 / 0.8717 / 0.8480`

### 当前消融结论

- `HCT-BGC-v1` 优于 `Baseline-CNN`：
  - Test OA: `0.8434 -> 0.8596`
  - Test Kappa: `0.8303 -> 0.8480`
- `fusion_layers=1` 最优，继续堆叠到 `2/3` 层会降低测试集表现。
- `gate` 有效，关闭门控后测试集性能下降。
- 当前这版 `contrastive alignment` 在验证集较强，但未提升测试集泛化，暂不作为主结果配置。

### 已知问题

- Apple Silicon 上 `Conv3d`、`MultiheadAttention`、`TransformerEncoder` 路径对 MPS 不稳定，脚本会自动回退到 CPU。
- 当前 contrastive 分支仍需进一步调参和约束设计，暂不稳定。
- 目前分类图和 gate 统计素材已能导出，但论文最终配色和排版仍需后期统一。
## 2026/3/16进度日志
- 完成三组实验：patch size（9、11、13）；pca（20，30，40）；embed_dim（64、96、128）
- 生成 mean+-std 图表，得到 patch size = 11、pca = 30、embed_dim = 128 的最优组合
## 2026/3/17进度日志
- 修改 evaluate.py ，输出每一类的评估的准确性对比（Per-Class Accuracy + Confusion Matrix）
- 修改 visualize_map.py，生成升级版分类图
- 整理实验表 + 写week 3实验小节：本周实验结果总结在 week3_summary.md 中