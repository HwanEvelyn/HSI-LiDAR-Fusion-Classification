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
