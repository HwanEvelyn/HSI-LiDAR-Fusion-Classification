# HSI + LiDAR Fusion Classification
## 2026/3/5 进度日志
- 完成数据pipeline的搭建:dataset/*、scripts/check_pipeline.py
- pipe1.读取 data_hl.mat文件，提取HSI、LiDAR和GT数据
- pipe2.Norm → PCA → Patch 
- pipe3.划分训练样本列表 + patch数据集 √
- 完成数据处理pipeline的检查
