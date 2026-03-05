from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Tuple

import numpy as np
import torch

from dataset.mat_loader import load_houston_hl  
from dataset.preprocessing import preprocess_hsi_lidar, pad_reflect
from dataset.patch_dataset import bulid_index, HsiLidarPatchDataset