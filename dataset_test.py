# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved 
@Time ： 2024/5/13 下午5:33
@Author ： mewhaku
@File ：dataset_test.py
@IDE ：PyCharm
"""
import numpy as np
import torch
torch.manual_seed(10)
np.random.seed(10)

import pandas as pd

path = "dataset/000858.csv"


data = [pd.read_csv(path).values[5000:6000].reshape(-1, 1),
                pd.read_csv(path).values[4000:5000].reshape(-1, 1)]
print(data)