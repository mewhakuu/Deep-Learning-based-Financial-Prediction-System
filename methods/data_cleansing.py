# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved 
@Time ： 2024/5/19 下午12:23
@Author ： mewhaku
@File ：data_cleansing.py
@IDE ：PyCharm
"""
import numpy as np
#df为传入数据结构
def data_cleansing(df):
    # 处理异常值
    # 定义一个函数来识别异常值
    # 处理空值
    # 用平均值填充空值
    df.fillna(df.mean(), inplace=True)
    # 应用异常值检测
    for col in df.columns:
        anomalies = find_anomalies(df[col])
        # 可以选择删除异常值或用其他值替换,这里我们选择用中位数替换异常值
        median_value = df[col].median()
        df[col] = np.where(df[col].isin(anomalies), median_value, df[col])

    return df

def find_anomalies(data):
    anomalies = []
    threshold = 3
    mean_y = np.mean(data)
    std_y = np.std(data)

    for y in data:
        z_score = (y - mean_y) / std_y
        if np.abs(z_score) > threshold:
            anomalies.append(y)
    return anomalies





