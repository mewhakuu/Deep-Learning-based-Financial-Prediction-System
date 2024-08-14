# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved 
@Time ： 2024/5/13 下午2:31
@Author ： mewhaku
@File ：stock_data_geter.py
@IDE ：PyCharm
"""
import akshare as ak
import os

stock_code = '600818'
# 创建输入框让用户输入开始时间
start_date = '20010101'
# 创建输入框让用户输入结束时间，默认值为今天
end_date = '20240520'
df = ak.stock_zh_a_hist(symbol=stock_code,period='daily', start_date=start_date, end_date=end_date)

print(df)

if not df.empty:
    # 确保dataset目录存在
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    file_path = os.path.join('dataset', f'{stock_code}_{start_date}_{end_date}.csv')
    df.to_csv(file_path, index=False)

