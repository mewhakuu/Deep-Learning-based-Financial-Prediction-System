# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved 
@Time ： 2024/5/25 下午3:02
@Author ： mewhaku
@File ：stock_data_download.py
@IDE ：PyCharm
"""
import streamlit as st
import akshare as ak
import datetime
import pandas as pd
import os

# 设置页面标题
st.title('股票数据下载器')

# 创建输入框让用户输入股票代码
stock_code = st.text_input('请输入股票代码', '000001')


# 自定义函数，将日期从'YYYY-MM-DD'格式转换为'YYYYMMDD'格式
def format_date(date_str):
    return ''.join(date_str.split('-'))


# 创建输入框让用户输入开始时间
start_date = st.text_input('请输入开始时间', format_date('2020-01-01'))

# 创建输入框让用户输入结束时间，默认值为今天
end_date = st.text_input('请输入结束时间', format_date(str(datetime.date.today())))

# 创建一个按钮，当用户点击时获取数据
if st.button('获取股票数据'):
    # 使用AKShare获取股票数据
    stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date,
                                    adjust="qfq")

    # 如果数据不为空，则保存到CSV文件
    if not stock_data.empty:
        # 确保dataset目录存在
        if not os.path.exists('dataset'):
            os.makedirs('dataset')

        # 使用格式化的日期创建文件名
        formatted_start_date = format_date(start_date)
        formatted_end_date = format_date(end_date)
        file_path = os.path.join('dataset', f'{stock_code}_{formatted_start_date}_{formatted_end_date}.csv')
        stock_data.to_csv(file_path, index=False)
        st.dataframe(stock_data)
        st.success('股票数据已保存到dataset目录并可下载！')
        # 提供下载链接
        with open(file_path, "rb") as fp:
            btn = st.download_button(
                label="点击下载数据",
                data=fp,
                file_name=file_path,
                mime="text/csv",
            )

    else:
        st.error('未找到股票数据，请检查输入的股票代码和日期。')