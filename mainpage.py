# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved
@Time ： 2024/4/16 下午6:52
@Author ： mewhaku
@File ：mainpage.py
@IDE ：PyCharm
"""
import streamlit as st
import random
import pandas as pd
# CSS 动画
animation_css = """
<style>
    @keyframes typewriter {
        from {
            width: 0%;
        }
        to {
            width: 100%;
        }
    }

    .typewriter-text {
        overflow: hidden;
        white-space: nowrap;
        animation: typewriter 2s steps(30) infinite;
        font-size: 2.5vw; /* 设置字体大小为视口宽度的3% */
    }
</style>
"""

st.markdown(animation_css, unsafe_allow_html=True)

# 动画文本
st.markdown('# **欢迎使用金融数据预测系统**')

st.markdown('### **本系统包含DASRF，LSTM和ARIMA模型，集成了数据获取、数据处理、数据预测和预测结果评估功能。**')


df = pd.DataFrame(
    {
        "模型名称": ["DSARF", "LSTM", "ARIMA"],
        "论文地址": ["https://arxiv.org/abs/2009.05135", "https://ieeexplore.ieee.org/abstract/document/6795963", "https://www.researchgate.net/publication/285902264_ARIMA_The_Models_of_Box_and_Jenkins"],
        "stars": [random.randint(0, 1000) for _ in range(3)],
        "使用历史": [[random.randint(0, 5000) for _ in range(30)] for _ in range(3)],
    }
)
st.dataframe(
    df,
    column_config={
        "name": "App name",
        "stars": st.column_config.NumberColumn(
            "Github Stars",
            help="Number of stars on GitHub",
            format="%d ⭐",
        ),
        "论文地址": st.column_config.LinkColumn("论文地址"),
        "使用历史": st.column_config.LineChartColumn(
            "过去三十天使用历史", y_min=0, y_max=5000
        ),
    },
    hide_index=True,
)

##st.markdown('<p class="typewriter-text">欢迎使用金融时序数据预测系统！</p>', unsafe_allow_html=True)
##st.markdown('<p class="typewriter-text">请在左侧边栏中选择使用的模型！</p>', unsafe_allow_html=True)

