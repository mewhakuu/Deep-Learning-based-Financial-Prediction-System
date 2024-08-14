# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved
@Time ： 2024/5/26 上午11:48
@Author ： mewhaku
@File ：feature_selection.py
@IDE ：PyCharm
"""
import streamlit as st
import os
from process import selection
from methods.data_cleansing import data_cleansing
import pandas as pd
import numpy as np
from scipy.io import loadmat

# 设置页面标题
st.title('特征选择')

# 上传文件
# 创建一个文件夹用于保存上传的文件
if not os.path.exists("dataset"):
    os.makedirs("dataset")

    # 选择数据来源
data_source = st.radio("选择数据来源", ('上传文件', '选择系统数据集'))

# 根据选择的数据来源进行操作
if data_source == '上传文件':
    # 显示文件上传器
    uploaded_file = st.file_uploader("选择一个文件", type=["csv", "txt", "json", "mat", "npy"])
    # 当文件被上传时
    if uploaded_file is not None:
        # 获取文件名和扩展名
        file_name = uploaded_file.name
        extension = file_name.split('.')[-1]
        # 根据文件扩展名保存文件
        if extension in ["csv", "txt", "json", "mat", "npy"]:
            file_path = os.path.join("dataset", file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"文件已保存为: {file_path}")
        else:
            st.error("不支持的文件类型。请上传 txt/csv/json/mat/npy 文件。")

        # 根据文件扩展名处理文件
        if extension == 'txt':
            # 读取 txt 文件并展示内容
            dot = ','
            content = uploaded_file.getvalue().decode("utf-8")
            st.text_area("TXT 文件内容", content, height=300)
        elif extension == 'csv':
            # 读取 csv 文件并展示内容
            dot = ';'
            df = pd.read_csv(uploaded_file)
            df = data_cleansing(df)
            st.success("已清洗")
            original_filename = uploaded_file.name
            path = "dataset/" + original_filename
            # 将清洗后的DataFrame保存为CSV文件，文件名与上传的文件相同
            df.to_csv(path, index=False)
            st.dataframe(df)
        elif extension == 'json':
            st.success("已读取")
        elif extension == 'mat':
            # 加载MATLAB文件
            mat_data = loadmat(uploaded_file)
            # 显示MATLAB文件中的数据
            st.subheader("MATLAB文件内容")
            for key, value in mat_data.items():
                st.write(f"变量名: {key}")
                if isinstance(value, (list, tuple)):
                    # 如果是数组或矩阵，将其转换为Pandas DataFrame
                    df = pd.DataFrame(value)
                    st.write(df)
                else:
                    # 否则直接显示值
                    st.write(value)
        elif extension == 'npy':
            # 读取 npy 文件并展示内容
            npy_data = np.load(file_path)
            # 将 NumPy 数组转换为 Pandas DataFrame
            df = pd.DataFrame(npy_data)
            # 展示表格
            st.dataframe(df)
        else:
            st.error("不支持的文件类型。请上传 txt/csv/json/mat/npy 文件。")
elif data_source == '选择系统数据集':
    # 列出/dataset/目录下的所有文件
    dataset_path = 'dataset/'
    available_datasets = os.listdir(dataset_path)
    # 让用户从列表中选择数据集
    selected_dataset = st.selectbox("选择一个数据集", available_datasets)
    file_path = os.path.join(dataset_path, selected_dataset)
    extension = selected_dataset.split('.')[-1]
    # 根据文件扩展名处理文件
    if extension == 'txt':
        # 读取 txt 文件并展示内容
        dot = ','
        with open(file_path, "r") as file:
            content = file.read()
            st.text_area("TXT 文件内容", content, height=300)
    elif extension == 'csv':
        # 读取 csv 文件并展示内容
        dot = ';'
        df = pd.read_csv(file_path)
        st.dataframe(df)
    elif extension == 'json':
        st.success("已读取")
    elif extension == 'mat':
        # 加载MATLAB文件
        mat_data = loadmat(file_path)
        # 显示MATLAB文件中的数据
        st.subheader("MATLAB文件内容")
        for key, value in mat_data.items():
            st.write(f"变量名: {key}")
            if isinstance(value, (list, tuple)):
                # 如果是数组或矩阵，将其转换为Pandas DataFrame
                df = pd.DataFrame(value)
                st.write(df)
            else:
                # 否则直接显示值
                st.write(value)
    elif extension == 'npy':
        # 读取 npy 文件并展示内容
        npy_data = np.load(file_path)
        # 将 NumPy 数组转换为 Pandas DataFrame
        df = pd.DataFrame(npy_data)
        # 展示表格
        st.dataframe(df)

# 下拉选择框，用户选择执行的脚本
script_option = st.selectbox(
    '选择一个脚本执行',
    ('dimension_reduction.py', 'fcbf.py', 'pehar_fselection.py')
)

# 执行按钮
if st.button('执行降维算法'):
    # 执行降维算法
    script_name = script_option if script_option else ""
    selection(file_path, script_name)

    # 获取结果文件夹路径
    data_name = file_path.split('/')[-1].split('.')[0]
    output_directory = 'results/selection/' + data_name + "/"

    # 列出所有结果文件并提供下载按钮
    if os.path.exists(output_directory):
        files = os.listdir(output_directory)
        for file in files:
            file_path = os.path.join(output_directory, file)
            with open(file_path, 'rb') as f:
                st.download_button(f'下载 {file}', f, file_name=file)
