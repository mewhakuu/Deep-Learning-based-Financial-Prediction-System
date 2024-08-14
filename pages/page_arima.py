# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved
@Time ： 2024/5/31 下午9:16
@Author ： mewhaku
@File ：page_arima.py
@IDE ：PyCharm
"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import streamlit as st
import os
from scipy.io import loadmat
import numpy as np
from sklearn import metrics # 计算指标
# 忽略可能出现的警告信息
warnings.filterwarnings("ignore")


# Custom theme configuration
st.set_page_config(layout="centered", page_title="在线金融时序预测系统")

# Streamlit应用程序
def main():
    # 添加CSS动画背景
    st.markdown("""
           <style>
               @keyframes gradientBackground {
                   0% {
                       background-color: #ee7752;
                   }
                   25% {
                       background-color: #e73c7e;
                   }
                   50% {
                       background-color: #23a6d5;
                   }
                   75% {
                       background-color: #23d5ab;
                   }
                   100% {
                       background-color: #ee7752;
                   }
               }

               body {
                   animation: gradientBackground 10s ease infinite;
                   background-size: cover;
                   margin: 0;
                   height: 100vh;
                   display: flex;
                   justify-content: center;
                   align-items: center;
               }
           </style>
       """, unsafe_allow_html=True)
    st.title('ARIMA模型在线预测')

    # 上传文件
    # 创建一个文件夹用于保存上传的文件
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

        # 选择数据来源
    data_source = st.radio("选择数据来源", ('选择系统数据集', '上传文件'))

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

    # 用户输入数据
    num = st.text_input('请输入数据量(默认500)：', '500')
    train = st.text_input('请输入训练集开始的地方：', '5000')
    test = st.text_input('请输入测试集开始的地方', '5500')


    # 当用户点击预测按钮时
    if st.button('预测'):
        # 这里应该有数据预处理的代码
        #feature_selection = st.text_input('请输入想要使用的数据预测方法：')
        # 例如：data_to_predict = preprocess(user_input)

        # GP251
        #数据分割
        path = file_path

        data = pd.read_csv(path)
        data_train = data.iloc[int(train):int(train) + int(num)*2].values.reshape(-1, 1)
        data_test = data.iloc[int(test):int(test) + int(num)].values.reshape(-1, 1)
        predictions = []

        # 滚动预测
        for i in range(500):
            # 使用当前窗口的数据作为训练集
            train_set = data_train[i:i + 500]
            model = ARIMA(train_set, order=(12, 1, 2))  # 使用合适的p, d, q值
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)


        """# 创建ARIMA模型
        model = ARIMA(data_train, order=(12, 1, 2))  # ARIMA模型参数可能需要根据数据进行调整
        fitted_model = model.fit()
        # 进行预测
        # 使用 get_prediction 来进行预测，从训练集的第一个点开始
        predictions = fitted_model.get_prediction(start=500, end=int(num)*2-1, dynamic=False)
        predicted_values = predictions.predicted_mean"""

        # 展示预测图片
        # Display the model's state plot
        st.subheader("预测结果")
        st.header("模型预测结果")
        # Display model metrics here

        # 显示动态图表
        y_pred = np.array(predictions).reshape(-1, 1)
        data_test = np.array(data_test).reshape(-1, 1)

        data_test_df = pd.DataFrame(data_test, columns=['测试数据'])
        y_pred_df = pd.DataFrame(y_pred, columns=['预测结果'])
        # 将两个DataFrame合并为一个，沿着列的方向
        combined_df = pd.concat([data_test_df, y_pred_df], axis=1)
        chart_data = pd.DataFrame(combined_df, columns=["测试数据", "预测结果"])
        st.line_chart(chart_data,color=["#3399FF", "#FF3366"] )


        st.header("详细数据")
        MSE = metrics.mean_squared_error(data_test, y_pred)
        RMSE = metrics.mean_squared_error(data_test, y_pred) ** 0.5
        MAE = metrics.mean_absolute_error(data_test, y_pred)
        R2 = metrics.r2_score(data_test, y_pred)

        st.markdown(f"#### MSE: {MSE}")
        st.markdown(f"#### RMSE: {RMSE}")
        st.markdown(f"#### MAE: {MAE}")
        st.markdown(f"#### R2: {R2}")





if __name__ == '__main__':
    main()


