# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved 
@Time ： 2024/4/16 下午8:27
@Author ： mewhaku
@File ：page_lstm.py
@IDE ：PyCharm
"""

from sklearn.preprocessing import MinMaxScaler # 数据归一化
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

import streamlit as st
import matplotlib.pyplot as plt

import torch
from scipy.io import loadmat
torch.manual_seed(10)
np.random.seed(10)

#评价
from sklearn import metrics # 计算指标
import pandas as pd

# Custom theme configuration
st.set_page_config(layout="centered", page_title="在线金融时序预测系统")

# Streamlit interface

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

#模型
def plot_load(model, X_test, y_test, scaler=MinMaxScaler(feature_range=(0, 1)), figure_size=(60,10),path = './plots/'):
    if not os.path.exists(path):
        os.makedirs(path)

    predicted_load = model.predict(X_test)                        # 测试集输入模型进行预测
    if scaler:
        predicted_load = scaler.inverse_transform(predicted_load)  # 对预测数据还原---从（0，1）反归一化到原始范围
        real_load      = scaler.inverse_transform(y_test)# 对真实数据还原---从（0，1）反归一化到原始范围
    else:
        real_load = y_test


    plt.figure(figsize=figure_size)
    # 画出真实数据和预测数据的对比曲线
    idx_locs = [i for i in range(0, 1, max(1 // 5, 1))]
    fig = plt.figure(figsize=(10, 7 / 3 * len(idx_locs)))
    plt.plot(real_load, color='red', label='Load')
    plt.plot(predicted_load, color='blue', label='Predicted Load')
    plt.title('R N N')
    plt.xlabel('Time')
    plt.ylabel('load')
    plt.legend()
    plt.show()
    fig.savefig(path + "prediction_%d.png" , bbox_inches='tight')
    save_path = path + "prediction_%d.png"
    plt.close()

    MSE   = metrics.mean_squared_error(real_load, predicted_load)
    RMSE  = metrics.mean_squared_error(real_load, predicted_load)**0.5
    MAE   = metrics.mean_absolute_error(real_load, predicted_load)
    R2    = metrics.r2_score(real_load, predicted_load)


    return real_load, predicted_load, MSE, RMSE, MAE, R2,save_path

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,:])
    return np.array(dataX),np.array(dataY)

def plot_loss(history, path = './plots/'):
    if not os.path.exists(path):
        os.makedirs(path)
    fig = plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    fig.savefig(path + 'States.png', bbox_inches='tight')

    return path+'States.png'


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
    st.title('LSTM模型在线预测')

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

    # 参数默认值设置

    # 用户输入数据
    data = st.text_input('请输入预测第个数据值：', '14')#14
    num = st.text_input('请输入数据量(默认500)：', '500')
    train = st.text_input('请输入训练集开始的地方：', '5000')
    test = st.text_input('请输入测试集开始的地方：', '5500')
    N_timestamp = st.text_input('请输入N_timestamp值(默认为30，30预测1)：', '30')
    X_dim = st.text_input('请输入X_dim值(默认为1)：', '1')
    Batch_Size = st.text_input('请输入Batch_Size(默认值为1)值：', '1')
    # Add interactive widgets
    st.sidebar.header("选择模型集体参数")
    Epoch = st.sidebar.slider("训练轮数", 0, 50, 100)
    #learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)

    # 当用户点击预测按钮时
    if st.button('预测'):
        # 这里应该有数据预处理的代码
        #feature_selection = st.text_input('请输入想要使用的数据预测方法：')
        # 例如：data_to_predict = preprocess(user_input)

        # 数据分割
        path = file_path

        df_for_training = pd.read_csv(path).values[int(train):int(train) + int(num)].reshape(-1, 1)
        df_for_training = np.array(df_for_training).reshape(-1, 1)

        df_for_testing  = pd.read_csv(path).values[int(test):int(test) + int(num)].reshape(-1, 1)
        df_for_testing = np.array(df_for_testing).reshape(-1, 1)


        scaler = MinMaxScaler(feature_range=(0, 1))
        df_for_training = scaler.fit_transform(df_for_training)
        df_for_testing = scaler.transform(df_for_testing)
        # 模型预测
        trainX, trainY = createXY(df_for_training, int(N_timestamp))
        testX, testY = createXY(df_for_testing, int(N_timestamp))


        # 构建神经网络模型LSTM
        # 输入层
        inputs = Input((int(N_timestamp), int(X_dim)), name='inputs')

        # LSTM
        # 全连接层
        LSTM_1 = LSTM(units=100, activation='relu', dropout=0.1)(inputs)
        dense_2 = Dense(units=50, activation='relu')(LSTM_1)
        dense_3 = Dense(units=24, activation="relu")(dense_2)
        output_4 = Dense(units=1, activation=None)(dense_3)

        # 构造一个新模型
        model = Model(inputs=inputs, outputs=output_4)
        #print(model.summary())

         # 编译
        optimizer = keras.optimizers.Adam(0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        #训练
        history = model.fit(trainX, trainY, epochs=int(Epoch), batch_size=int(Batch_Size), validation_split=0.2)
        # 画图 Loss, result
        img = plot_loss(history)
        #显示预测结果
        real_load, predicted_load, MSE, RMSE, MAE, R2, save_path= plot_load(model, testX, testY, scaler)




        # 展示预测图片
        # Display the model's state plot
        st.header("预测结果")
        st.subheader("模型预测结果")
        # Display model metrics here
        st.image(save_path, caption='predict', use_column_width=True)
        # 显示动态图表
        predicted_load = np.array(predicted_load).reshape(-1, 1)
        real_load = np.array(real_load).reshape(-1, 1)
        data_test_df = pd.DataFrame(real_load, columns=['测试数据'])
        y_pred_df = pd.DataFrame(predicted_load, columns=['预测结果'])
        # 将两个DataFrame合并为一个，沿着列的方向
        combined_df = pd.concat([data_test_df, y_pred_df], axis=1)
        chart_data = pd.DataFrame(combined_df, columns=["测试数据", "预测结果"])
        st.line_chart(chart_data, color=["#3399FF", "#FF3366"])

        # Use columns for layout
        col1, col2 = st.columns(2)
        with col1:
            st.header("状态转换图")
            st.image(img, caption='model_test', use_column_width=True)

        with col2:
            st.header("详细数据")
            # 展示预测结果
            st.markdown(f"#### MSE: {MSE}")
            st.markdown(f"#### RMSE: {RMSE}")
            st.markdown(f"#### MAE: {MAE}")
            st.markdown(f"#### R2: {R2}")




if __name__ == '__main__':
    main()

