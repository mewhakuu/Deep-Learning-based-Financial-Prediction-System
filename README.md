# 介绍

这个项目是我的大学毕业设计，题目为 基于深度学习的金融数据预测系统设计与实现。此仓库包含了系统的所有源码，通过anaconda安装所有依赖包并部署到linux环境启动streamlit包进行使用。本系统还包含了数据数据库组件，请自行配置进行使用，数据库相关接口数据请在web文件夹下的.php文件中查看。祝本仓库能给您带来灵感！

# 核心环境

Anaconda

Python 3.10.14

Numpy 1.26.4

Pandas 2.2.2

Torch 2.3.0

Tensorflow 2.16.1

Streamlit 1.35.0

scikit-learn1.35.0

cuda(optional)

推荐使用environments.yml文件进行环境部署,使用命令conda env create -f environment.yml进行环境安装,请自行配置anaconda环境！environments.yml文件中的name和prefix属性请更改成您自己的环境信息。

# 使用方法

将系统源码文件夹内的文件部署到linux环境下通过streamlit run mainpage.py启动streamlit以提供可交互的网页(不嫌麻烦可以部署到windows环境)，nginx和mysql请自行部署。

# 使用到的代码和参考的论文

Hmamouche, Y., Lakhal, L. & Casali, A. A scalable framework for large time series prediction. *Knowl Inf Syst* **63**, 1093–1116 (2021). https://doi.org/10.1007/s10115-021-01544-w

Farnoosh A, Azari B, Ostadabbas S. Deep switching auto-regressive factorization: Application to time series forecasting[C],Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(8): 7394-7403.https://doi.org/10.48550/arXiv.2009.05135

[Hmamouche/Large-Time-Series-Prediction: A framework for predicting large time series (github.com)](https://github.com/Hmamouche/Large-Time-Series-Prediction)

[ostadabbas/DSARF: DSARF: Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting (AAAI2021) (github.com)](https://github.com/ostadabbas/DSARF)

# Description

This project is my university graduation design, titled Deep Learning-based Financial Data Prediction System Design and Implementation. This repository contains all the source code of the system, install all the dependent packages through anaconda and deploy to linux environment to start streamlit package for use. This system also contains a data database component, please configure yourself to use, database related data in the web folder under the .php file to view. Wish this repository can bring you inspiration!

https://github.com/ostadabbas/DSARF)

# Core environment

Anaconda

Python 3.10.14

Numpy 1.26.4

Pandas 2.2.2

Torch 2.3.0

Tensorflow 2.16.1

Streamlit 1.35.0

scikit-learn 1.35.0

cuda(optional)

Recommended to use environments.yml file for environment deployment, use the command conda env create -f environment.yml environment installation, please configure your own anaconda environment! environments.yml file in the name and prefix attributes please change to your own environment! Please change the name and prefix attributes in the environments.yml file to your own environment information.

# Usage

Deploy the files in the system source folder to linux environment by streamlit run mainpage.py to start streamlit to provide interactive web pages (not too much trouble can be deployed to the windows environment), nginx and mysql please deploy by yourself.

# Code used and papers referenced

Hmamouche, Y., Lakhal, L. & Casali, A. A scalable framework for large time series prediction. *Knowl Inf Syst* **63**, 1093–1116 (2021). https://doi.org/10.1007/s10115-021-01544-w

Farnoosh A, Azari B, Ostadabbas S. Deep switching auto-regressive factorization: Application to time series forecasting[C],Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(8): 7394-7403.(https://doi.org/10.48550/arXiv.2009.05135

[Hmamouche/Large-Time-Series-Prediction: A framework for predicting large time series (github.com)](https://github.com/Hmamouche/Large-Time-Series-Prediction)

[ostadabbas/DSARF: DSARF: Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting (AAAI2021) (github.com)](https://github.com/ostadabbas/DSARF)

