# -*- coding: utf-8 -*-
"""
@Copyright (C) 2024 mewhaku . All Rights Reserved 
@Time ： 2024/6/2 下午5:36
@Author ： mewhaku
@File ：arima_tools.py
@IDE ：PyCharm
"""
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# 加载数据
ts = pd.read_csv('dataset/000858.csv', parse_dates=True)


# 平稳性检验
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

# 检查数据平稳性
stationarity_test = test_stationarity(ts)

# 如果数据不平稳，进行差分
if stationarity_test['p-value'] > 0.05:
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    stationarity_test = test_stationarity(ts_diff)
    d = 1
    print(d)
else:
    d = 0
    print(d)

df = pd.read_csv('dataset/000858.csv', parse_dates=True)

df.plot()
plot_acf(df, lags=50, adjusted=False)
plot_pacf(df, lags=50,  method='ols')
plt.show()
# 显示图表
plt.show()

