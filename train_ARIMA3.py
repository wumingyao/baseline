# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 17:02
# @Author  : unnoyy
# @File    : train_ARIMA3.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa as smt
import sklearn.metrics as skm

from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

import warnings
warnings.filterwarnings("ignore")

def to_series(data):
    '''
    该函数将多变量序列转化成单变量序列
    '''
    # series = [week[:, 0] for week in data]
    series = np.array(data).flatten()
    return series

def plot_acf_pacf(series, lags):
    '''
    该函数实现绘制acf图和pacf图
    '''
    plt.figure(figsize=(16, 12), dpi=150)
    axis = plt.subplot(2, 1, 1)
    sm.graphics.tsa.plot_acf(series, ax=axis, lags=lags)

    axis = plt.subplot(2, 1, 2)
    sm.graphics.tsa.plot_pacf(series, ax=axis, lags=lags)

    plt.tight_layout()
    plt.show()

def plot(datas):
    fig = plt.figure(1, figsize=[12, 4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    data = np.array(to_series(datas))
    autocorr = acf(data)
    pac = pacf(data)

    x = [x for x in range(len(pac))]
    ax1.plot(x[1:], autocorr[1:])
    ax1.grid(True)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')

    ax2.plot(x[1:], pac[1:])
    ax2.grid(True)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    plt.show()

def ARIMAmodel(s,timeslices):
    # print(s.shape)
    s=s.reshape((1*timeslices,560))
    series = to_series(s)
    if np.all(series == 0):
        return np.zeros(timeslices*560)
    model = smt.arima_model.ARIMA(series, order=(7, 0, 0))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(series), len(series) + (timeslices)*560-1)
    # yhat = model_fit.predict((len(series), len(series) + timeslices-1),560)
    return yhat

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        # print(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def evaluation(y_true, y_pred):
    # MSE = np.mean(np.square(y_true - y_pred))
    RMSE = np.sqrt(np.mean(np.square(y_true - y_pred)))
    MAE = np.mean(np.abs(y_true - y_pred))
    MAPE = masked_mape_np(y_true, y_pred,y_true.all())
    # MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return MAE,RMSE,MAPE

if __name__ == '__main__':
    # dataset = np.load('dataset/way_volume_560.npy')  ##(34560, 560, 1)
    dataset = np.load('/public/lhy/wmy/dataset/Taian/slice_5min_month_2_3_4_5/way_volume_560_acc.npy')  ##(34560, 560, 1)
    datas = np.squeeze(dataset, axis=2)  ## (34560, 560)
    data = datas[:16992]  ## (16992, 560)
    plot(data[:12 * 8, 0:560])

    # data为按时间片的统计数据，shape为(16992, 560)
    k = 2  ## 训练的时间片 12 --> 12
    timeslics =3  ## 12, 6, 3

    size = int(data.shape[0] / timeslics)  ## 1416/12
    s = []
    for i in range(size):
        s.append(data[timeslics * i:timeslics * (i + 1)])
    s = np.array(s)  ## (1416, 12, 560)
    batchsize = int(s.shape[0] / k)  ## 1416/8 = 177 ; 1416/4=354; 1416/2=708
    mae, rmse, mape = 0, 0, 0

    for i in range(batchsize):
        predict = ARIMAmodel(s[k * i:k * (i + 1) - 1], timeslics)
        truth = s[k * (i + 1) - 1]
        truth = to_series(truth)
        MAE, RMSE, MAPE = evaluation(truth, predict)
        print("MAE:", MAE)
        print("RMSE:", RMSE)
        print("MAPE:", MAPE)
        print("===============================")

        mae += MAE
        rmse += RMSE
        mape += MAPE
    mae = mae / batchsize
    rmse = rmse / batchsize
    mape = mape / batchsize
    print("MAE Average:", mae)
    print("RMSE Average:", rmse)
    print("MAPE Average:", mape)
