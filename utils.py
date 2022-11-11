import numpy as np
from math import sqrt
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


###### Dataset ######
def preprocess_data(data, rate, seq_len, pre_len):  # 输入参数分别为原始数据(col)，划分比例，输入长度（特征纬度），预测长度
    train_size = int(len(data) * rate)
    train_data = data[0:train_size, :]
    test_data = data[train_size:len(data), :]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(np.size(train_data, 0) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len, :]
        trainX.append(a[0: seq_len, :])
        trainY.append(a[seq_len: seq_len + pre_len, :])
    for i in range(np.size(test_data, 0) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len, :]
        testX.append(b[0: seq_len, :])
        testY.append(b[seq_len: seq_len + pre_len, :])
    return trainX, trainY, testX, testY


def evaluation(real_p,sim_p):
    MAE = mean_absolute_error(real_p, sim_p)
    RMSE = sqrt(mean_squared_error(real_p, sim_p))
    MAPE = np.mean(np.abs((real_p - sim_p) / real_p)) * 100
    R2 = r2_score(real_p, sim_p)
    return MAE, RMSE, MAPE, R2



