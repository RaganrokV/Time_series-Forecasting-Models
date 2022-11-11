
import numpy as np
import torch


def divide_data(data, train_size, seq_len, pre_len):  # 输入参数分别为原始数据(col)，训练数据长，输入长度（特征纬度），预测长度
    #     train_size = int(len(data) * rate)                  #选择训练集尺寸，可按比例也可按周定制
    train_size = train_size
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

    trainX = torch.tensor(np.array(trainX))
    trainY = torch.tensor(np.array(trainY))
    testX = torch.tensor(np.array(testX))
    testY = torch.tensor(np.array(testY))

    return trainX, trainY, testX, testY


