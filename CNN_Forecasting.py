#%%
import math

import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from My_utils.preprocess_data import divide_data
from My_utils.evaluation_scheme import evaluation
#%%
class CNNnetwork(nn.Module):
    def __init__(self):
        super(CNNnetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 35, 640)
        self.fc2 = nn.Linear(640, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(x.size(0),-1)   #batch_size
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#%% import data
df = pd.read_csv(r'C:\Users\admin\Desktop\My master-piece_LOL\data\PeMS.csv', parse_dates = True)    #前面加r避免地址识别错误
df.head()  # 观察数据集，这是一个单变量时间序列
TF=df['767838'].values.reshape(-1, 1)
# plt.figure(figsize=(12, 8))
# plt.grid(True)  # 网格化
# plt.plot(TF)
# plt.show()
#%% divide data into train set and test set
# Normalization
Normalization = preprocessing.MinMaxScaler()
Norm = Normalization.fit_transform(TF)       #

# divide data
trainX, trainY, testX, testY = divide_data(data=Norm, train_size=int(len(Norm) * 0.8), seq_len=72, pre_len=1)  #转置为列向量
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

train_dataset = Data.TensorDataset(trainX, trainY)
test_dataset = Data.TensorDataset(trainX, trainY)

#put into loader
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=144, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=144, shuffle=False)
#%%
device = torch.device("cuda")
CNN_model = CNNnetwork().to(device)

loss_func= nn.MSELoss()
optimizer = torch.optim.Adam(CNN_model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

#%% training
batch_size = 144
epochs=200                    ###效果不好就epoch不够多lol
CNN_model.train()

best_val_loss = float("inf")
best_model = None
train_loss_all = []
total_loss = 0.

for epoch in range(epochs):
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pre_y = CNN_model(x)
        loss = loss_func(pre_y, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        train_num += x.size(0)

        total_loss += loss.item()
        log_interval = int(len(trainX) / batch_size / 5)
        if (step + 1) % log_interval == 0 and (step + 1) > 0:
            cur_loss = total_loss / log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, (step + 1), len(trainX) // batch_size, scheduler.get_lr()[0],
                cur_loss, math.exp(cur_loss)))
            total_loss = 0

    if (epoch + 1) % 5 == 0:
        print('-' * 89)
        print('end of epoch: {}, Loss:{:.7f}'.format(epoch + 1, loss.item()))
        print('-' * 89)
        train_loss_all.append(train_loss / train_num)

    if train_loss < best_val_loss:
       best_val_loss = train_loss
       best_model = CNN_model

    scheduler.step()

#%%
# 预测
best_model = best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device))
Norm_pred = pred.data.cpu().numpy()   # f=放回cpu里，转换成一维的ndarray数据，这是预测值

# 反归一
simu = Normalization.inverse_transform(Norm_pred.reshape(-1,1))
real = Normalization.inverse_transform(testY[:, :, -1].data.numpy())
#%%
# 误差指标
MAE, RMSE, MAPE, R2 = evaluation(real, simu)
print('\n ', "MAE:", MAE, '\n ', "RMSE:", RMSE, '\n ', "MAPE:", MAPE, '\n ', "R2:", R2)
# 画图
plt.plot(simu[0:24*12], 'r', label='prediction')
plt.plot(real[0:24*12], 'b', label='ground_truth')
plt.legend(loc='best')
plt.show()




