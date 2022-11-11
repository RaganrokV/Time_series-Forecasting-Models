#%%
import pandas as pd
import torch
from torch import nn
from sklearn import preprocessing
import math
import torch.utils.data as Data
from matplotlib import pyplot as plt
from My_utils.evaluation_scheme import evaluation
from My_utils.preprocess_data import divide_data

#%%
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=seq_len,               # 输入纬度   记得加逗号
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True)                  #是否双向Bi-LSTM
        self.out = nn.Linear(hidden_size*2, pre_len)

    def forward(self, x):
        temp, _ = self.lstm(x)
        s, b, h = temp.size()
        temp = temp.view(s * b, h)
        outs = self.out(temp)
        lstm_out = outs.view(s, b, -1)
        return lstm_out
#%%
#数据读取与预处理
data_csv = pd.read_csv(r'data\Roadside data-SH.csv')
data_csv=data_csv.fillna(data_csv.interpolate())
# data = data_csv.values
# data_csv.head()
TS=data_csv['NO(μg/m³)'].values.reshape(-1, 1)          #Norman需要2d data(N,) 所以reshape(-1,1)变成 (n,1)
#%%
# Normalization
Normalization = preprocessing.MinMaxScaler()
Norm_TS = Normalization.fit_transform(TS)       #

# divide data
trainX, trainY, testX, testY = divide_data(data=Norm_TS, train_size=int(len(Norm_TS) * 0.8) , seq_len=12, pre_len=3)  #转置为列向量
trainX, trainY = trainX.transpose(1, 2).float(), trainY.transpose(1, 2).float()
testX, testY = testX.transpose(1, 2).float(), testY.transpose(1, 2).float()

train_dataset = Data.TensorDataset(trainX, trainY)
test_dataset = Data.TensorDataset(trainX, trainY)

#put into liader
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=144, shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=144, shuffle=False)
#%%
# Hyper Parameters
seq_len = 12  #
hidden_size = 64
pre_len = 3
num_layers = 2
lr=0.005
epochs = 100
batch_size = 144
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = LSTM().to(device)
print(lstm_model)


optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
loss_func = nn.MSELoss()
#%%
#  训练
best_val_loss = float("inf")
best_model = None

train_loss_all = []
lstm_model.train()  # Turn on the train mode
total_loss = 0.

for epoch in range(epochs):
    train_loss = 0
    train_num = 0
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pre_y = lstm_model(x)

        loss = loss_func(pre_y, y)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 0.5)  #梯度裁剪，放backward和step直接，小模型可以不考虑用于缓解梯度爆炸
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
       best_model = lstm_model

    scheduler.step()
#%%
# 预测
best_model = best_model.eval()  # 转换成测试模式
pred = best_model(testX.float().to(device))
# print(pred.size())
Norm_pred = pred.data.cpu().numpy()   # f=放回cpu里，转换成一维的ndarray数据，这是预测值

# 反归一
simu = Normalization.inverse_transform(Norm_pred[:,:,-1])
real = Normalization.inverse_transform(testY[:, :, -1].data.numpy())
#%%
# 误差指标
MAE, RMSE, MAPE, R2 = evaluation(real, simu)
print('\n ', "MAE:", MAE, '\n ', "RMSE:", RMSE, '\n ', "MAPE:", MAPE, '\n ', "R2:", R2)
# 画图
plt.plot(simu[0:24*7], 'r', label='prediction')
plt.plot(real[0:24*7], 'b', label='ground_truth')
plt.legend(loc='best')
plt.show()