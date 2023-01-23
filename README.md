# mcm2022C
## using for mcm2022c learning
### hth
####  pandas 处理csv文件详解https://zhuanlan.zhihu.com/p/340441922
####  arima时序预测 https://github.com/3030712382/2022-C-/blob/main/ARIMA%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E7%BE%8E%E8%B5%9B%E9%A2%84%E6%B5%8B.ipynb
```python
import matplotlib.pyplot as plt
2 import torch
3 import torch.nn as nn
4 import torch.nn.functional as F
5 from torch.utils.data import TensorDataset, DataLoader
6 import torch.optim as optim
7 from math import *
8 import pandas as pd
9 import numpy as np
10 import sys
11 import math
12 import random
13 # define the parameters
14 num_layers = 1
15 rnn_unit = 16
16 input_size = 3
17 output_size = 1
18 batch_size = 4
19 time_step1 = 7
20 time_step2 = 5
21 epochs = 200
22 lr = 0.001
23 SEED = 0
24 torch.manual_seed(SEED)
25 torch.cuda.manual_seed(SEED)
26 def getData(df, begin, end, time_step):
27 columns_list = [’RSI’,’MACD’,’Price’]
28 df = pd.DataFrame(df, columns=columns_list)
29 data = df.iloc[:, 0:].values
30 data_s = data[begin:end]
31 train_x, train_y = [], []
32 X = []
33 for i in range(len(data_s) − time_step):
34 x = data_s[i:i + time_step, :input_size]
35 y = data_s[i + time_step, −1]
36 train_x.append(x.tolist())
37 train_y.append(y.tolist())
38 train_x = np.array(train_x).reshape((len(data_s) − time_step, time_step, −1))
39 train_y = np.array(train_y).reshape((len(data_s) − time_step, −1))
40 X = torch.from_numpy(np.array(train_x))
41 y = torch.from_numpy(np.array(train_y))
42 return X, y
43 class MyLSTM(nn.Module):
44 def __init__(self):
45 super().__init__()
46 self.i_h = nn.Linear(input_size, input_size)
47 self.h_h = nn.LSTM(input_size, rnn_unit, 1, batch_first=True)
48 self.h_o = nn.Linear(rnn_unit, 1)
Team # 2200912 Page 20 of 22
49 def forward(self, x):
50 x = self.i_h(x)
51 x = F.relu(x)
52 h0 = torch.randn(1, x.shape[0], rnn_unit).float()
53 c0 = torch.randn(1, x.shape[0], rnn_unit).float()
54 _, (h_n, c_n) = self.h_h(x, (h0, c0))
55 x = h_n.view(x.shape[0], −1)
56 x = self.h_o(x)
57 return x
```
####  lstm
```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from math import *
import pandas as pd
import numpy as np
import sys
import math
import random

num_layers = 1
rnn_unit = 16
input_size = 1
output_size = 1
batch_size = 4
time_step1 = 7
time_step2 = 5
epochs = 200
lr = 0.001
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



df = pd.read_csv('BCHAIN-MKPRU.csv' ,index_col=['Date'], parse_dates=True)
train_x, train_y = [], []
X = []
x ,y = [],[]
time_step = 7
data_p = df.iloc[:, 0:].values[0:]
data_len = len(data_p)
t = np.linspace(0, data_len, data_len + 1)
data_max = data_p.max()
data_min = data_p.min()
data_s = df.iloc[:, 0:].values[0:1400]
t_for_training = t[7:1400]
t_for_testing = t[1400:data_len-7]
data_s = (data_s-data_min) / (data_max- data_min)
for i in range(len(data_s) - time_step):
    x = data_s[i:i + time_step, :input_size]
    y = data_s[i + time_step, -1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
train_x = np.array(train_x).reshape((len(data_s) - time_step, time_step, -1))
train_y = np.array(train_y).reshape((len(data_s) - time_step, -1))
X = torch.from_numpy(np.array(train_x))
y = torch.from_numpy(np.array(train_y))

class MyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Linear(input_size, input_size)
        self.h_h = nn.LSTM(input_size, rnn_unit, 1, batch_first=True)
        self.h_o = nn.Linear(rnn_unit, 1)
    def forward(self, x):
        #x = self.i_h(x)
        #x = F.relu(x)
        h0 = torch.randn(1, x.shape[0], rnn_unit).float()
        c0 = torch.randn(1, x.shape[0], rnn_unit).float()
        _, (h_n, c_n) = self.h_h(x, (h0, c0))
        x = h_n.view(x.shape[0], -1)
        x = self.h_o(x)
        return x

LSTM = MyLSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(LSTM.parameters(), lr=1e-3)
for epoch in range(1393):
    X = X.to(torch.float32)
    y = y.to(torch.float32)
    output = LSTM(X)
    loss = loss_function(output, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if loss.item() < 1e-6:
        print('Epoch [{}/1400], Loss: {:.5f}'.format(epoch+1, math.sqrt(loss.item())))
        print("The loss value is reached")
        break
    elif (epoch+1) % 100 == 0:
        print('Epoch: [{}/1400], Loss:{:.5f}'.format(epoch+1, math.sqrt(loss.item())))

pred_y_for_train = LSTM(X)
pred_y_for_train = pred_y_for_train.view(-1, 1).data.numpy()

train_x1=[]
train_y1=[]
data_s = df.iloc[:, 0:].values[1400:]
data_s = (data_s-data_min) / (data_max- data_min)
for i in range(len(data_s) - time_step):
    x = data_s[i:i + time_step, :input_size]
    y = data_s[i + time_step, -1]
    train_x1.append(x.tolist())
    train_y1.append(y.tolist())
train_x1 = np.array(train_x1).reshape((len(data_s) - time_step, time_step, -1))
train_y1 = np.array(train_y1).reshape((len(data_s) - time_step, -1))
X = torch.from_numpy(np.array(train_x1))
y = torch.from_numpy(np.array(train_y1))
X = X.to(torch.float32)
y = y.to(torch.float32)
LSTM = LSTM.eval()
pred_y_for_test = LSTM(X)
pred_y_for_test = pred_y_for_test.view(-1, 1).data.numpy()

loss = loss_function(torch.from_numpy(pred_y_for_test), y)
print("test loss：", loss.item())


plt.figure()
plt.plot(t_for_training, train_y, 'b', label='y_trn')
plt.plot(t_for_training, pred_y_for_train, 'y--', label='pre_trn')

plt.plot(t_for_testing, train_y1, 'k', label='y_tst')
plt.plot(t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

plt.xlabel('t')
plt.ylabel('Vce')
plt.show()
```
### xy
### zjy
