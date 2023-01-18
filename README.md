# mcm2022C
## using for mcm2022c learning
### hth
####  pandas 处理csv文件详解https://zhuanlan.zhihu.com/p/340441922
####  arima时序预测 https://github.com/3030712382/2022-C-/blob/main/ARIMA%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E7%BE%8E%E8%B5%9B%E9%A2%84%E6%B5%8B.ipynb
    '''python
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
'''
### xy
### zjy
