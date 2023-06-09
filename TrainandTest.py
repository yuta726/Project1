import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from time_series_Transformer import Transformer
from torch.utils.data import DataLoader
from pandas_datareader import data as wb
import yfinance as yfin
yfin.pdr_override()


# Parameter for data
## Security code
stock_code='9984.T'
## Start Date
start_date='2003-1-1'
## End Date
end_date='2022-12-31'
## Split ratio of train data and validation data
train_rate=0.7
## How many business days to see
observation_period_num=60
## How many business days to predict
predict_period_num=1

# Parameter for mdoel
## Learning Rate
lr=0.0001
## Epoch Number
epochs=10
## Mini-Batch size
batch_size=64
## How many epochs to stop train if valid loss is not decreasing
patience=20

d_model = 512
nhead = 8
d_ff = 2048
N = 6
dropout = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data
df=wb.DataReader(stock_code,start=start_date,end=end_date)

# Normalization
mean_list=df.mean().values
std_list=df.std().values
df=(df-mean_list)/std_list

# Array initialization
src_data=[]
tgt_data=[]

# Put data in array
for i in range(len(df)-observation_period_num-predict_period_num):
  data=df.iloc[i:i+observation_period_num,4].values
  label=df.iloc[i+predict_period_num:i+observation_period_num+predict_period_num,4].values
  src_data.append(data)
  tgt_data.append(label)
src_data = torch.tensor(np.array(src_data)).float()
tgt_data = torch.tensor(np.array(tgt_data)).float()

dataset = torch.utils.data.TensorDataset(src_data, tgt_data)

# 各データセットのサンプル数を決定
n_train = int(len(src_data)*0.2)
n_test = int(len(src_data)*0.1) + n_train

train_src = src_data[0:n_train]
train_tgt = tgt_data[0:n_train]
test_src = src_data[n_train:n_test]
test_tgt = tgt_data[n_train:n_test]

train_dataset = torch.utils.data.TensorDataset(train_src, train_tgt) #データセット作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #データローダー作成
test_dataset = torch.utils.data.TensorDataset(test_src, test_tgt) #データセット作成
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #データローダー作成

def generate_square_subsequent_mask(seq_len, batch_size):
        mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
        mask = torch.where(mask==1, True, False)
        mask = mask.repeat(batch_size, 1, 1)
        return mask


def train(model, data_provider, optimizer, criterion):
  model.train()
  total_loss = []

  for src, tgt in data_provider:
    src = torch.unsqueeze(src, dim=-1)
    tgt = torch.unsqueeze(tgt, dim=-1)
    src = src.float().to(device)
    tgt = tgt.float().to(device)

    output = model(src)
    optimizer.zero_grad()

    loss = criterion(output, tgt)

    loss.backward()
    total_loss.append(loss.cpu().detach())
    optimizer.step()

  return np.average(total_loss)


best_loss = float('Inf')
best_model = None

model = Transformer(N=N,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout_rate=dropout, heads_num=nhead
                   )

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

model = model.to(device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

valid_losses = []
for epoch in range(1, epochs + 1):
    
    loss_train = train(
        model=model, data_provider=train_loader, optimizer=optimizer, criterion=criterion
    )

    
    #if epoch%10==0:
    print('[{}/{}] train loss: {:.2f}'.format(
            epoch, epochs, loss_train,
        ))
 

model.eval()
result=torch.Tensor(0)
actual=torch.Tensor(0)

with torch.no_grad():
  for data, target in test_loader:
    data = torch.unsqueeze(data, dim=-1)
    output=model(data)
    result=torch.cat((result, output[:, -1, :].view(-1).cpu()),0)
    actual=torch.cat((actual,target[:, -1].view(-1).cpu()),0)

plt.plot(range(len(df.iloc[0:n_test,4])), df.iloc[0:n_test, 4], label='Correct')
plt.plot(range(n_train, n_train+len(result)), result, label='Predicted')
plt.legend()
plt.show()
