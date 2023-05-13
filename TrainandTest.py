import math
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from Transformer import Transformer
from embedding import TokenEmbedding


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
predict_period_num=5

# Parameter for mdoel
## Learning Rate
#lr=0.00005
## Epoch Number
#epochs=1000
## Mini-Batch size
batch_size=64
## How many epochs to stop train if valid loss is not decreasing
patience=20

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
  label=df.iloc[i+observation_period_num:i+observation_period_num+predict_period_num,4].values
  src_data.append(data)
  tgt_data.append(label)
src_data = torch.tensor(np.array(src_data)).float()
tgt_data = torch.tensor(np.array(tgt_data)).float()

dataset = torch.utils.data.TensorDataset(src_data, tgt_data)

# 各データセットのサンプル数を決定
# train : val: test = 60%　: 20% : 20%
n_train = int(len(dataset) * 0.5)
n_val = int(len(dataset) * 0.1)
n_test = int(len(dataset) * 0.1)
no_use= len(dataset) - n_train - n_val - n_test

train, val, test, _ = torch.utils.data.random_split(dataset, [n_train, n_val, n_test, no_use])
print('train_len', len(train))
print('val_len', len(val))
print('test_len', len(test))
train_loader = torch.utils.data.DataLoader(train, batch_size)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

def generate_square_subsequent_mask(seq_len, batch_size):
        mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
        mask = torch.where(mask==1, True, False)
        mask = mask.repeat(batch_size, 1, 1)
        return mask


def train(model, data_provider, optimizer, criterion):
  model.train()
  total_loss = []
  i = 0
  for src, tgt in data_provider:
    src = torch.unsqueeze(src, dim=-1)
    tgt = torch.unsqueeze(tgt, dim=-1)
    src = src.float().to(device)
    tgt = tgt.float().to(device)

    input_tgt = torch.cat((src[:,-1:,:],tgt[:,:-1,:]), dim=1)

    output = model(src=src, tgt=input_tgt)
    optimizer.zero_grad()

    loss = criterion(output, tgt)

    loss.backward()
    total_loss.append(loss.cpu().detach())
    optimizer.step()

  return np.average(total_loss)

def evaluate(flag, model, data_provider, criterion):
  model.eval()
  total_loss = []
  result=torch.Tensor(0)
  actual=torch.Tensor(0)

  for src, tgt in data_provider:
    src = torch.unsqueeze(src, dim=-1)
    tgt = torch.unsqueeze(tgt, dim=-1)
    src = src.float().to(device)
    tgt = tgt.float().to(device)
    batch_size = src.size(0)

    seq_len_src = src.shape[1]
  
    mask_src = (torch.zeros(seq_len_src, seq_len_src)).type(torch.bool)
    mask_src = mask_src.repeat(batch_size, 1, 1)

    memory = model.encode(src, mask_src)

    outputs = src[:,-1:,:]

    seq_len_tgt = tgt.shape[1]

    for i in range(seq_len_tgt-1):
      mask_tgt = (generate_square_subsequent_mask(outputs.size(1), batch_size)).to(device)
      
      output = model.decode(outputs, memory, None, mask_tgt)
      output = model.linear(output)

      outputs = torch.cat([outputs, output[:,-1:,:]], dim=1)

    loss = criterion(outputs, tgt)
    total_loss.append(loss.cpu().detach())

    result=torch.cat((result, output[-1].view(-1).cpu()),0)
    actual=torch.cat((actual,tgt[-1].view(-1).cpu()),0)

  result = result.to('cpu').detach().numpy().copy()
  actual = actual.to('cpu').detach().numpy().copy()
  print('result: ', result)
  print('actual: ', actual)
  if flag=='test':
    plt.plot(actual,color='red',alpha=0.7)
    plt.plot(result,color='black',linewidth=1.0)
    plt.show()

  return np.average(total_loss)


d_model = 512
nhead = 8
d_ff = 2048
N = 6
dropout = 0.01
epochs = 10
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

optimizer = torch.optim.RAdam(model.parameters(), lr=0.0001)

valid_losses = []
for epoch in range(1, epochs + 1):
    
    loss_train = train(
        model=model, data_provider=train_loader, optimizer=optimizer, criterion=criterion
    )
        
    loss_valid = evaluate(
        flag='val', model=model, data_provider=val_loader, criterion=criterion
    )
    
    #if epoch%10==0:
    print('[{}/{}] train loss: {:.2f}, valid loss: {:.2f}'.format(
            epoch, epochs,
            loss_train, loss_valid,
        ))
        
    valid_losses.append(loss_valid)
    
    if best_loss > loss_valid:
        best_loss = loss_valid
        best_model = model

evaluate(flag='test', model=best_model, data_provider=test_loader, criterion=criterion)  
