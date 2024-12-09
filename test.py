import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from asa import ArSentiment
import numpy as np
from hyperdash import Experiment
import calendar;
import time;
import warnings
import arabic_reshaper
from bidi.algorithm import get_display


warnings.filterwarnings("ignore")

input_dim = 300
hidden_dim = 1
n_layers = 1
batch_size = 10
seq_len = 100




torch.manual_seed(1)

class LSTMTagger(nn.Module):
    def __init__(self):
        super(LSTMTagger, self).__init__()
        self.input_dim = 300
        self.hidden_dim = 6
        self.n_layers = 2
        self.seq_len = 70
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, inp,hidden):
        out, hidden = self.lstm_layer(inp, hidden)
        x = out[:,-1,:]
        x = self.flatten(x)
        y = self.out(x)
        tag_scores = F.sigmoid(y)
        return tag_scores
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

model = LSTMTagger()
model.load_state_dict(torch.load('m2/state_dict_final3.pt'))
# loss_function = nn.BCELoss()
# from adabelief_pytorch import AdaBelief
# optimizer = AdaBelief(model.parameters(), lr=1e-6, eps=1e-8, betas=(0.9,0.999))

# batch_size = 32

data = ArSentiment("c:/Users/aly17/Desktop/arProject/iot/embeddings/arabic-news.bin","c:/Users/aly17/Desktop/arProject/iot/datasets/test.csv",batch=70)
x_train,x_test,y_train,y_test,txt1,txt2 = data.get()
x_train = np.concatenate((x_train,x_test))
y_train = np.concatenate((y_train,y_test))
txt = np.concatenate((txt1,txt2))


train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, shuffle=False, batch_size=x_train.shape[0])
h = model.init_hidden(x_train.shape[0])
h = tuple([e.data for e in h])
model.eval()
w = []
l = []
for sentence, tags in train_loader:
    tag_scores = model(sentence,h)
    z = np.where(tag_scores >= 0.5,'ايجابي','سلبي')
    for i in range(tags.shape[0]):
        w.append(txt[i])
        l.append(z[i])
        
import pandas as pd
new_df = pd.DataFrame({'txt':w,'label':l}) 
new_df.to_csv('res.csv',encoding='utf-8-sig')
