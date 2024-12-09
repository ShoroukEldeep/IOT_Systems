import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import random_split
# from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn

import torch.optim as optim


# import torch_xla
# import torch_xla.core.xla_model as xm
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from hyperdash import Experiment
from adabelief_pytorch import AdaBelief

device = torch.device( "cpu") #"cuda:0" if torch.cuda.is_available() else

def get_data(length):
    df = pd.read_csv('Train_Test_IoT_Fridge.csv')
    df['h'] = df.time.apply(lambda x : x.split(':')[0])
    df['m'] = df.time.apply(lambda x : x.split(':')[1])
    df['s'] = df.time.apply(lambda x : x.split(':')[2])

    def temp_change(x):
        if 'low' in x:
            return 0
        else:
            return 1

    df.temp_condition = df.temp_condition.apply(temp_change)
    df.set_index(keys=['date','h'], drop=False,inplace=True)
    fdf = df.drop(['ts','date','time','h','m','s'], axis=1)

    x = []
    y = []
    for index in tqdm(df.index.unique()):
        arr = [[0,0] for i in range(0,length)]
        sub = fdf.loc[index]
        for i in range(1,sub.shape[0]): #
            arr = arr[1:]
            l1 = sub.drop(['label','type'],axis=1).values[i].tolist()
            l2 = sub.drop(['label','type'],axis=1).values[i-1].tolist()
            arr.append([l1[0] - l2[0],l1[1]])
            z = arr
            x.append(z)
            y.append(sub.drop(['fridge_temperature','temp_condition','label'],axis=1).values[i][0])

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(y)).tolist()
    encoded = tf.keras.utils.to_categorical(integer_encoded,7).tolist()

    x_train,x_test,y_train,y_test = train_test_split(x,integer_encoded,shuffle=True,random_state=42,train_size=0.9)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,shuffle=True,random_state=42,train_size=0.9)
    return x_train,x_val,x_test,y_train,y_val,y_test

input_dim = 2
batch_size = 16
num_classes = 7


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        hidden_dim, n_layers, seq_len, dense, dim1, dim2, dim3, drop1, drop2 = 54,1,24,0,433,173 ,38,0 ,18
        super().__init__()
        self.loss = float('inf')
        self.val_loss = float('inf')
        self.input_dim = 2
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.batch_size = 16
        self.seq_len = int(seq_len)
        self.dims = [int(dim1), int(dim2), int(dim3)]
        self.drop1 = float(drop1/100)
        self.drop2 = float(drop2/100)

        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True,dropout=self.drop1).to(device)
        self.fc = []
        self.flatten = nn.Flatten().to(device)
        self.dropout = nn.Dropout(self.drop2).to(device)
        fc_input_dim = self.hidden_dim
        last = fc_input_dim
        for i in range(dense):
            self.fc.append(nn.Linear(fc_input_dim, self.dims[i]).to(device))
            last = self.dims[i]
            fc_input_dim = last
        self.out = nn.Linear(last, num_classes).to(device)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

    def forward(self, x):
        h = self.init_hidden(self.batch_size)
        h = tuple([e.data for e in h])
        out, hidden = self.lstm_layer(x, h)
        x = out[:,-1,:]
        x = self.flatten(x)
        x = self.dropout(x)
        for layer in self.fc:
            x.to(device)
            x = layer(x)
            x = F.relu(x)
        y = self.out(x)
        # tag_scores = F.softmax(y)
        return y
        # print(x.shape)
        # embedding = self.encoder(x,h)
        # return embedding

    def configure_optimizers(self):
        optimizer = AdaBelief(self.parameters(), lr=1e-3, eps=1e-7, betas=(0.9,0.999))
        return optimizer

    def training_step(self, train_batch, batch_idx):
        h = self.init_hidden(self.batch_size)
        h = tuple([e.data for e in h])
        x, y = train_batch
        x = x.view(x.size(0),x.size(1), -1)
        # y_hat = self.encoder(x)
        out, hidden = self.lstm_layer(x, h)
        x = out[:,-1,:]
        x = self.flatten(x)
        x = self.dropout(x)
        for layer in self.fc:
            x = x.to(device)
            x = layer(x)
            x = F.relu(x)
        y_hat = self.out(x)
        y = y.type(torch.LongTensor).to(device)
        loss = F.cross_entropy(y_hat, y)
        return {'loss':loss }
    def training_epoch_end(self,outputs):
        if(len(outputs) > 0):
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            if(avg_loss < self.loss):
                self.loss = avg_loss
            self.log('loss',avg_loss)
            # print('loss : ')
            # print(avg_loss)
    def validation_epoch_end(self,outputs):
        if(len(outputs) > 0):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            if(avg_loss < self.val_loss):
                self.val_loss = avg_loss
            # exp.metric("val_loss", avg_loss)
            self.log('val_loss',avg_loss)
            # print('vloss : ')
            # print(avg_loss)

    def validation_step(self, val_batch, batch_idx):
        h = self.init_hidden(self.batch_size)
        h = tuple([e.data for e in h])
        x, y = val_batch
        x = x.view(x.size(0),x.size(1), -1)
        out, hidden = self.lstm_layer(x, h)
        x = out[:,-1,:]
        x = self.flatten(x)
        x = self.dropout(x)
        for layer in self.fc:
            x = x.to(device)
            x = layer(x)
            x = F.relu(x)
        y_hat = self.out(x)
        y = y.type(torch.LongTensor).to(device)
        # y_hat = self.encoder(x)    
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss':loss}
    def get_loss(self):
      return (self.loss,self.val_loss)

model = LitAutoEncoder()

x_train,x_val,x_test,y_train,y_val,y_test = get_data(24)
r1 = len(x_train) % batch_size
r2 = len(x_test) % batch_size
x_val = np.concatenate((x_val,x_train[-r1:],x_test[-r2:]))
y_val = np.concatenate((y_val,y_train[-r1:],y_test[-r2:]))

r3 = len(x_val) % batch_size

x_val = np.array(x_val[0:-r3],dtype=np.float32)
y_val = np.array(y_val[0:-r3],dtype=np.float32)

x_train = np.array(x_train[0:-r1],dtype=np.float32)
y_train = np.array(y_train[0:-r1],dtype=np.float32)
x_test = np.array(x_test[0:-r2],dtype=np.float32)
y_test = np.array(y_test[0:-r2],dtype=np.float32)

train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
val_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))


valid_loss_min = float('inf')
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)


from pytorch_lightning.callbacks import ModelCheckpoint
#tpu_cores=8, gpus=1
checkpoint_callback = ModelCheckpoint(monitor='val_loss')
exp = Experiment("IoT rnn model 1.0")
trainer = pl.Trainer(max_epochs=85,default_root_dir='C:/Users/aly17/Desktop/iot/',callbacks=[checkpoint_callback]) #gpus=1,
trainer.fit(model, train_loader, val_loader)
test_res = trainer.test(model,test_loader)
print(test_res)
(loss,val_loss) = model.get_loss()
try:
    exp.param("loss", loss.detach().cpu().numpy())
    exp.param("val loss", val_loss.detach().cpu().numpy())
except:
    exp.param("loss", loss.detach().numpy())
    exp.param("val loss", val_loss.detach().numpy())
exp.end()
# print(loss.detach().cpu().numpy())
# print(val_loss.detach().cpu().numpy())