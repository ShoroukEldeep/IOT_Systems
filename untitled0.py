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

device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu") #"cuda:0" if torch.cuda.is_available() else 

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
# c = 0
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        hidden_dim, n_layers, seq_len, dense, dim1, dim2, dim3, drop1, drop2 = 52,2,24,0,349,158,44,0,16
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
        # layers = []
        # layers.append(nn.LSTM(self.input_dim, self.hidden_dim,
        #             self.n_layers, batch_first=True, dropout=self.drop1))
        # layers.append(nn.Flatten())
        # layers.append(nn.Dropout(self.drop2))
        # fc_input_dim = self.hidden_dim
        # last = fc_input_dim
        # for i in range(dense):
        #     layers.append(nn.Linear(fc_input_dim, self.dims[i]))
        #     last = self.dims[i]
        #     fc_input_dim = last
        # layers.append(nn.Linear(last, num_classes))
        # self.encoder = nn.Sequential(*layers)

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
        optimizer = AdaBelief(self.parameters(), lr=1e-5, eps=1e-7, betas=(0.9,0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=4,mode='min')
        return {
       'optimizer': optimizer,
       'lr_scheduler': scheduler,
       'monitor': 'val_loss'
        }

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
            print('loss : ')
            print(avg_loss)
    def validation_epoch_end(self,outputs):
        if(len(outputs) > 0):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            if(avg_loss < self.val_loss):
                self.val_loss = avg_loss
            # exp.metric("val_loss", avg_loss)
            self.log('val_loss',avg_loss)
            print('vloss : ')
            print(avg_loss)

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


def compare(arr1,arr2):
    an_array = np.array(arr1)
    another_array = np.array(arr2)
    comparison = an_array == another_array
    equal_arrays = comparison.all()
    return equal_arrays
def inc(counter):
    counter = counter + 1
    print(counter)
    return counter
def train(para):
#     c = 0
#     if compare(para,[42,1,10,1,379,196,63,0,9]):
#         c = inc(c)
#         return np.array([1.3949862])
#     if compare(para,[35,1,19,1,461,128,60,0,18]):
#         c = inc(c)
#         return np.array([1.1182994])
#     if compare(para,[60,1,24,0,482,165,49,0,20]):
#         c = inc(c)
#         return np.array([0.3744779])
#     if compare(para,[45,1,10,2,443,239,117,1,14]):
#         c = inc(c)
#         return np.array([1.3958929])
#     if compare(para,[47,1,17,0,438,200,53,0,20]):
#         c = inc(c)
#         return np.array([1.1010963])
#     if compare(para,[58,1,23,1,344,235,73,1,14]):
#         c = inc(c)
#         return np.array([0.5350919])
#     if compare(para,[60,2,11,1,268,129,99,0,17]):
#         c = inc(c)
#         return np.array([1.0264472])
#     if compare(para,[24,1,6,1,372,150,94,1,16]):
#         c = inc(c)
#         return np.array([1.4223425])
#     if compare(para,[37,2,9,2,376,169,123,1,3]):
#         c = inc(c)
#         return np.array([1.3910831])
#     if compare(para,[33,1,19,2,372,160,97,1,10]):
#         c = inc(c)
#         return np.array([1.2042433])
#     if compare(para,[52,1,16,2,511,192,59,1,6]):
#         c = inc(c)
#         return np.array([1.0841968])
#     if compare(para,[34,1,10,2,399,237,100,1,14]):
#         c = inc(c)
#         return np.array([1.3978055])
#     if compare(para,[35,2,22,1,482,212,62,0,11]):
#         c = inc(c)
#         return np.array([0.26460952])
#     if compare(para,[11,2,9,2,399,204,69,1,5]):
#         c = inc(c)
#         return np.array([1.4073899])
#     if compare(para,[2,1,7,1,372,171,69,1,11]):
#         c = inc(c)
#         return np.array([1.4250569])
#     if compare(para,[12,1,10,2,319,250,45,1,17]):
#         c = inc(c)
#         return np.array([1.4170002])
#     if compare(para,[19,1,5,1,442,154,74,0,11]):
#         c = inc(c)
#         return np.array([1.425])
#     if compare(para,[29,2,23,1,483,248,87,0,6]):
#         c = inc(c)
#         return np.array([0.3937])
#     if compare(para,[11,2,9,2,399,204,69,1,5]):
#         c = inc(c)
#         return np.array([1.411])
#     if compare(para,[12,1,10,2,319,250,45,1,17]):
#         c = inc(c)
#         return np.array([1.4170002])
#     if compare(para,[1,2,24,2,306,154,119,0,6]):
#         c = inc(c)
#         return np.array([1.4251999])
#     if compare(para,[19,1,5,1,442,154,74,0,11]):
#         c = inc(c)
#         return np.array([1.329])
#     if compare(para,[1,2,24,2,306,154,119,0,6]):
#         c = inc(c)
#         return np.array([1.4249207])
#     if compare(para,[4,1,15,1,345,186,98,1,10]):
#         c = inc(c)
#         return np.array([1.4175856])
#     if compare(para,[22,1,25,1,512,200,42,0,12]):
#         c = inc(c)
#         return np.array([1.2012181])
#     if compare(para,[60,1,18,0,446,149,57,0,20]):
#         c = inc(c)
#         return np.array([0.7905772])
#     if compare(para,[64,1,24,0,433,256,63,0,16]):
#         c = inc(c)
#         return np.array([0.34150985])
#     if compare(para,[64,1,19,0,438,195,96,0,15]):
#         c = inc(c)
#         return np.array([0.64331144])
#     if compare(para,[38,2,25,0,448,203,32,1,10]):
#         c = inc(c)
#         return np.array([0.1575887])
#     if compare(para,[63,1,16,0,340,128,61,0,15]):
#         c = inc(c)
#         return np.array([0.9652897])
#     if compare(para,[42,1,6,0,418,173,59,0,16]):
#         c = inc(c)
#         return np.array([1.4221038])
#     if compare(para,[41,2,21,2,415,169,81,1,17]):
#         c = inc(c)
#         return np.array([0.60868835])
#     if compare(para,[56,1,10,2,343,185,109,1,4]):
#         c = inc(c)
#         return np.array([1.3775604])
#     if compare(para,[22,1,20,2,512,256,66,0,9]):
#         c = inc(c)
#         return np.array([1.2841259])
#     if compare(para,[33,2,17,1,439,246,62,0,11]):
#         c = inc(c)
#         return np.array([0.6692452])
#     if compare(para,[30,2,18,2,491,238,59,0,0]):
#         c = inc(c)
#         return np.array([0.87270516])
#     if compare(para,[20,1,6,1,372,171,53,1,11]):
#         c = inc(c)
#         return np.array([1.4224573])
#     if compare(para,[29,2,20,1,439,225,90,0,8]):
#         c = inc(c)
#         return np.array([0.5545489])
#     if compare(para,[45,1,17,1,280,234,57,0,16]):
#         c = inc(c)
#         return np.array([1.1525257])
#     if compare(para,[3,2,23,1,306,187,94,0,4]):
#         c = inc(c)
#         return np.array([1.4013252])
#     if compare(para,[20,1,6,1,372,171,53,1,11]):
#         c = inc(c)
#         return np.array([1.4219939])
#     if compare(para,[29,2,20,1,439,225,90,0,8]):
#         c = inc(c)
#         return np.array([0.49320677])
#     if compare(para,[45,1,17,1,280,234,57,0,16]):
#         c = inc(c)
#         return np.array([1.1139671])
#     if compare(para,[50,1,23,0,465,203,32,0,16]):
#         c = inc(c)
#         return np.array([0.7228077])
#     if compare(para,[40,2,22,0,437,212,49,0,14]):
#         c = inc(c)
#         return np.array([0.23088372])
#     if compare(para,[52,1,22,0,433,211,60,0,16]):
#         c = inc(c)
#         return np.array([0.68917525])

#     if compare(para,[50,1,22,0,498,176,44,0,15]):
#         c = inc(c)
#         return np.array([0.6417112])
#     if compare(para,[22,2,22,1,467,245,66,0,4]):
#         c = inc(c)
#         return np.array([0.7080421])
#     if compare(para,[54,1,19,0,432,168,74,0,17]):
#         c = inc(c)
#         return np.array([0.7925005])
#     if compare(para,[55,1,19,0,369,197,63,0,17]):
#         c = inc(c)
#         return np.array([0.84135556])
#     if compare(para,[24,2,22,0,452,236,37,0,0]):
#         c = inc(c)
#         return np.array([0.7241197])
#     if compare(para,[44,1,15,1,341,128,79,0,15]):
#         c = inc(c)
#         return np.array([1.2151865])
#     if compare(para,[47,1,16,1,451,192,53,1,10]):
#         c = inc(c)
#         return np.array([1.1421])
#     if compare(para,[35,1,19,0,461,128,59,0,18]):
#         c = inc(c)
#         return np.array([1.1206125])
#     if compare(para,[38,1,21,0,352,161,32,0,10]):
#         c = inc(c)
#         return np.array([1.0129683]) # 57
#     if compare(para,[43,2,23,0,448,218,32,0,17]):
#         c = inc(c)
#         return np.array([0.1375943]) #59
#     if compare(para,[39,2,20,0,380,173,32,0,14]):
#         c = inc(c)
#         return np.array([0.2922857])
#     if compare(para,[50,1,23,0,433,256,63,0,16]):
#         c = inc(c)
#         return np.array([0.65571177])
#     if compare(para,[53,1,24,0,482,157,41,0,18]):
#         c = inc(c)
#         return np.array([0.50835973])
#     if compare(para,[45,2,22,0,326,173,56,0,18]):
#         c = inc(c)
#         return np.array([0.1939584]) #63
#     if compare(para,[53,1,17,2,415,166,43,0,14]):
#         c = inc(c)
#         return np.array([1.0791688])
#     if compare(para,[24,1,19,0,454,224,32,0,7]):
#         c = inc(c)
#         return np.array([1.2388446])
#     if compare(para,[30,2,25,0,443,218,32,0,6]):
#         c = inc(c)
#         return np.array([0.29459706])
#     if compare(para,[62,1,16,0,394,175,61,0,16]):
#         c = inc(c)
#         return np.array([0.9997949])
#     if compare(para,[22,2,21,0,408,239,63,0,4]):
#         c = inc(c)
#         return np.array([0.81521606])
#     if compare(para,[63,1,16,2,414,143,54,1,16]):
#         c = inc(c)
#         return np.array([1.081231])
#     if compare(para,[43,2,22,0,505,256,36,0,17]):
#         c = inc(c)
#         return np.array([0.14533605])
#     if compare(para,[50,2,24,0,349,158,35,0,19]): #71
#         c = inc(c)
#         return np.array([0.10567277])
#     if compare(para,[31,2,23,1,355,174,45,0,17]):
#         c = inc(c)
#         return np.array([0.35290456])
#     if compare(para,[49,2,23,0,381,173,40,0,19]):
#         c = inc(c)
#         return np.array([0.13474214])
#     if compare(para,[31,2,23,1,355,174,45,0,17]):
#         c = inc(c)
#         return np.array([0.36442107])
#     if compare(para,[49,2,23,0,381,173,40,0,19]): #76
#         c = inc(c)
#         return np.array([0.108108915])
# #72
#     if compare(para,[64,1,24,0,400,200,66,0,21]): #76
#         c = inc(c)
#         return np.array([0.36839247])
#     if compare(para,[60,1,23,0,468,159,49,0,19]): #81
#         c = inc(c)
#         return np.array([0.46246737])
#     if compare(para,[40,2,21,0,428,193,61,0,15]):
#         c = inc(c)
#         return np.array([0.28928405])
#     if compare(para,[38,1,19,0,336,195,86,0,15]):
#         c = inc(c)
#         return np.array([1.080088])
#     if compare(para,[29,2,21,0,441,239,45,0,10]): #84
#         c = inc(c)
#         return np.array([0.5932732])
# #77
#     if compare(para,[52,2,24,0,349,158,44,0,16]):
#         c = inc(c)
#         return np.array([0.07192245])
#     if compare(para,[41,2,23,0,381,173,32,0,17]):
#         c = inc(c)
#         return np.array([0.18209319])
#     if compare(para,[41,2,21,0,452,218,32,0,17]):
#         c = inc(c)
#         return np.array([0.22989325])
#     if compare(para,[52,1,22,0,476,188,50,0,18]):
#         c = inc(c)
#         return np.array([0.6489607])
#     if compare(para,[50,1,22,0,446,176,52,0,15]):
#         c = inc(c)
#         return np.array([0.7559397])
#     if compare(para,[54,1,24,0,433,173 ,38,0 ,18]):
#         c = inc(c)
#         return np.array([0.56561214])
#     if compare(para,[49,1,24,0,446,194 ,50,0 ,17]):
#         c = inc(c)
#         return np.array([0.6511137])
#     if compare(para,[46,2,23,0,349,191 ,44,0 ,16]):
#         c = inc(c)
#         return np.array([0.09137838])
#     if compare(para,[49,2,23,0,381,173 ,47,0 ,19]):
#         c = inc(c)
#         return np.array([0.12467674]) #97
# #86
#     if compare(para,[39,2,20,0,448,204 ,50,0 ,17]):
#         c = inc(c)
#         return np.array([0.34854722])
#     if compare(para,[49,2,23,0,437,170 ,49,0 ,15]):
#         c = inc(c)
#         return np.array([0.11539475])
#     if compare(para,[40,1,22,0,406,199 ,57,0 ,17]):
#         c = inc(c)
#         return np.array([0.8771669])
#     if compare(para,[53,1,23,0,427,229 ,55,0 ,15]):
#         c = inc(c)
#         return np.array([0.61150885])
#     if compare(para,[53,2,23,0,397,140 ,55,0 ,14]):
#         c = inc(c)
#         return np.array([0.08076307])
#     if compare(para,[49,2,25,0,437,180 ,32,0 ,17]):
#         c = inc(c)
#         return np.array([0.08429098])

    # print(c)
    # model = LitAutoEncoder(para[0],para[1],para[2],para[3],para[4],para[5],para[6],para[7],para[8]).to(device)
    # model = LitAutoEncoder.load_from_checkpoint('./lightning_logs/version_94/checkpoints/epoch=81.ckpt').to(device)
    model = LitAutoEncoder.load_from_checkpoint('./lightning_logs/version_118/checkpoints/epoch=16.ckpt').to(device)
    batch_size = 16
    x_train,x_val,x_test,y_train,y_val,y_test = get_data(para[2])
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

    exp = Experiment("IoT rnn model 1.0")
    exp.param("hidden_dim", para[0])
    exp.param("n_layers", para[1])
    exp.param("seq_len", para[2])
    exp.param("dense", para[3])
    exp.param("dim1", para[4])
    exp.param("dim2", para[5])
    exp.param("dim3", para[6])
    exp.param("drop1", para[7])
    exp.param("drop2", para[8])
    



    from pytorch_lightning.callbacks import ModelCheckpoint
    #tpu_cores=8, gpus=1
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer(gpus=1,max_epochs=200,default_root_dir='C:/Users/aly17/Desktop/iot/',callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    test_res = trainer.test(model,test_loader)
    print(test_res)
    (loss,val_loss) = model.get_loss()

    print(loss.detach().cpu().numpy())
    print(val_loss.detach().cpu().numpy())
    exp.param("loss", loss.detach().cpu().numpy())
    exp.param("val loss", val_loss.detach().cpu().numpy())
    exp.end()
    return val_loss.detach().cpu().numpy()
    # if(valid_loss_min):
    #     return valid_loss_min
    # else:
    # return 1000

# from pytorch_lightning import  seed_everything
# seed_everything(0)
# # from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils
# # TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()
# # print(TPU_AVAILABLE)

# !nvidia-smi

# We import the algorithm (You can use from pyade import * to import all of them)
from pyade.newJso import new_jso
import numpy as np
import os
# from rnn import train
import json
from collections import namedtuple

# def run():
#     algorithm = new_jso()
#     params = algorithm.get_default_params(dim=9)
#     params['bounds'] = np.array([[1,64],
#                                 [1,3],
#                                 [5,25],
#                                 [0,3],
#                                 [256,512],
#                                 [128,256],
#                                 [32,128],
#                                 [0,2],
#                                 [0,25]])

#     params['opts'] = (0, 0)
#     params['func'] = lambda x, y: train(x) - y

#     solution, fitness = algorithm.apply(**params)
#     print(solution,fitness)

# run()

# train([52,2,24,0,349,158,44,0,16])



def get_data2(length):
    df = pd.read_csv('Train_Test_IoT_Garage_Door.csv')
    df['h'] = df.time.apply(lambda x : x.split(':')[0])
    df['m'] = df.time.apply(lambda x : x.split(':')[1])
    df['s'] = df.time.apply(lambda x : x.split(':')[2])
    df = df[df.type != 'normal']
    df.set_index(keys=['date','h'], drop=False,inplace=True)
    fdf = df.drop(['ts','date','time','h','m','s'], axis=1)
    
    x = []
    y = []
    # fdf.FC1_Read_Input_Register = df.FC1_Read_Input_Register.apply(lambda x: x/10000)
    # fdf.FC2_Read_Discrete_Value = df.FC2_Read_Discrete_Value.apply(lambda x: x/10000)
    # fdf.FC3_Read_Holding_Register = df.FC3_Read_Holding_Register.apply(lambda x: x/10000)
    # fdf.FC4_Read_Coil = df.FC4_Read_Coil.apply(lambda x: x/10000)
    def temp_change(x):
        if 'open' in x:
            return 1
        elif 'closed' in x:
            return -1
    def temp_change2(x):
        if 'false' in x or not x:
            return -1
        else:
            return 1
    def ons(x):
        if x < 1:
            return -1
        else:
            return 1
    fdf.door_state = fdf.door_state.apply(temp_change)
    fdf.sphone_signal = fdf.sphone_signal.apply(temp_change2)
    fdf.sphone_signal = fdf.sphone_signal.apply(ons)
    # df.light_status = df.light_status.apply(pd.to_numeric)
    for index in tqdm(df.index.unique()):
        arr = [[0,0] for i in range(0,length)]
        sub = fdf.loc[index]
        for i in range(0,sub.shape[0]): #
            arr = arr[1:]
            l1 = sub.drop(['label','type'],axis=1).values[i].tolist()
            # l2 = sub.drop(['label','type'],axis=1).values[i-1].tolist()
            # if('on' in l1[1] or l1[1] == 1):
            #     l1[1] = 1
            # else:
            #     l1[1] = 0
            # l2 = sub.drop(['label','type'],axis=1).values[i-1].tolist()
            # arr.append([l1[0]-l2[0],l1[1]-l2[1]])
            arr.append(l1)
            z = arr
            x.append(z)
            y.append(sub.drop(['door_state','sphone_signal','label'],axis=1).values[i][0])

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(y)).tolist()
    encoded = tf.keras.utils.to_categorical(integer_encoded,8).tolist()

    x_train,x_test,y_train,y_test = train_test_split(x,integer_encoded,shuffle=True,random_state=42,train_size=0.9)
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,shuffle=True,random_state=42,train_size=0.9)
    return x_train,x_val,x_test,y_train,y_val,y_test

input_dim = 2
batch_size = 1024
num_classes = 7





class LitAutoEncoder2(pl.LightningModule):
    def __init__(self):
        hidden_dim, n_layers, seq_len, dense, dim1, dim2, dim3, drop1, drop2 = 52,2,24,0,349,158,44,0,16
        super().__init__()
        self.loss = float('inf')
        self.val_loss = float('inf')
        self.input_dim = 2
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.batch_size = 1024
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
        optimizer = AdaBelief(self.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=10,mode='min',verbose=True,factor=0.01,min_lr=1e-5)
        return {
       'optimizer': optimizer,
       'lr_scheduler': scheduler,
       'monitor': 'val_loss'
        }
        # return optimizer

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
            exp.metric("loss", float(avg_loss.detach().cpu().numpy()))
            # exp.metric("loss", avg_loss.detach().cpu().numpy())
            # print('loss : ')
            # print(avg_loss)
    def validation_epoch_end(self,outputs):
        if(len(outputs) > 0):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            if(avg_loss < self.val_loss):
                self.val_loss = avg_loss
            # exp.metric("val_loss", avg_loss)
            self.log('val_loss',avg_loss)

            
            exp.metric("val_loss", float(avg_loss.detach().cpu().numpy()))
            
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



model = LitAutoEncoder2().to(device)


x_train,x_val,x_test,y_train,y_val,y_test = get_data2(24)

print(len(x_train))
print(len(x_val))
print(len(x_test))

r1 = len(x_train) % batch_size
r2 = len(x_test) % batch_size
if(r1 > 0):
    x_val = np.concatenate((x_val,x_train[-r1:]))
    y_val = np.concatenate((y_val,y_train[-r1:]))
if(r2 > 0):
    x_val = np.concatenate((x_val,x_test[-r2:]))
    y_val = np.concatenate((y_val,y_test[-r2:]))


r3 = len(x_val) % batch_size
if(r3 > 0):
    x_val = np.array(x_val[0:-r3],dtype=np.float32)
    y_val = np.array(y_val[0:-r3],dtype=np.float32)
else:
    x_val = np.array(x_val,dtype=np.float32)
    y_val = np.array(y_val,dtype=np.float32)
if(r1 > 0):
    x_train = np.array(x_train[0:-r1],dtype=np.float32)
    y_train = np.array(y_train[0:-r1],dtype=np.float32)
else:
    x_train = np.array(x_train,dtype=np.float32)
    y_train = np.array(y_train,dtype=np.float32)
if(r2 > 0):
    x_test = np.array(x_test[0:-r2],dtype=np.float32)
    y_test = np.array(y_test[0:-r2],dtype=np.float32)
else:
    x_test = np.array(x_test,dtype=np.float32)
    y_test = np.array(y_test,dtype=np.float32)

train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
val_data = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))


valid_loss_min = float('inf')
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)



# import os 
# os.system("shutdown /s /t 1") 


from pytorch_lightning.callbacks import ModelCheckpoint
# #tpu_cores=8, gpus=1

exp = Experiment("IoT rnn model WEATHER")
checkpoint_callback = ModelCheckpoint(monitor='val_loss')
trainer = pl.Trainer(gpus=1,max_epochs=100,default_root_dir='C:/Users/aly17/Desktop/iot/test',callbacks=[checkpoint_callback]) #gpus=1,
trainer.fit(model, train_loader, val_loader)
test_res = trainer.test(model,test_loader)
print(test_res)
(loss,val_loss) = model.get_loss()

exp.param("loss", loss.detach().cpu().numpy())
exp.param("val loss", val_loss.detach().cpu().numpy())
exp.end()


# import os
# os.system("shutdown /s /t 1")