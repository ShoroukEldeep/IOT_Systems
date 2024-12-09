
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

input_dim = 2
# hidden_dim = 1
# n_layers = 1
batch_size = 16
# seq_len = 100
num_classes = 7

class LitAutoEncoder(pl.LightningModule):
	def __init__(self,hidden_dim,n_layers,seq_len,dense,dim1,dim2,dim3,drop1,drop2):
		super().__init__()
        self.input_dim = 2
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.batch_size = 512
        self.seq_len = int(seq_len)
        self.dims = [int(dim1),int(dim2),int(dim3)]
        self.drop1 = float(drop1/100)
        self.drop2 = float(drop2/100)
        layers = []
        layers.append(nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True,dropout=self.drop1))
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(self.drop2))
        fc_input_dim = self.hidden_dim
        last = fc_input_dim
        for i in range(dense):
            layers.append(nn.Linear(fc_input_dim, self.dims[i]))
            last = self.dims[i]
            fc_input_dim = last
        layers.append(nn.Linear(last, num_classes))
		self.encoder = nn.Sequential(*layers)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

	def forward(self, x):
        h = self.init_hidden(self.batch_size)
        h = tuple([e.data.to(dev) for e in h])
		embedding = self.encoder(x,h)
		return embedding

	def configure_optimizers(self):
		optimizer = AdaBelief(self.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999))
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		x = x.view(x.size(0), -1)
		y_hat = self.encoder(x)    
		loss = F.cross_entropy(y_hat, y)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		x = x.view(x.size(0), -1)
		y_hat = self.encoder(x)    
		loss = F.cross_entropy(y_hat, y)
		self.log('val_loss', loss)




# ======================


def train(para):
    model = LitAutoEncoder(para[0],para[1],para[2],para[3],para[4],para[5],para[6],para[7],para[8])

    batch_size = 512
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
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    exp = Experiment("IoT rnn model 0.2")
    exp.param("hidden_dim", para[0])
    exp.param("n_layers", para[1])
    exp.param("seq_len", para[2])
    exp.param("dense", para[3])
    exp.param("dim1", para[4])
    exp.param("dim2", para[5])
    exp.param("dim3", para[6])
    exp.param("drop1", para[7])
    exp.param("drop2", para[8])
    





    trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)


    exp.end()
    except Exception as e:
        print(e)
    if(valid_loss_min):
        return valid_loss_min
    else:
        return 1000
  
    
