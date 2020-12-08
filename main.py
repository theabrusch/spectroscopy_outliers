import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import *
from src.models import *
from src.trainfunctions import *
import pickle

#load data
with open('/data/smalldata.pickle', 'rb') as f:
    dataset = pickle.load(f)

with open('/data/smalldata_validation.pickle', 'rb') as f:
    val_dataset = pickle.load(f)

with open('/data/smalldata_outlier.pickle', 'rb') as f:
    out_dataset = pickle.load(f)

x_train = dataset['data']
x_val = val_dataset['data']
x_out = out_dataset['data']
outlier = out_dataset['outlier']

dataset_torch = torch.utils.data.TensorDataset(x_train)
dset_torch_val = torch.utils.data.TensorDataset(x_val)
dset_torch_out = torch.utils.data.TensorDataset(x_out, outlier) 
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset_torch, batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(dset_torch_val, batch_size, shuffle=True)
out_loader =  torch.utils.data.DataLoader(dset_torch_out, batch_size, shuffle=True)

#initialize pseudo inputs
n_pseudo = 5
random_idx = np.random.choice(range(len(x_train)), size = n_pseudo, replace = False)
random_sample = x_train[random_idx,:,:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dims = [1, 2, 4, 8, 12, 25, 50, 100]
for latent_dim in latent_dims:
    epochs = 500
    model_vamp = VAE(latent_dim, input_shape = (5, 5, 10), hidden_dim = [32, 64, 64, 128], data_sample = random_sample, data_init = True, device = device, n_pseudo = n_pseudo).to(device)
    optimizer_vamp = optim.Adam(model_vamp.parameters(), lr = 1e-4)

    model_stand = VAE(latent_dim, input_shape = (5, 5, 10), hidden_dim = [32, 64, 64, 128], data_sample = random_sample, data_init = False, device = device, n_pseudo = n_pseudo).to(device)
    optimizer_stand = optim.Adam(model_stand.parameters(), lr = 1e-4)

    out = train_models([model_vamp, model_stand], [optimizer_vamp, optimizer_stand], device, epochs, ['vampprior', 'standard'], train_loader)
