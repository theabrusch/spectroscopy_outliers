import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
with open('data/smalldata.pickle', 'rb') as f:
    dataset = pickle.load(f)

with open('data/smalldata_validation.pickle', 'rb') as f:
    val_dataset = pickle.load(f)

with open('data/smalldata_outlier.pickle', 'rb') as f:
    out_dataset = pickle.load(f)

x_train = dataset['data']
x_val = val_dataset['data']
x_out = torch.Tensor(out_dataset['data'])
outlier = torch.Tensor(out_dataset['outlier'])

dataset_torch = torch.utils.data.TensorDataset(x_train)
dset_torch_val = torch.utils.data.TensorDataset(x_val)
#dset_torch_out = torch.utils.data.TensorDataset(x_out, outlier) 
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset_torch, batch_size, shuffle=True)
val_loader =  torch.utils.data.DataLoader(dset_torch_val, batch_size, shuffle=True)
#out_loader =  torch.utils.data.DataLoader(dset_torch_out, batch_size, shuffle=True)

#initialize pseudo inputs
n_pseudo = 10
random_idx = np.random.choice(range(len(x_train)), size = n_pseudo, replace = False)
random_sample = x_train[random_idx,:,:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dims = [1, 2, 4, 8, 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]

output = dict()
runs = 1
epochs = 500

latent_dim_stand = 20
latent_dim_vamp = 8
val_stand = np.zeros(runs)
val_vamp = np.zeros(runs)
aucs_stand = np.zeros((runs, 5))
aucs_vamp = np.zeros((runs, 5))
final_stand = np.zeros(runs)
final_vamp = np.zeros(runs)

for run in range(runs):
    model_vamp = VAE(latent_dim_vamp, input_shape = (5, 5, 10), hidden_dim = [32, 64, 64, 128], data_sample = random_sample, data_init = True, device = device, n_pseudo = n_pseudo).to(device)
    optimizer_vamp = optim.Adam(model_vamp.parameters(), lr = 1e-4)

    model_stand = VAE(latent_dim_stand, input_shape = (5, 5, 10), hidden_dim = [32, 64, 64, 128], data_sample = random_sample, data_init = False, device = device, n_pseudo = n_pseudo).to(device)
    optimizer_stand = optim.Adam(model_stand.parameters(), lr = 1e-4)

    models, final_loss = train_models([model_stand, model_vamp], [optimizer_stand, optimizer_vamp], device, epochs, ['standard', 'vampprior'], train_loader)

    #get validation loss
    stand_elbo = 0
    vamp_elbo = 0
    for i, x in enumerate(val_loader):
        loss, recon, kl = models[1].elbo_standard(x[0].float().to(device), beta = 1, training = False, prior = 'vampprior')
        vamp_elbo += loss.detach().cpu()

        loss, recon, kl = models[0].elbo_standard(x[0].float().to(device), beta = 1, training = False, prior = 'standard')
        stand_elbo += loss.detach().cpu()

    val_stand[run] = stand_elbo/(i+1)
    val_vamp[run] = vamp_elbo/(i+1)
    
    final_stand[run] = final_loss[0]
    final_vamp[run] = final_loss[1]

    aucs_vamp[run,:] = get_outliers(models[1], x_out.to(device), 'vampprior', outlier)
    aucs_stand[run,:] = get_outliers(models[0], x_out.to(device), 'standard', outlier)

output = {'validation_loss': [np.mean(val_stand), np.mean(val_vamp)],
                        'aucs': [aucs_stand, aucs_vamp],
                        'final_loss': [np.mean(final_stand), np.mean(final_vamp)]}

torch.save(models[0].state_dict(), 'outputs/model_stand.pt')
torch.save(models[1].state_dict(), 'outputs/model_vamp.pt')

with open('outputs/output_final_run.pickle', 'wb') as f:
    pickle.dump(output, f)
