import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import *

def train_models(models, optimizers, device, epochs, priors, trainloader, scheduler = None, beta_scheduler = None):
    ## training
    loss_collect = [torch.zeros(epochs) for i in len(models)]
    recon_loss = [torch.zeros(epochs) for i in len(models)]
    kl_loss = [torch.zeros(epochs) for i in len(models)]

    for epoch in range(epochs):
        running_loss = [0 for j in len(models)]
        running_recon = [0 for j in len(models)]
        running_kl = [0 for j in len(models)]

        if beta_scheduler:
            beta = beta_scheduler(epoch)
        else:
            beta = 1

        for j,x in enumerate(train_loader):
            k=0
            for model in models:
                optimizers[k].zero_grad()
                training = True
                loss, recon, kl = model.elbo_standard(x[0].float().to(device), beta = beta, training = training, prior = prior[k])
                loss.backward()

                optimizers[k].step()

                running_loss[k] += loss.detach().cpu()
                running_recon[k] += recon.detach().cpu()
                running_kl[k] += kl.detach().cpu()

        for k in range(len(models)):
            loss_collect[k][epoch] = running_loss[k]/(j+1)
            recon_loss[k][epoch] = running_recon[k]/(j+1)
            kl_loss[k][epoch] = running_kl[k]/(j+1)

    return models, loss_collect, recon_loss, kl_loss
