import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import *

def train_models(models, optimizers, epochs, priors, trainloader, valloader, scheduler = None, beta_scheduler = None):
    ## training
    loss_collect = [torch.zeros(epochs) for i in len(models)]
    recon_loss = [torch.zeros(epochs) for i in len(models)]
    kl_loss = [torch.zeros(epochs) for i in len(models)]
    val_loss = [torch.zeros(epochs) for i in len(models)]

    for epoch in range(epochs):
        running_loss = [0 for j in len(models)]
        running_recon = [0 for j in len(models)]
        running_kl = [0 for j in len(models)]
        running_val_loss = [0 for j in len(models)]

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

        #scheduler.step()
        #validation

        for i,x in enumerate(val_loader):
            k=0
            for model in models:
                training = True
                loss, recon, kl = model.elbo_standard(x[0].float().to(device), beta = beta, training = training, prior = priors[k])
                running_val_loss[k] += loss.detach().cpu()

        for k in range(len(models)):
            loss_collect[k][epoch] = running_loss[k]/(j+1)
            recon_loss[k][epoch] = running_recon[k]/(j+1)
            kl_loss[k][epoch] = running_kl[k]/(j+1)
            val_loss[epoch] = running_val_loss/(i+1)

        if (epoch + 1) % 10 == 0:
            print(str(epoch + 1) + ' out of ' + str(epochs))
            print('Training loss: ' + str(loss_collect[epoch]))
            print('Reconstruction loss:' + str(recon_loss[epoch]))
            print('KL loss:' + str(kl_loss[epoch]))
            print('Validation loss:' + str(val_loss[epoch]))

