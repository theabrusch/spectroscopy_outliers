import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def conv3d_output_shape(t_h_w, kernel_size = 1, stride = 1, padding = 1, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if type(stride) is not tuple:
        stride = (stride, stride, stride)
    if type(padding) is not tuple:
        padding = (padding, padding, padding)
    t = floor( ((t_h_w[0] + (2 * padding[0]) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride[0]) + 1)
    h = floor( ((t_h_w[1] + (2 * padding[1]) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride[1]) + 1)
    w = floor( ((t_h_w[2] + (2 * padding[2]) - ( dilation * (kernel_size[2] - 1) ) - 1 )/ stride[2]) + 1)
    return t, h, w


def get_grad(model, x, prior = 'vampprior'):
    #get elbo grad
    model.zero_grad()
    x = x.detach()
    input = torch.autograd.Variable(x.data, requires_grad=True)
    loss, recon, kl, full_recon = model.elbo_standard(input, prior = prior, return_full_recon = True)
    loss.backward()

    elbo_grad = input.grad.detach()
    model.zero_grad()

    #get recon grad
    model.zero_grad()
    x = x.detach()
    input = torch.autograd.Variable(x.data, requires_grad=True)
    loss, recon, kl = model.elbo_standard(input)
    recon.backward()

    recon_grad = input.grad.detach()
    model.zero_grad()

    #get kl grad
    model.zero_grad()
    input = torch.autograd.Variable(x.data, requires_grad=True)
    loss, recon, kl = model.elbo_standard(input)
    kl.backward()

    kl_grad = input.grad.detach()
    model.zero_grad()

    return torch.abs(elbo_grad.detach()).cpu(), torch.abs(recon_grad.detach()).cpu(), torch.abs(kl_grad.detach()).cpu(), full_recon.detach().cpu()

def get_outliers(model, x, prior, outliers):
    elbo_grad, recon_grad, kl_grad, recon = get_grad(model, x, prior = prior)
    outliers = ((outliers>0).sum(2)>0).long().reshape(-1,1)
    auc_elbo_grad = roc_auc_score(outliers, elbo_grad.mean(2).view(-1,1))
    auc_recon_grad = roc_auc_score(outliers, recon_grad.mean(2).view(-1,1))
    auc_kl_grad = roc_auc_score(outliers, kl_grad.mean(2).view(-1,1))
    auc_recon = roc_auc_score(outliers, recon.mean(3).view(-1,1))

    return auc_elbo_grad, auc_recon_grad, auc_kl_grad, auc_recon
