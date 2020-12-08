import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import *

class print_shape(nn.Module):
  def forward(self, x):
    print(x.shape)
    return x


class CasConv3d(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
    super(CasConv3d, self).__init__()
    self.conv_1d = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                             kernel_size = (kernel_size, kernel_size, 1), stride = (stride, stride, 1),
                             padding = (padding, padding, 0))
    self.conv_2d = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                             kernel_size = (kernel_size, 1, kernel_size), stride = (stride, 1, stride),
                             padding = (padding, 0, padding))
    self.conv_3d = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                             kernel_size = (1, kernel_size, kernel_size), stride = (1, stride, stride),
                             padding = (0, padding, padding))    
  def forward(self, x):
    xd1 = self.conv_1d(x)
    xd2 = self.conv_2d(x)
    xd3 = self.conv_3d(x)

    x = torch.cat((xd1, xd2, xd3), dim = 1)

    return x

class Encoder(nn.Module):
  def __init__(self, latent_dim, input_shape, hidden_dim):
    super().__init__()
    self.latent_dim = latent_dim
    self.input_shape = input_shape

    self.layer1 = nn.Sequential(
        CasConv3d(in_channels = 1, out_channels = hidden_dim[0], 
                           kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        CasConv3d(in_channels = hidden_dim[0] * 3, out_channels = hidden_dim[1], 
                           kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))         
    )
    t, h, w = conv3d_output_shape(input_shape, kernel_size = 2, stride = 2, padding = 0)

    self.layer2 = nn.Sequential(
          CasConv3d(in_channels = hidden_dim[1] * 3, out_channels = hidden_dim[2],
                            kernel_size = 3, stride = 1, padding = 1),
          nn.ReLU(),
          CasConv3d(in_channels = hidden_dim[2] * 3, out_channels = hidden_dim[3],
                            kernel_size = 3, stride = 1, padding = 1),
          nn.ReLU(),
          nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))
    )

    #t, h, w = conv3d_output_shape((t,h,w), kernel_size = 2, stride = 2, padding = 0)

    self.flatten = nn.Flatten(start_dim = 1)
    self.fc1 = nn.Linear(in_features = hidden_dim[1] * 3 * t  * h * w, out_features = self.latent_dim * 2)

  def forward(self, x):
    batchsize = x.shape[0]
    x = x.view(batchsize, *self.input_shape)
    x = x.unsqueeze(1)
    x = self.layer1(x)
    #x = self.layer2(x)

    x = self.flatten(x)
    mu, log_std = self.fc1(x).chunk(2, dim = 1)

    return mu, log_std

class Decoder(nn.Module):
  def __init__(self, latent_dim, output_shape, hidden_dim):
    super().__init__()
    self.latent_dim = latent_dim
    self.output_shape = output_shape
    self.hidden_dim = hidden_dim

    t, h, w = conv3d_output_shape(output_shape, kernel_size = 2, stride = 2, padding = 0)
    #t, h, w = conv3d_output_shape((t,h,w), kernel_size = 2, stride = 2, padding = 0)

    self.base_size = (hidden_dim[1] * 3, t, h, w)

    self.fc1 = nn.Linear(in_features = self.latent_dim, out_features = np.prod(self.base_size))
    
    self.layer1 = nn.Sequential(
        nn.ConvTranspose3d(in_channels = hidden_dim[3] * 3, out_channels = hidden_dim[3] * 3,
                                         kernel_size = 2, stride = 2, output_padding = (0,0,1)),
        nn.ReLU(),
        CasConv3d(in_channels = hidden_dim[3] * 3, out_channels = hidden_dim[2],
                           kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        CasConv3d(in_channels = hidden_dim[2] * 3, out_channels = hidden_dim[1],
                           kernel_size = 3, stride = 1, padding = 1)           
    )

    self.layer2 = nn.Sequential(
        nn.ConvTranspose3d(in_channels = hidden_dim[1] * 3, out_channels = hidden_dim[1] * 3,
                                         kernel_size = 2, stride = 2, output_padding = (1,1,0)),
        nn.ReLU(),
        CasConv3d(in_channels = hidden_dim[1] * 3, out_channels=hidden_dim[0],
                           kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        CasConv3d(in_channels = hidden_dim[0] * 3, out_channels=1,
                           kernel_size = 3, stride = 1, padding = 1)
    )
    
    self.final_conv_mean = nn.Conv3d(in_channels = 3, out_channels = 1,
                                kernel_size = 3, stride = 1 , padding = 1)
    #self.final_conv_logst = nn.Conv3d(in_channels = 3, out_channels = 1,
    #                            kernel_size = 3, stride = 1 , padding = 1)

  def forward(self, z):
    batchsize = z.shape[0]
    z = self.fc1(z)
    out = z.view(batchsize, *self.base_size)

    #out = self.layer1(out)
    out = self.layer2(out)

    out_mean = self.final_conv_mean(out)
    #out_logstd = self.final_conv_logst(out)

    return out_mean#, out_logstd

class VAE(nn.Module):
  def __init__(self, latent_dim, input_shape, hidden_dim, device, data_sample, data_init, cuda = True, n_pseudo = 5):
    super().__init__()
    self.latent_dim = latent_dim
    self.input_shape = input_shape
    self.n_pseudo = n_pseudo
    self.cuda = cuda
    self.device = device
    self.hidden_dim = hidden_dim
    self.data_init = data_init
    self.data_sample = torch.Tensor(data_sample)

    self.pseudo_linear = nn.Linear(self.n_pseudo, np.prod(self.input_shape), bias = False)

    if self.data_init:
      means = torch.transpose(self.data_sample.reshape(self.n_pseudo, -1), 0, 1)
      self.pseudo_linear.weight.data = means

    self.encoder = Encoder(self.latent_dim, self.input_shape, self.hidden_dim)
    self.decoder = Decoder(self.latent_dim, self.input_shape, self.hidden_dim)


  def forward(self, x, training = True):
    mu, log_std = self.encoder(x)

    #reparametrization
    if training:
      with torch.no_grad():
        eps = torch.randn_like(mu)
      z = eps * log_std.exp() + mu
    else:
      z = mu

    out = self.decoder(z)
    return out.squeeze(), mu, log_std, z

  def pseudo_inputs(self):
    u = torch.autograd.Variable(torch.eye(self.n_pseudo, self.n_pseudo), requires_grad = False)
    if self.cuda:
      u = u.to(self.device)

    u = self.pseudo_linear(u)

    return u
  
  def log_Normal_diag(self, x, mean, log_std, average = False, dim = None):
    log_normal = -log_std - 0.5 * ( torch.pow( x - mean, 2 ) / torch.exp( 2 * log_std ) )

    if average:
        return torch.mean( log_normal, dim = dim)
    else:
        return torch.sum( log_normal, dim = dim)
  
  def log_Normal_standard(self, x, average = False, dim = None):
    log_normal = - 0.5 * torch.pow( x , 2 ) 

    if average:
        return torch.mean(log_normal, dim = dim)
    else:
        return torch.sum(log_normal, dim = dim)

  def prior(self, z2, prior):

    if prior == 'vampprior':
      u = self.pseudo_inputs()
      #u = u.view(self.n_pseudo, *self.input_shape)
      z2_mean, z2_logstd = self.encoder(u)

      z_expand = z2.unsqueeze(1)
      means = z2_mean.unsqueeze(0)
      logstd = z2_logstd.unsqueeze(0)

      a = self.log_Normal_diag(z_expand, means, logstd, dim = 2) - np.log(self.n_pseudo)  # MB x C
      a_max, _ = torch.max(a, 1)  # MB
      # calculte log-sum-exp
      log_prior = (a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))) 

    elif prior == 'standard':
      log_prior = self.log_Normal_standard(z2, dim=1)

    return log_prior

  def elbo_standard(self, x, beta = 1, training = True, prior = 'vampprior'):
    mu_out, mu, log_std, z = self.forward(x, training = training)
    
    log_p_z = self.prior(z, prior)
    log_q_z = self.log_Normal_diag(z, mu, log_std, dim = 1)

    x = x.reshape(x.shape[0], *self.input_shape)
    recon_loss = ((x - mu_out) ** 2).sum(axis=(1,2,3)).mean(axis=0)
    #recon_loss = -self.log_Normal_diag(x, mu_out, sigma_out, dim = 1).mean()
    if beta > 0:
      kl_loss = -(log_p_z - log_q_z)
      kl_loss = kl_loss.mean()
    else:
      kl_loss = 0


    return recon_loss + beta * kl_loss, recon_loss, kl_loss

  def get_log_prob(self, x, x_hat, x_sigma, mu, log_std, z, prior):    
    log_p_z = self.prior(z, prior)
    log_q_z = self.log_Normal_diag(z, mu, log_std, dim = 1)

    recon_loss = self.log_Normal_diag(x, x_hat, x_sigma, dim = 1)
    kl_loss = log_p_z - log_q_z

    return recon_loss + kl_loss


  def reconstruction(self, x, L = 10, prior = 'vampprior'):
    mu, log_std = self.encoder(x)
    x_recons = torch.zeros((x.shape[0], x.shape[1], L))
    x_sigma = torch.zeros((x.shape[0], x.shape[1], L))
    log_prob = torch.zeros((x.shape[0], L))

    for i in range(L):
       z = torch.randn_like(mu) * log_std.exp() + mu
       x_recons[:, :, i], x_sigma[:, :, i] = self.decoder(z)
       log_prob[:, i] = self.get_log_prob(x, x_recons[:, :, i], x_sigma[:, :, i],
                                          mu, log_std, z, prior)

    
    return x_recons, x_sigma, mu, log_std, log_prob.mean(dim = 1)    


