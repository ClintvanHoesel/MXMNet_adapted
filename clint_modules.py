import numpy as np
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, constant_, zeros_, uniform_
from torch.nn import Parameter, Sequential, ModuleList, Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, sort_edge_index
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_sparse import coalesce
from torch_scatter import scatter
from torch_geometric.io import read_txt_array
import pytorch_lightning as pl

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from operator import itemgetter
from collections import OrderedDict

import os
import os.path as osp
import shutil
import glob

EPS = 1e-8
STANDARD_GAIN = 0.01
RES_GAIN = 2.
BIAS_GAIN = 2.

class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_init=kaiming_uniform_,
        bias_init=uniform_,
        gain = STANDARD_GAIN,
        bias_gain = BIAS_GAIN
    ):

        self.weight_init = weight_init
        self.gain = gain
        self.bias_init = bias_init
        self.bias_gain = bias_gain

        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight, a = self.gain)
        if self.bias is not None:
            bound = np.sqrt(2.0/(self.out_features*(1+self.bias_gain**2)))
            self.bias_init(self.bias, a =-bound, b=bound)
            # self.bias_init(self.bias)
    # def forward(self, inputs):
    #     # self.to(inputs.device)
    #     y = super().forward(inputs)
    #     return y

class SiLU(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, input):
        return silu(input)


def silu(input):
    return input * torch.sigmoid(input)

class HardSiLU(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, input):
        return hardsilu(input)

def hardsilu(x):
    out = torch.where(x<=-2., 2./(3.*x), x)
    out = torch.where(x>-2.&x<3., (1./6)*x*(x+3), out)
    # if x>3:
    #     return x
    # elif x<-2:
    #     return 2./(3.*x)
    # else:
    #     return (1./6)*x*(x+3)
    return out
    
def MLP(channels, bias = True, act_out = True, gain = STANDARD_GAIN):
    if not isinstance(gain, list):
        gain = [gain]*(len(channels)-1)
    if not act_out:
        gain[-1] = 1.
    
    list_ops = []
    for i in range(1,len(channels)):
        list_ops.append(Dense(channels[i - 1], channels[i], bias, gain = gain[i-1]))
        list_ops.append(nn.LeakyReLU())
    if not act_out:
        list_ops = list_ops[:-1]
    return Sequential(*list_ops)

class Res(nn.Module):
    def __init__(self, dim, bias = True, act_out = True):
        super(Res, self).__init__()

        self.mlp = MLP([dim, dim, dim], bias, act_out, gain = [STANDARD_GAIN, STANDARD_GAIN])
        self.corr = 1./(2. - 0.5*(1. - STANDARD_GAIN)*act_out)

    def forward(self, m):
        m1 = self.mlp(m)
        m = m1 + m
        return m*self.corr

class NaNLoss(nn.Module):
    def __init__(self, eps = 1e-5):
        self.eps = eps
        self.epssq = eps*eps
        super(NaNLoss,self).__init__()
        
    def forward(self, output, target, weights = None, power = 2, mean = True):
        x = output
        y = target
        is_nan = torch.logical_or(torch.isnan(y).detach(),torch.isnan(x).detach())
        x[is_nan] = 0
        y[is_nan] = 0
        if power==1:
            out = torch.abs(x - y)
        else:
            out = torch.pow(torch.abs(x - y), power)
        if weights is not None:
            out = out * weights
        out = torch.sum(out) + self.eps

        if mean:
            if weights is not None:
                denom = ((torch.sum((~is_nan).float() * weights) + self.epssq).detach())
            else:
                denom = ((torch.sum((~is_nan).float()) + self.epssq).detach())
            out_mean = out / denom

            return out, out_mean
        else:
            return out