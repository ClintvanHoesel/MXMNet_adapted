# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:39:49 2021

@author: s164097
"""
import torch
import torch.nn as nn
import numpy as np

MIN_VAL = 1e-6

class Config(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class Envelope(nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -8./3
        self.b = 2.
        self.c = -1./3

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_2p0 = x_pow_p0 * x_pow_p0
        env_val = 1. + a * x_pow_p0 + b * x_pow_2p0 + c * (x_pow_2p0 * x_pow_2p0)

        zero = torch.zeros_like(x)
        return torch.where(x < 1., env_val, zero)
        
class RadialBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 exponent = 8):
        super().__init__()

        self.n = torch.arange(1, n_rbf + 1).float()
        if learnable_k:
            self.n = nn.Parameter(self.n)

        self.cutoff = cutoff
        self.envelope = Envelope(exponent = exponent)
        
        self.prefactor = 1.#np.sqrt(2./(self.cutoff**3))

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist / self.cutoff#.unsqueeze(-1)
        device = shape_d.device
        n = self.n.to(device)
        coef = n * np.pi
        
        output = torch.sinc(coef * shape_d)
        
        output = self.prefactor * output * self.envelope(shape_d)

        return output