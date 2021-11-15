import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, zeros_initializer
EPS = 1e-15

def norm(vec):
    result = ((vec ** 2 + EPS).sum(-1)) ** 0.5
    return result


def preprocess_r(r_ij):
    """
    r_ij (n_nbrs x 3): tensor of interatomic vectors (r_j - r_i)
    """

    dist = norm(r_ij)
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit

class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`
    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=F.leaky_relu,
        dropout_rate=0.,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        gain = 1.
    ):

        self.weight_init = weight_init
        self.gain = gain
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight, gain = self.gain)
        if self.bias is not None:
            self.bias_init(self.bias)
    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.
        Returns:
            torch.Tensor: Output of the dense layer.
        """
        self.to(inputs.device)
        y = super().forward(inputs)
        if hasattr(self, "dropout"):
            y = self.dropout(y)
        if self.activation:
            y = self.activation(y)
        return y

class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):
        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output

    
class PainnRadialBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 learnable_k):
        super().__init__()

        self.n = torch.arange(1, n_rbf + 1).float()
        if learnable_k:
            self.n = nn.Parameter(self.n)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function

        denom = torch.where(shape_d == 0,
                            torch.tensor(1.0, device=device),
                            shape_d)
        num = torch.where(shape_d == 0,
                          coef,
                          torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff,
                             torch.tensor(0.0, device=device),
                             num / denom)

        return output
    
class ExpNormalBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 learnable_mu,
                 learnable_beta):
        super().__init__()

        self.mu = torch.linspace(np.exp(-cutoff), 1, n_rbf)

        init_beta = (2 / n_rbf * (1 - np.exp(-cutoff))) ** (-2)
        self.beta = (torch.ones_like(self.mu) * init_beta)

        if learnable_mu:
            self.mu = nn.Parameter(self.mu)
        if learnable_beta:
            self.beta = nn.Parameter(self.beta)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        mu = self.mu.to(dist.device)
        beta = self.beta.to(dist.device)

        arg = beta * (torch.exp(-shape_d) - mu) ** 2
        output = torch.exp(-arg)

        return output
    

class InvariantDense(nn.Module):
    def __init__(self,
                 dim,
                 dropout,
                 activation='swish'):
        super().__init__()
        self.layers = nn.Sequential(Dense(in_features=dim,
                                          out_features=dim,
                                          bias=True,
                                          dropout_rate=dropout,
                                          activation=to_module(activation)),
                                    Dense(in_features=dim,
                                          out_features=3 * dim,
                                          bias=True,
                                          dropout_rate=dropout))

    def forward(self, s_j):
        output = self.layers(s_j)
        return output
    
class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 feat_dim,
                 learnable_k,
                 dropout):

        super().__init__()
        rbf = PainnRadialBasis(n_rbf=n_rbf,
                               cutoff=cutoff,
                               learnable_k=learnable_k)

        dense = Dense(in_features=n_rbf,
                      out_features=3 * feat_dim,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        self.to(dist.device)
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope #Does this make sense?

        return output
    
class InvariantMessage(nn.Module):
    def __init__(self,
                 feat_dim,
                 activation,
                 n_rbf,
                 cutoff,
                 learnable_k,
                 dropout):
        super().__init__()

        self.inv_dense = InvariantDense(dim=feat_dim,
                                        activation=activation,
                                        dropout=dropout)
        self.dist_embed = DistanceEmbed(n_rbf=n_rbf,
                                        cutoff=cutoff,
                                        feat_dim=feat_dim,
                                        learnable_k=learnable_k,
                                        dropout=dropout)

    def forward(self,
                s_j,
                dist,
                nbrs):

        phi = self.inv_dense(s_j)[nbrs[:, 1]]
        w_s = self.dist_embed(dist)
        output = phi * w_s

        # split into three components, so the tensor now has
        # shape n_atoms x 3 x feat_dim

        feat_dim = s_j.shape[-1]
        out_reshape = output.reshape(output.shape[0], 3, feat_dim)

        return out_reshape