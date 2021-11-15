import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
import pytorch_lightning as pl

from utils import MLP, Res, MessagePassing
import itertools

class Global_MP(MessagePassing):

    def __init__(self, config):
        super(Global_MP, self).__init__()
        self.dim = config.dim
        self.mos = len(config.mo_indices)
        self.config = config

        self.h_mlp = MLP([self.dim,
                          self.dim])

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)
        self.mlp = MLP([self.dim, self.dim])
        # self.mlp = MLP([self.dim + 4, self.dim])

        self.x_edge_mlp = MLP([self.dim * 3 + (config.orbs>=2)*(self.mos**2 + 2*self.mos)
                               + (config.orbs>=3)*(4*self.mos**2),
                               self.dim])
        self.linear = nn.Linear(self.dim, self.dim, bias=False)
        self.d_indices = torch.Tensor(list(zip(itertools.product(range(self.mos),range(self.mos)))))[:,0,:].long()

    def forward(self, h, edge_attr, edge_index, p_orbs, d_orbs, pos):
        # edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        # if self.config.orbs>=2:
        #     p_abs = torch.sqrt(torch.sum(torch.pow(p_orbs, 2), dim=-1))
        #     if self.config.orbs>=3:
        #         d_had = torch.sum(d_orbs[:,self.d_indices[:,0],:,:]*d_orbs[:,self.d_indices[:,0],:,:], dim=[2,3]).reshape(d_orbs.shape[0],-1)

        res_h = h
        
        # Integrate the Cross Layer Mapping inside the Global Message Passing
        # if self.config.orbs>=2:
        #     h = torch.cat((h, p_abs), -1)
        #     if self.config.orbs>=3:
        #         h = torch.cat((h, d_had), -1)
        h = self.h_mlp(h)
        # print(d_orbs.shape)
        # Message Passing operation
        
        h = h + self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr,
                           p=p_orbs,d=d_orbs, pos=pos)
        
        
        # Update function f_u
        h = self.res1(h)
        # if self.config.orbs>=2:
            # h = torch.cat((h, p_abs), -1)
            # if self.config.orbs>=3:
            #     h = torch.cat((h, d_had), -1)
        h = self.mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)
        
        # Message Passing operation
        h = h + self.propagate(edge_index, x=h, num_nodes=h.size(0), edge_attr=edge_attr,
                           p=p_orbs,d=d_orbs, pos=pos)

        return h

    def message(self, x_i, x_j, p_i, p_j, d_i, d_j, pos_i, pos_j, edge_attr, edge_index, num_nodes):
        # num_edge = edge_attr.size()[0]
        diff_pos = pos_j - pos_i
        diff_pos = diff_pos/(torch.sum(torch.pow(diff_pos,2),dim=-1, keepdim=True).sqrt())
        x_edge = torch.cat((x_i, x_j, edge_attr), -1)
        # print(torch.mean(torch.abs(x_edge)))
        if self.config.orbs>=2:
            # print(p_i.shape)
            # print(p_j.shape)
            # p_scalars = torch.matmul(p_i[:,self.d_indices[:,0],:], torch.swapaxes(p_j[:,self.d_indices[:,1],:],1,2)).reshape((p_i.shape[0], -1))
            p_scalars = torch.sum(p_i[:,self.d_indices[:,0],:]*p_j[:,self.d_indices[:,1],:], dim=-1)
            # print(p_scalars.shape)
            # print(p_scalars.shape)
            # print(diff_pos.shape)
            # print(p_i.swapaxes(1,2).shape)
            
            p_inner = torch.matmul(diff_pos[:,None,:], p_i.swapaxes(1,2))[:,0,:]
            # print(p_inner.shape)
            p_inner = torch.cat((p_inner, torch.matmul(diff_pos[:,None,:], p_j.swapaxes(1,2))[:,0,:]),-1)
            # print(p_inner.shape)
            p_inner = p_inner.reshape(p_inner.shape[0],-1)
            
            # print(p_inner.shape)
            # print(x_edge.shape)
            x_edge = torch.cat((x_edge, p_scalars, p_inner), -1)
            # print(torch.mean(torch.abs(x_edge)))
            # print(x_edge.shape)
            
            if self.config.orbs>=3:
                
                # d_i_had = d_i.repeat(1,2,1,1)
                # d_i_had = 
                # d_i_had = torch.reshape(d_i_had, (d_i.shape[0],d_i.shape[1],d_i.shape[1],d_i.shape[2],d_i.shape[3]))
                # d_j_had = d_j.repeat(1,2,1,1)[:,list(range()),:,:]
                # d_j_had = 
                # d_j_had = torch.reshape(d_j_had, (d_j.shape[0],d_j.shape[1],d_j.shape[1],d_j.shape[2],d_j.shape[3]))
                d_had = torch.sum(d_i[:,self.d_indices[:,0],:,:]*d_j[:,self.d_indices[:,1],:,:], dim=[2,3]).reshape(d_i.shape[0],-1)
                
                d_ij = torch.matmul(d_i[:,self.d_indices[:,0],:,:],d_j[:,self.d_indices[:,1],:,:])
                # print(d_ij.shape)
                # print(diff_pos[:,None,None,:].shape)
                d_ij = torch.matmul(diff_pos[:,None,None,:],d_ij)[:,:,0,:]
                d_ij = torch.matmul(d_ij,diff_pos[:,:,None])[:,:,0]
                
                pd_ij = torch.matmul(p_j[:,self.d_indices[:,1],None,:],d_i[:,self.d_indices[:,0],:,:])[:,:,0,:]
                pd_ij = torch.matmul(pd_ij,diff_pos[:,:,None])[:,:,0]
                pd_ji = torch.matmul(p_i[:,self.d_indices[:,1],None,:],d_j[:,self.d_indices[:,0],:,:])[:,:,0,:]
                pd_ji = torch.matmul(pd_ji,diff_pos[:,:,None])[:,:,0]
                # print(torch.mean(torch.abs(x_edge)))
                # print(d_had.shape)
                # print(d_ij.shape)
                # print(pd_ij.shape)
                # print(pd_ji.shape)
                
                
                
                x_edge = torch.cat((x_edge, d_had, d_ij, pd_ij, pd_ji), -1)
                
                
        x_edge = self.linear(edge_attr) * self.x_edge_mlp(x_edge)

        # x_j = torch.cat((self.linear(edge_attr) * x_edge, x_j[num_edge:]), dim=0)

        return x_edge

    def update(self, aggr_out):

        return aggr_out


class Local_MP(pl.LightningModule):
    def __init__(self, config):
        super(Local_MP, self).__init__()
        self.dim = config.dim
        if hasattr(config, "outdim"):
            self.outdim = config.outdim
        else:
            self.outdim = self.dim

        self.h_mlp = MLP([self.dim, self.dim])

        self.mlp_kj = MLP([3 * self.dim, self.dim])
        self.mlp_ji_1 = MLP([3 * self.dim, self.dim])
        self.mlp_ji_2 = MLP([self.dim, self.dim])
        self.mlp_jj = MLP([self.dim, self.dim])

        self.mlp_sbf1 = MLP([self.dim, self.dim, self.dim])
        self.mlp_sbf2 = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf1 = nn.Linear(self.dim, self.dim, bias=False)
        self.lin_rbf2 = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)

        self.h_mlp = MLP([self.dim, self.dim])

        self.y_mlp = MLP([self.dim, self.dim, self.dim, self.dim])
        self.y_W = nn.Linear(self.dim, self.outdim)

    def forward(self, h, rbf, sbf1, sbf2, idx_kj, idx_ji_1, idx_jj, idx_ji_2, edge_index, num_nodes=None):
        res_h = h
        
        # Integrate the Cross Layer Mapping inside the Local Message Passing
        h = self.h_mlp(h)
        
        # Message Passing 1
        j, i = edge_index
        m = torch.cat([h[i], h[j], rbf], dim=-1)

        m_kj = self.mlp_kj(m)
        m_kj = m_kj * self.lin_rbf1(rbf)
        m_kj = m_kj[idx_kj] * self.mlp_sbf1(sbf1)
        m_kj = scatter(m_kj, idx_ji_1, dim=0, dim_size=m.size(0), reduce='add')
        
        m_ji_1 = self.mlp_ji_1(m)

        m = m_ji_1 + m_kj

        # Message Passing 2       (index jj denotes j'i in the main paper)
        m_jj = self.mlp_jj(m)
        m_jj = m_jj * self.lin_rbf2(rbf)
        m_jj = m_jj[idx_jj] * self.mlp_sbf2(sbf2)
        m_jj = scatter(m_jj, idx_ji_2, dim=0, dim_size=m.size(0), reduce='add')
        
        m_ji_2 = self.mlp_ji_2(m)

        m = m_ji_2 + m_jj

        # Aggregation
        m = self.lin_rbf_out(rbf) * m
        h = scatter(m, i, dim=0, dim_size=h.size(0), reduce='add')
        
        # Update function f_u
        h = self.res1(h)
        h = self.h_mlp(h) + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Output Module
        y = self.y_mlp(h)
        y = self.y_W(y)
        return h, y
