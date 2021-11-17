# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:36:38 2021

@author: s164097
"""
import os
import os.path as osp
from shutil import copy2
from glob import glob
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, Dataset, download_url
from torch_geometric.nn import radius
from torch_geometric.utils import remove_self_loops

from joblib import Parallel, delayed

from clintutils import load_obj, SaveClass
import itertools

def mo_into_x(data_obj, mo_indices = [3,4], orbs = False):
    d_indices = torch.Tensor(list(zip(itertools.product(range(len(mo_indices)),range(len(mo_indices))))))[:,0,:].long()
    # pd_indices = torch.Tensor(list(zip(itertools.product(range(len(mo_indices)),range(len(mo_indices)),range(len(mo_indices))))))[:,0,:].long()
    
    x = data_obj.x
    mo = data_obj.mo
    
    mo = mo[:,:,mo_indices]
    s_orbs = mo[:,0,:]
    s_orbs = s_orbs[:, None, :]
    
    p_orbs = mo[:,1:4,:]
    # p_orbs_abs = torch.sqrt(torch.sum(torch.pow(p_orbs, 2), dim = 1, keepdim = True))
    p_orbs = p_orbs.swapaxes(1,2)
    p_orbs_all = torch.sum(p_orbs[:,d_indices[:,0],:]*p_orbs[:,d_indices[:,1],:], dim=-1)
    
    
    d_orbs = torch.zeros((mo.shape[0], 9, mo.shape[2]), device = x.device)
    d_orbs[:,0,:] = mo[:,4,:]#xx
    d_orbs[:,4,:] = mo[:,5,:]#yy
    d_orbs[:,8,:] = mo[:,6,:]#zz
    d_orbs[:,1,:] = mo[:,7,:]#xy
    # d_orbs[:,3,:] = mo[:,7,:]#yx
    d_orbs[:,2,:] = mo[:,8,:]#xz
    # d_orbs[:,6,:] = mo[:,8,:]/2#zx
    d_orbs[:,5,:] = mo[:,9,:]#yz
    # d_orbs[:,7,:] = mo[:,9,:]#zy
    d_orbs_abs = torch.sqrt(torch.sum(torch.pow(d_orbs, 2), dim = 1, keepdim = True))
    d_orbs = d_orbs.swapaxes(1,2).reshape((x.shape[0], len(mo_indices), 3, 3))
    d_had = torch.sum(d_orbs[:,d_indices[:,0],:,:]*d_orbs[:,d_indices[:,1],:,:], dim=[2,3]).reshape(d_orbs.shape[0],-1)
    # d_orbs = mo[:,4:,:]
    
    d_orbs = torch.zeros((mo.shape[0], 9, mo.shape[2]), device = x.device)
    d_orbs[:,0,:] = mo[:,4,:]#xx
    d_orbs[:,4,:] = mo[:,5,:]#yy
    d_orbs[:,8,:] = mo[:,6,:]#zz
    d_orbs[:,1,:] = mo[:,7,:]/2#xy
    d_orbs[:,3,:] = mo[:,7,:]/2#yx
    d_orbs[:,2,:] = mo[:,8,:]/2#xz
    d_orbs[:,6,:] = mo[:,8,:]/2#zx
    d_orbs[:,5,:] = mo[:,9,:]/2#yz
    d_orbs[:,7,:] = mo[:,9,:]/2#zy
    d_orbs = d_orbs.swapaxes(1,2).reshape((x.shape[0], len(mo_indices), 3, 3))
    
    
    # mo_abs = torch.cat((s_orbs, p_orbs_abs, d_orbs_abs), dim = 1)
    # mo_abs = mo_abs[:,:orbs,:]
    
    # mo = torch.cat((mo, p_orbs), dim = 1)
    # mo = torch.cat((mo, d_orbs), dim = 1)
    
    # mo = torch.reshape(mo, (mo.shape[0], -1))
    # x = torch.cat((x, mo), dim=1)
    # mo_abs = torch
    if orbs>0:
        x = torch.cat((x, s_orbs.reshape((x.shape[0], -1))), dim=1)
        if orbs>1:
            x = torch.cat((x, p_orbs_all.reshape((x.shape[0], -1))), dim=1)
            if orbs>2:
                x=torch.cat((x, d_had.reshape((x.shape[0], -1))), dim=-1)
    
    # if add_vectors:
    #     print(f"Adding vectors :D")
    data_obj.p_orbs = p_orbs
    data_obj.d_orbs = d_orbs
    
    data_obj.x = x
    return data_obj

def add_edges_based_distance(x, cutoff, k_max = 500):
    row, col = radius(x.pos, x.pos, cutoff, max_num_neighbors=k_max)
    edge_index_g = torch.stack([row, col], dim=0)
    edge_index_g, _ = remove_self_loops(edge_index_g)
    x.edge_index_g = edge_index_g
    return x

def check_if_contains_properties(data_obj, attrs = ["x", "edge_index", "edge_index_g",  "pos", "y", "smile", "experiment", "mo", "en"]):
    for attr in attrs:
        if not hasattr(data_obj, attr):
            print(f"Data object did not have property {attr}")
            return False
    return True

def remove_attributes(x, attrs):
    for attr in attrs:
        delattr(x, attr)
    return x

def remove_filenames(filename, disallowed = ["2_"]):
    out = [False if string in filename else True for string in disallowed]
    return all(out)

def check_experiment(data_obj, experiments = [0,1,3,4,5,6,7,8,9]):
    if int(data_obj.experiment) in experiments:
        return True
    else:
        return False
    
def check_smiles(data_obj, smiles):
    if str(data_obj.smile) in smiles:
        return True
    else:
        return False
    
def saved_into_DataObj(x):
    return Data(x = x.x, edge_index = x.edge_index, edge_index_g = x.edge_index_g, pos = x.pos, y = x.y,
                smile = x.smile, experiment = x.experiment, p_orbs = x.p_orbs,
                d_orbs = x.d_orbs)

def create_vector_feature(x, vecdim, add_p_orbs = False):
    x.v = torch.zeros((x.x.shape[0], vecdim, 3))
    x.t = torch.zeros((x.x.shape[0], vecdim, 3, 3))
    for i in range(vecdim):
        x.v[:,i,i%3] = 1.
    if add_p_orbs:
        for i in range(x.p_orbs.shape[2]):
        # print("ADDING PORBS TO V :D")
            x.v[:,i,:] = x.p_orbs[:,:,i]
    return x
    
class TRANSFORMER():
    def __init__(self, mo_indices, remove, orbs, smiles, cutoff):
        self.mo_indices = mo_indices
        self.remove = remove
        self.orbs = orbs
        self.smiles = smiles
        self.cutoff = cutoff
        
    def pre_filter(self, x):
        bool1 = check_if_contains_properties(x)
        bool2 = check_experiment(x)
        bool3 = True
        if self.smiles:
            bool3 = check_smiles(x, self.smiles)
        return bool1*bool2*bool3
    
    def pre_transform(self, x):
        try:
            x = mo_into_x(x, self.mo_indices, self.orbs)
            if np.abs(self.cutoff-5.0)>0.1:
                x = add_edges_based_distance(x, self.cutoff)
            x = remove_attributes(x, self.remove)
            # x = create_vector_feature(x, self.vecdim, add_p_orbs=bool(self.orbs>=2))
            x = saved_into_DataObj(x)
            # print(x.num_node_features)
            # x = saved_into_DataObj(x)
            return x
        except Exception as e:
            print(e)
            return False
    
    def transform(self, x):
        # if self.cutoff:
        #     x = add_edges_based_distance(x, self.cutoff)
        return x


class MoleculeMemoryDataset(InMemoryDataset):
    def __init__(self, config, root = os.getcwd(),
                 transform=None, pre_filter=None, pre_transform = None):
        self.config = config
        # self.dataset_path = config.dataset_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = self.data_temp, self.slices_temp
        delattr(self, "data_temp")
        delattr(self, "slices_temp")
        # self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     files = os.listdir(self.dataset_path)
        # files = [x for x in glob(self.dataset_path + "\\*") if osp.isfile(x)]
        # if self.raw_filter is not None:
        #     files = [file for file in files if self.raw_filter(file)]
        # if self.config.max_input>0:
        #     files = random.sample(files, k = self.config.max_input)
        # print(len(files))
        # return files
    
    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self):
        return [self.config.data_file]
        
    @property
    def processed_file_names(self):
        return ["NON_EXISTENT.py"]
    # @property
    # def raw_dir(self) -> str:
    #     return self.dataset_path

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_obj(self.raw_paths[0], datapath = "", extension = "")
        print(len(data_list))
        # print(len(data_list))
        
        # print(data_list[:5])
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # print(len(data_list))
        if self.config.max_input>0 and self.config.max_input<len(data_list):
            data_list = random.sample(data_list, k = self.config.max_input)
        print(len(data_list))
            
        # print(data_list[:5])
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            # data_list = Parallel(n_jobs=self.config.procs, verbose=1,prefer="processes"
            #                      )(delayed()(data) for data in data_list)
            # data_list = [self.pre_transform(data) for data in data_list]
        print(len(data_list))
        
        data_list = [data for data in data_list if data]
        print(len(data_list))
        # os.makedirs(self.processed_dir, exist_ok=True)
        # print(data_list[:5])
        self.data_temp, self.slices_temp = self.collate(data_list)
        # print(self.data_temp)

class MoleculeDataset(Dataset):
    def __init__(self, root, dataset_path = os.getcwd(), transform=None, pre_transform=None, pre_filter = None):
        self.dataset_path = dataset_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.processed_file_names = [file for file in os.listdir(self.processed_dir) if "data_" in file]

    @property
    def raw_file_names(self):
        files = os.listdir(self.dataset_path)
        return files
    
    @property
    def raw_dir(self) -> str:
        return self.dataset_path

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(len(self.raw_file_names))]#os.listdir(self.dataset_path)

    # def download(self):
        # Download to `self.raw_dir`.
        # for file in os.listdir(self.dataset_path):
        #     copy2(osp.join(self.dataset_path, file), osp.join(self.raw_dir, file))
        # path = download_url(dataset, self.raw_dir)

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data  = load_obj(raw_path, extension = "", datapath = "")

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
    
class MoleculeRawDataset(Dataset):
    def __init__(self, root = os.getcwd(), transform=None, pre_transform=None, pre_filter = None, raw_filter = None):
        self.dataset_path = root
        self.raw_filter = raw_filter
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        out = os.listdir(self.dataset_path)
        if self.raw_filter is not None:
            out = self.raw_filter(out)
        return out
    
    @property
    def raw_dir(self) -> str:
        return self.dataset_path

    # @property
    # def processed_file_names(self):
    #     return [f"data_{i}.pt" for i in range(len(self.raw_file_names))]#os.listdir(self.dataset_path)

    # def download(self):
        # Download to `self.raw_dir`.
        # for file in os.listdir(self.dataset_path):
        #     copy2(osp.join(self.dataset_path, file), osp.join(self.raw_dir, file))
        # path = download_url(dataset, self.raw_dir)

    # def process(self):
    #     i = 0
    #     for raw_path in self.raw_paths:
    #         # Read data from `raw_path`.
    #         data  = load_obj(raw_path, extension = "", datapath = "")

    #         if self.pre_filter is not None and not self.pre_filter(data):
    #             continue

    #         if self.pre_transform is not None:
    #             data = self.pre_transform(data)

    #         torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
    #         i += 1

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        # data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        data = load_obj(self.raw_file_names[idx], datapath=self.dataset_path, extension = "")
        return data
    