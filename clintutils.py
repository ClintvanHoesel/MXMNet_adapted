import collections
import csv
import enum
import math
from math import ceil
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import pysmiles
import random
import seaborn as sns
import sklearn
from sklearn.preprocessing import OneHotEncoder
# import torch
# import torch.nn.functional as F
# import torch_geometric
# from torch_geometric.data import Data
import tqdm

import pytorch_lightning as pl
import copy

import logging

class SaveClass():
    def __init__(self, x, y, edge_index, edge_index_g, pos, mo, en, smile, experiment):
        self.x = x
        self. y = y
        self.edge_index = edge_index
        self.edge_index_g = edge_index_g
        self.pos = pos
        self.mo = mo
        self.en = en
        self.smile = smile
        self.experiment = experiment

class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []
        self.metrics_train_step = []
        self.metrics_val_step = []

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     metric = copy.deepcopy(trainer.callback_metrics)
    #     new_metric = {}
    #     for key in list(metric.keys()):
    #         if 'step' not in key:
    #             continue
    #         if "train" in key:
    #             try:
    #                 new_metric[key] = metric[key].cpu().numpy()[0]
    #             except:
    #                 new_metric[key] = metric[key].cpu().numpy()

    #     self.metrics_train_step.append(new_metric)

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     metric = copy.deepcopy(trainer.callback_metrics)
    #     new_metric = {}
    #     for key in list(metric.keys()):
    #         if 'step' not in key:
    #             continue
    #         if "val" in key:
    #             try:
    #                 new_metric[key] = metric[key].cpu().numpy()[0]
    #             except:
    #                 new_metric[key] = metric[key].cpu().numpy()

    #     self.metrics_val_step.append(new_metric)

    # def on_train_epoch_end(self, trainer, pl_module):
    #     metric = copy.deepcopy(trainer.callback_metrics)
    #     new_metric = {}
    #     for key in list(metric.keys()):
    #         if 'step' in key:
    #             continue
    #         if "train" in key:
    #             try:
    #                 new_metric[key] = metric[key].cpu().tolist()[0]
    #             except:
    #                 new_metric[key] = metric[key].cpu().tolist()
    #     self.metrics.append(new_metric)
                    
    def on_validation_epoch_end(self, trainer, pl_module):
        metric = copy.deepcopy(trainer.callback_metrics)
        new_metric = {}
        for key in list(metric.keys()):
            if 'step' in key:
                continue
            # if "val" in key:
            try:
                new_metric[key] = metric[key].cpu().numpy()[0]
            except:
                new_metric[key] = metric[key].cpu().numpy()

        self.metrics.append(new_metric)

def live_plot(data_dict, figsize=(7,5), title=''):
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show();

def r_squared(pred, target, power = 2):
    target_mean = np.nanmean(target)
    SS_model = np.nansum((np.abs(pred - target))**power)
    SS_mean = np.nansum((np.abs(target_mean - target))**power)
    return 1 - SS_model/SS_mean

def mean_power_error(pred, target, power = 2, normalise = False, logar = False):
    error = np.nanmean((np.abs(pred - target))**power)
    if normalise:
        std = np.nanstd(target)
        error = error/std
    if logar:
        error = np.log(error)
    return error

def target_output_plot(x, y, x_test=None, y_test = None, labels = None, figsize=(20, 10), title='', size_dots = (0.5,0.5), useTex = False):
    plt.rcParams.update({
    "text.usetex": useTex})
    if labels is None:
        labels = list(range(x.shape[1]))
    fig, axs = plt.subplots(2, ceil(x.shape[1]/2), figsize = figsize)
    for i in range(x.shape[1]):
        axs[i%2,i//2].scatter(x[:,i].detach().numpy(),y[:,i].detach().numpy(), s=size_dots[0])
        if x_test is not None:
            axs[i%2,i//2].scatter(x_test[:,i].detach().numpy(), y_test[:,i].detach().numpy(), s=size_dots[1])
        LINE_ENDS=1e10
        axs[i%2,i//2].plot([-LINE_ENDS,LINE_ENDS],[-LINE_ENDS,LINE_ENDS], c='black', scalex = False, scaley = False, linewidth=0.5)
        
        axs[i%2,i//2].set_xlabel('Predicted')
        axs[i%2,i//2].set_ylabel('Actual')
        
        if x_test is not None:
            axs[i%2,i//2].set_xlim([np.min(np.concatenate((x[:,i].detach().numpy(),x_test[:,i].detach().numpy()))),np.max(np.concatenate((x[:,i].detach().numpy(),x_test[:,i].detach().numpy())))])
            axs[i%2,i//2].set_ylim([np.nanmin(np.concatenate((y[:,i].detach().numpy(),y_test[:,i].detach().numpy()))),np.nanmax(np.concatenate((y[:,i].detach().numpy(),y_test[:,i].detach().numpy())))])
        else:
            axs[i%2,i//2].set_xlim([np.min(x[:,i].detach().numpy()),np.max(x[:,i].detach().numpy())])
            axs[i%2,i//2].set_ylim([np.nanmin([y[:,i].detach().numpy()]),np.nanmax(y[:,i].detach().numpy())])
        
        if x_test is not None:
            axs[i%2,i//2].set_title(f"{labels[i]} | $R^2_{{test}}$ = {r_squared(x_test[:,i].detach().numpy(),y_test[:,i].detach().numpy()):.3f} | $R^2_{{train}}$ = {r_squared(x[:,i].detach().numpy(),y[:,i].detach().numpy()):.3f} ")
        else:
            axs[i%2,i//2].set_title(f"{labels[i]} | $R^2_{{train}}$ = {r_squared(x[:,i].detach().numpy(),y[:,i].detach().numpy()):.3f} ")

        axs[i%2,i//2].legend(['truth', 'train'] if x_test is None else ['truth', 'train', 'test'])
#     plt.legend(loc='center left') # the plot evolves to the right
    return fig

def plot_to_tensorboard(fig):
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # print(img.shape)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = img / 255.0
    # print(img.shape)
    img = np.swapaxes(img, 0, 2)
    # print(img.shape)
    img = np.rot90(img, axes = (1, 2), k=3)
    img = img[:,:,::-1]
    # print(img.shape)
    plt.close(fig)
    return img

# class PairData(Data):
#     def __init__(self, edge_index=None, x=None, edge_index_solv=None, x_solv=None, y=None):
#         super().__init__()
#         self.edge_index = edge_index
#         self.x = x
#         self.edge_index_solv = edge_index_solv
#         self.x_solv = x_solv
#         self.y = y
#     def __inc__(self, key, value, *args, **kwargs):
#         if key == 'edge_index':
#             return self.x.size(0)
#         if key == 'edge_index_solv':
#             return self.x_solv.size(0)
#         else:
#             return super().__inc__(key, value, *args, **kwargs)

# def mse_loss_with_nans_target(x ,y):
#     y[y!=y] = x[y!=y]
#     out = F.mse_loss(x, y)
#     out_fix = torch.zeros_like(out)
#     return out

def fill_pandas_NaNs_with_mean_per_group(df, target_cols = [], group = "Chromophore"):
    for col in target_cols:
        df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
    return df

def normalize_cols(df, cols = []):
    dictionary = {}
    for col in cols:
        dictionary[col] = (df[col].mean(), df[col].std())
        df[col] = (df[col] - df[col].mean())/df[col].std()
    return df, dictionary

def denormalize_col(arr, mean, std):
    arr = arr*std
    arr = arr + mean
    return arr

def save_obj(obj, name, datapath = 'objects/'):
    filename = os.path.join(datapath,name + '.pkl')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, datapath = 'objects/', extension = ".pkl"):
    f = open(os.path.join(datapath,name + extension), 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def list_of_dictionaries_to_csv(name, list_of_dict, datapath = 'csvs/'):
    filename = os.path.join(datapath,name + '.csv')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, set(list_of_dict[-1].keys())|set(list_of_dict[0].keys()))
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dict)



def plot_mean(x_dict, columns, cols = 3, a_fill = 0.4, a_lines = 0.3, figsize = (20,20)):
    rows = cols
    fig, axs = plt.subplots(ceil(len(columns)/rows), rows, figsize = figsize)
    for i, col in enumerate(columns):
        mean = np.nanmean(x_dict[col], axis = 1)
        std = np.nanstd(x_dict[col], axis = 1)
        x_list = list(range(len(x_dict[col])))
        
        axs[i//rows,i%rows].plot(x_list,mean)
        axs[i//rows,i%rows].plot(x_list, x_dict[col], alpha = a_lines)
        axs[i//rows,i%rows].fill_between(x_list, mean-std, mean+std, alpha = a_fill)
        
        axs[i//rows,i%rows].set_xlabel('Epoch')
        axs[i//rows,i%rows].set_ylabel(col)
    return fig