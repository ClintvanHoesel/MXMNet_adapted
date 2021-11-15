# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:45:08 2021

@author: s164097
"""

import os
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
import random
import torch
from torch.nn import LeakyReLU, Identity, ModuleList, Linear, HuberLoss, L1Loss, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
# from torch_geometric.utils import dense_to_sparse, add_self_loops, degree
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import MessagePassing, GCNConv, GraphConv, global_mean_pool, GraphNorm

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from clint_modules import NaNLoss
from clintutils import target_output_plot, plot_to_tensorboard, list_of_dictionaries_to_csv, save_obj, MetricsCallback
from adabelief import AdaBelief
from model import MXMNet, Config
from dataset import MoleculeRawDataset, TRANSFORMER, MoleculeMemoryDataset
# torch.autograd.set_detect_anomaly(True)

class Prediction(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MXMNet(self.config).to(self.device)
        # self.to(self.config.device)
        
        for key, value in self.config.__dict__.items():
            if key in ["device"]:
                continue
            setattr(self, key, value)
        
        
        
        self.head = Linear(self.outdim, self.outfeatures)
        self.act = LeakyReLU()
        # self.loss_func = NaNLoss()
        self.loss_func = HuberLoss(delta = self.delta_huber)
        self.mse_func = MSELoss()
        self.mae_func = L1Loss()
        self.loss_weights = torch.Tensor(self.loss_weights).to(self.device)
        
        self.training = False
        self.eval()
        self.model.eval()
        self.head.eval()

        self.save_hyperparameters()
        
    def forward(self, data):
        data.to(self.device)
        out = self.model(data)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.act(out)
        out = self.head(out)
        # assert not torch.sum(torch.isnan(out)), f"out after head has NaNs"
        return out

    def training_step(self, batch, batch_idx):
        batch.to(self.device)
        # batch_idx.to(self.device)
        self.loss_weights.to(self.device)

        self.train()
        self.model.train()
        self.head.train()
        
        out = self.forward(batch)
        
        target = batch.y.reshape(out.shape)
        # if torch.mean(out-target)>10:
        # print(f"out = {out}")
        # print(f"target = {target}")
        
        # loss, mae_loss = self.loss_func(out, target, self.loss_weights, power = 1)
        # ___, mse_loss = self.loss_func(out, target, self.loss_weights)
        loss = self.loss_func(out, target)
        with torch.no_grad():
            mae_loss = self.mae_func(out.detach(),target)
            mse_loss = self.mse_func(out.detach(),target)
        

        # self.log("train_loss_step", loss, prog_bar=False, logger=True, on_step = True)
        # self.log("train_mse_loss_step", mse_loss, prog_bar=False, logger=True, on_step = True)
        # self.log("train_mae_loss_step", mae_loss.detach(), prog_bar=False, logger=True, on_step = True)

        self.log("train_total_loss", loss.detach(), prog_bar=True, logger=True, on_epoch = True, on_step = True)
        self.log("train_mse_loss", mse_loss.detach(), prog_bar=True, logger=True, on_epoch = True, on_step = False)
        self.log("train_mae_loss", mae_loss.detach(), prog_bar=True, logger=True, on_epoch = True, on_step = False)
        return loss

    def validation_step(self, batch, batch_idx):
        batch.to(self.device)
        # batch_idx.to(self.device)
        self.loss_weights = self.loss_weights.to(self.device)
        
        self.eval()
        self.model.eval()
        self.head.eval()

        with torch.no_grad():
            out = self.forward(batch)
            
            target = batch.y.reshape(out.shape)
            # val_loss, val_mae_loss = self.loss_func(out, target, self.loss_weights, power = 1)
            # ___, val_mse_loss = self.loss_func(out, target, self.loss_weights)
            val_loss = self.loss_func(out.detach(), target)
            val_mae_loss = self.mae_func(out.detach(),target)
            val_mse_loss = self.mse_func(out.detach(),target)
            
        # self.log("val_loss_step", val_loss.detach(), prog_bar=False, logger=True, on_step = True)
        # self.log("val_mse_loss_step", val_mse_loss.detach(), prog_bar=False, logger=True, on_step = True)
        # self.log("val_mae_loss_step", val_mae_loss.detach(), prog_bar=False, logger=True, on_step = True)

        self.log("val_total_loss", val_loss.detach(), prog_bar=True, logger=True, on_epoch = True, on_step = True)
        self.log("val_mse_loss", val_mse_loss.detach(), prog_bar=True, logger=True, on_epoch = True, on_step = False)
        self.log("val_mae_loss", val_mae_loss.detach(), prog_bar=True, logger=True, on_epoch = True, on_step = False)
        return val_loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        optimizer = AdaBelief(self.parameters(), lr=self.lr, weight_decay = self.weight_decay,\
            betas = (self.beta1, self.beta2), rectify = self.rectify)
        # optimizer = SGD(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, factor = self.lr_factor, patience = self.lr_patience)
        scheduler = ExponentialLR(optimizer, gamma = self.gamma)
        
        return {"optimizer" : optimizer, "lr_scheduler" : {"scheduler": scheduler, "monitor" : "val_loss", "interval" : "epoch"}}


    
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    # pl.utilities.seed.seed_everything(seed)
    # torch_geometric.seed.seed_everything(seed)

class Trainer():
    def __init__(self, config):
        set_seed(config.seed)
        self.transformer = TRANSFORMER(mo_indices=config.mo_indices,
                                       remove=config.remove,
                                       orbs=config.orbs,
                                       smiles=config.smiles,
                                       cutoff = config.cutoff)
        
        self.datas = MoleculeMemoryDataset(config,
                       root = config.dataset_path,
                        pre_transform=self.transformer.pre_transform,
                        pre_filter=self.transformer.pre_filter,
                        transform=self.transformer.transform)
        
        # if config.max_input>0:
        #     indices = random.sample(self.datas.indices(), k = config.max_input)
        #     self.datas = self.datas.index_select(indices)
        
        self.config = config
        self.config.inpdim = int(self.datas.num_node_features)
        self.config.inp = self.config.inpdim
        self.config.outfeatures = int(self.datas[0].y.shape[0])
        
        for key, value in self.config.__dict__.items():
            setattr(self, key, value)
        
        self.lightning_model = Prediction
        if isinstance( self.splits, int):
            self.splitter = KFold(n_splits = self.splits, shuffle = True, random_state = config.seed)
        elif isinstance(self.splits, float):
            self.splitter = ShuffleSplit(n_splits = 1, train_size = self.splits, random_state = config.seed)


        self.models = []



    def get_dataloader(self, idxs = None, shuffle = True):
        if idxs is None:
            return DataLoader(self.datas, batch_size = self.batch_size, shuffle = shuffle,
                              num_workers = self.num_workers, persistent_workers=bool(self.num_workers),
                              pin_memory=bool(self.num_workers))
        else:
            return DataLoader(self.datas.index_select(idxs), batch_size = self.batch_size,
                              shuffle = shuffle, num_workers = self.num_workers,
                              persistent_workers=bool(self.num_workers), pin_memory=bool(self.num_workers))

    # def get_list_chromophores(self):
    #     return list(set(self.df['Chromophore'].tolist()))

    def set_loggers(self):
        self.loggers = []
        self.loggers.append(TensorBoardLogger(os.path.join(self.output_path,"tb_logs"), name=''.join(self.name_model.split('.'))))

    def set_callbacks(self, patience = 8, fold = 0):
        self.callbacks = []
        self.callbacks.append(EarlyStopping(monitor = 'val_mse_loss', patience = patience))
        self.callbacks.append(ModelCheckpoint(dirpath = os.path.join(self.output_path,'models/'),\
            filename=self.name_model+f"fold={fold}"+'{epoch}-{val_mse_loss:.2f}-{val_mae_loss:.2f}',\
            monitor = 'val_mae_loss', save_top_k = 2))
        self.callbacks.append(MetricsCallback())
        self.callbacks.append(ProgressBar(self.progress_bar))

    def predict_on_dataloader(self, dataloader, model = -1):
        out_train = None
        target_train = None
        for batch in dataloader:
            with torch.no_grad():
                new_out = self.models[model](batch).cpu().detach()
                new_target = batch.y.reshape((torch.unique(batch.batch).shape[0] ,-1)).cpu().detach()

                if out_train is None:
                    out_train = new_out
                    target_train = new_target
                else:
                    out_train = torch.cat((out_train, new_out), 0)
                    target_train = torch.cat((target_train, new_target), 0)
        return out_train, target_train
    
    def get_y_dataloader(self, dataloader):
        target_train = None
        for batch in dataloader:
            with torch.no_grad():
                new_target = batch.y.reshape((torch.unique(batch.batch).shape[0] ,-1)).cpu().detach()

                if target_train is None:
                    target_train = new_target
                else:
                    target_train = torch.cat((target_train, new_target), 0)
        return target_train
    
    def get_smiles_dataloader(self, dataloader):
        target_train = []
        for batch in dataloader:
            with torch.no_grad():
                target_train.append(batch.smile)
                # print(batch.smile)
                # print(type(batch.smile))
                # new_target = batch.smile.reshape((torch.unique(batch.batch).shape[0] ,-1)).cpu().detach()

                # if target_train is None:
                #     target_train = batch.smile
                # else:
                #     target_train = torch.cat((target_train, new_target), 0)
            # print(target_train)
        return target_train


    def train(self):
        # variables_to_log = ['train_total_loss', 'train_mse_loss', 'train_mae_loss', 'val_mse_loss', 'val_mae_loss', 'val_total_loss']
        losses = {}
        # for variable in variables_to_log:
        #     losses[variable] = []
        kfold = 0

        for train_indices, val_indices in self.splitter.split(list(range(len(self.datas)))):
            pred_output_path = os.path.join(self.output_path, "preds")
            save_obj(train_indices, 'train_dataset'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            save_obj(val_indices, 'val_dataset'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            
            # train_indices = np.array(train_indices)
            # val_indices = np.array(val_indices)
            # print(train_indices)
            # print(type(train_indices))
            
            train_dataloader = self.get_dataloader(tuple(train_indices))
            val_dataloader = self.get_dataloader(tuple(val_indices), shuffle = False)
            
            train_smiles = self.get_smiles_dataloader(train_dataloader)
            val_smiles = self.get_smiles_dataloader(val_dataloader)
            save_obj(train_smiles, 'train_smiles'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            save_obj(val_smiles, 'val_smiles'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
    
            self.set_loggers()
            self.set_callbacks(patience = self.patience, fold = kfold)
    
            # print(hidden_channels)
            self.models.append(self.lightning_model(self.config))
    
            trainer = pl.Trainer(gpus = self.gpus, max_epochs = self.max_epochs, \
                callbacks = self.callbacks, logger = self.loggers, accelerator = self.accelerator,\
                gradient_clip_val = self.clip_value, progress_bar_refresh_rate = self.progress_bar,
                gradient_clip_algorithm=self.clip_alg, max_time=self.max_time)
            trainer.fit(self.models[-1], train_dataloader, val_dataloader)
    
    
            train_dataloader = self.get_dataloader(tuple(train_indices), shuffle = False)
            val_dataloader = self.get_dataloader(tuple(val_indices), shuffle = False)
            # train_x = trainer.predict(dataloaders = train_dataloader, ckpt_path = "best")
            # val_x = self.get_y_dataloader(self, val_dataloader)
            train_x, train_y = self.predict_on_dataloader(train_dataloader, model = -1)
            val_x, val_y = self.predict_on_dataloader(val_dataloader, model = -1)
    
            save_obj(train_x, 'train_out'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            save_obj(train_y, 'train_target'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            save_obj(val_x, 'val_out'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            save_obj(val_y, 'val_target'+self.name_model+f"fold{kfold}", datapath = pred_output_path)
            
            fig = target_output_plot(train_x, train_y, val_x, val_y,\
                self.target_cols,\
                (20,10))
            img = plot_to_tensorboard(fig)
            self.loggers[0].experiment.add_image("Predictions", img)
    
    
            lookback = 1#max(min(2, self.max_epochs-1),1)#max(min(self.patience, self.max_epochs-1),1)
            print(f"Best metrics for fold {kfold} are {self.callbacks[2].metrics[-lookback]}")
            for variable in list(self.callbacks[2].metrics[-lookback].keys()):
                losses[variable] = []
            for variable in list(self.callbacks[2].metrics[-lookback].keys()):
                losses[variable].append(self.callbacks[2].metrics[-lookback][variable])
    
            # print(self.callbacks[2].metrics)
            list_of_dictionaries_to_csv(f"losses_fold{kfold}_model{self.name_model}", self.callbacks[2].metrics, datapath = os.path.join(self.output_path,"csvs/"))
            # list_of_dictionaries_to_csv(f"losses_val_step_fold{kfold}_model{self.name_model}", self.callbacks[2].metrics_val_step, datapath = os.path.join(self.output_path,"csvs/"))
            # list_of_dictionaries_to_csv(f"losses_train_step_fold{kfold}_model{self.name_model}", self.callbacks[2].metrics_train_step, datapath = os.path.join(self.output_path,"csvs/"))
            kfold += 1

        return losses