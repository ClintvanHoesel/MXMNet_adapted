# -*- coding: utf-8 -*-
import os
from model import Config
import numpy as np
import torch
from Trainer import Trainer
import time
from clintutils import SaveClass

# import cProfile, pstats, io

def main_train(config):
    trainer = Trainer(config)
    
    start = time.time()
    losses = trainer.train()
    time_taken = time.time()-start
    
    for key in list(losses.keys()):
        # losses[key + "_median"] = np.median(losses[key])
        losses[key] = np.mean(losses[key])
        print(f"{key} : {losses[key]}")
    losses["time"] = time_taken

    # sigopt_metrics = ["val_loss_median", "val_loss", "val_mae_loss",\
    # "val_mae_loss_median", "val_mse_loss_median", "val_mse_loss",\
    # "train_loss_median", "train_loss",\
    # "train_mae_loss", "train_mae_loss_median", "train_mse_loss", "train_mse_loss_median", "time"]
    sigopt_metrics = list(losses.keys())
    sigopt_returns = list()
    for key in sigopt_metrics:
        sigopt_returns.append({'name': key, 'value': losses[key]})
    
    return sigopt_returns, None

if __name__ == "__main__":
    config = Config(output_path = os.path.join(os.getcwd(),"output"),
                    dataset_path = "Z:\\EGNN_Clint\\processed",
                    data_file = "processed_dataset_co5.0_tot10000.pkl",
                    name_model = "MXMNet_vectors",
                    target_cols = ["S-T", "Oscillator log", "Absoprtion wavelength"],
                    device = "cuda" if torch.cuda.is_available() else "cpu",
                    gpus = 1 if torch.cuda.is_available() else 0,
                    max_time="1:18:00:00",
                    
                    num_workers = 0,
                    procs = 4,
                    batch_size = 32,
                    splits = 0.75,
                    patience = 30,
                    max_epochs = 3,
                    max_input = 0,
                    progress_bar = 10,
                    accelerator = None,
                    loss_weights = [1.],
                    seed = 77661237,
                    remove = ["mo", "en"],
                    smiles = False,
                    
                    dim = 8,
                    # vecdim = 8,
                    outdim = 32,
                    # edge_dim = 0,
                    n_layer = 3,
                    cutoff = 5.,
                    mo_indices = [3,4],
                    # aggr = 'mean',
                    # n_rbf = 16,
                    # delta_pos = 0.1,
                    # delta_v = 1.,
                    orbs = 3,
                    # max_vec_size=100.,
                    
                    lr = 1e-3,
                    weight_decay = 0.,#1e-5,
                    delta_huber = 1.,
                    beta1 = 0.9,
                    beta2 = 0.999,
                    rectify = True,
                    gamma = (1./10)**(1./100),#1.-1./100,#0.9961697
                    dropout = 0.,
                    clip_value = 10.,
                    clip_alg = "norm"
                    )
    print(f"Device = {config.device}")
    config.name_model = config.name_model + "orb" + str(config.orbs) + "mo" + "".join(str(i) for i in config.mo_indices) + f"dim{config.dim}" + f"cutoff{config.cutoff}" + f"n_layer{config.n_layer}"
    config.name_model = ''.join(config.name_model.split('.'))

    losses, __ = main_train(config)
    print(f"The losses were {losses}")
    