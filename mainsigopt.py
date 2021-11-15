# -*- coding: utf-8 -*-
import os
import math
from model import Config
import numpy as np
import torch
from Trainer import Trainer
import time
from clintutils import SaveClass

# import cProfile, pstats, io

def main_train(config):
    config_paths = config
    config = config["suggestion"]
    config = Config(output_path = os.path.join(config_paths['output_path'],"output"),
                    dataset_path = config_paths['dataset_path'],
                    data_file = "processed_dataset_co5.0_tot10000.pkl",
                    name_model = "MXMNet_vectorssh",
                    target_cols = ["S-T", "Oscillator log", "Absoprtion wavelength"],
                    device = "cuda" if torch.cuda.is_available() else "cpu",
                    gpus = 1 if torch.cuda.is_available() else 0,
                    max_time="1:18:00:00",
                    
                    num_workers = config["num_workers"],
                    procs = config["procs"],
                    batch_size = int(config["batch_size"]),
                    splits = 0.75,
                    patience = max(30, config["gamma_epoch"]),
                    max_epochs = 400,
                    max_input = 0,
                    progress_bar = 0,
                    accelerator = None,
                    loss_weights = [1.],
                    seed = 77661237,
                    remove = ["mo", "en"],
                    smiles = False,
                    
                    dim = int(config["hidden_parameters"]),
                    # vecdim = 8,
                    outdim = int(config["outdim"]),
                    # edge_dim = 0,
                    n_layer = config["n_layer"],
                    cutoff = config["cutoff"],
                    mo_indices = [3,4],
                    # aggr = 'mean',
                    # n_rbf = 16,
                    # delta_pos = 0.1,
                    # delta_v = 1.,
                    orbs = 3,
                    # max_vec_size=100.,
                    
                    lr = config["learning_rate"],
                    weight_decay = config["weight_decay"],#1e-5,
                    delta_huber = 1.,
                    beta1 = 1.-1./config["beta_one"],
                    beta2 = 1.-1./config["beta_two"],
                    rectify = True,
                    gamma = (1./10)**(1./config["gamma_epoch"]),#1.-1./100,#0.9961697
                    dropout = config["dropout"],
                    clip_value = config["clip"],
                    clip_alg = "norm"
                    )
    config.name_model = config.name_model + "orb" + str(config.orbs) + "mo" + "".join(str(i) for i in config.mo_indices) + f"dim{config.dim}" + f"cutoff{config.cutoff}" + f"n_layer{config.n_layer}"
    config.name_model = ''.join(config.name_model.split('.'))
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

# if __name__ == "__main__":
#     config.name_model = config.name_model + "orb" + str(config.orbs) + "mo" + "".join(str(i) for i in config.mo_indices) + f"dim{config.dim}" + f"cutoff{config.cutoff}" + f"n_layer{config.n_layer}"
#     config.name_model = ''.join(config.name_model.split('.'))
    # print(f"Device = {config.device}")
    # losses, __ = main_train(config)
    # print(f"The losses were {losses}")
    