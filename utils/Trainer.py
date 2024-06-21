import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns



import sys

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
import h5py
import json
import matplotlib.cm as cm

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as pltq
from matplotlib.lines import Line2D

import sklearn
import numpy.random as random

import time
import utils 
import sys
import glob
# import models
from losses import contrastive_losses

# Imports neural net tools
import itertools
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,  auc
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from loguru import logger


def load_matching_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape and v.dtype == model_state_dict[k].dtype:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping key '{k}' from the loaded state dict due to shape or data type mismatch.")

    model.load_state_dict(filtered_state_dict, strict=False)

class Trainer:
    def __init__(
        self,
        model,
        train_data,
        val_data,
        optimizer,
        save_every,
        outdir, 
        max_epochs,
        args,
        scheduler
        
    ):
        self.gpu_id = args.gpu
        self.global_rank = args.rank
        self.model = model.to(self.gpu_id)
        self.name = self.model.name
        self.model.gpu_id = self.gpu_id
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)
        
        self.loss_clr= contrastive_losses.SimCLRLoss(temperature = args.temperature,base_temperature = args.temperature)
        
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.args = args
        self.scheduler = scheduler
        self.outdir = outdir
        self.max_epochs =max_epochs
        self.loss_vals_training = []
        self.loss_vals_validation = []
        self.lr_history = []
       
        try:
            self.args.anom_indices = [int(i) for i in self.args.anom_indices]
        except ValueError:
            print("Error: All elements in anom_indices must be convertible to integers.")

        
       

    def _run_epoch_val(self, epoch):
       
        with torch.no_grad():
            cspace = []
            testingLabels = []
            b_sz = len(next(iter(self.val_data))[0])
            if self.global_rank == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.val_data)}")
            loss_validation, acc_validation = [], []
            self.model.train(False)
                
            for istep, (x_pf,jet_features,label,v) in enumerate(self.val_data):
               
                if istep != len(self.val_data) - 1:
                    lv,output = self._run_batch_val(istep,x_pf, v, jet_features, label)
                    if not np.isnan(lv):
                        loss_validation.append(lv)
            
            epoch_val_loss = np.mean(loss_validation)   
            self.loss_vals_validation.append(epoch_val_loss)
            
        
        
    def _run_batch_val(self, istep, x_pf, x_sv, jet_features, label):
       
        self.model.eval()
        
        
        for param in self.model.parameters():
            param.grad = None
        
        x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
        x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
        
        for param in self.model.parameters():
            param.grad = None
        
        x_pf = x_pf.to(self.gpu_id)
        x_sv = x_sv.to(self.gpu_id)
        jet_features = jet_features.to(self.gpu_id)
        label = label.to(self.gpu_id)
            
        mask = ~torch.isin(torch.argmax(label, dim=1), torch.tensor(self.args.anom_indices).to(self.gpu_id))

        # Apply mask
        x_pf = x_pf[mask]
        x_sv = x_sv[mask]
        jet_features = jet_features[mask]
        label = label[mask]

        
        output = self.model(x_pf,jet_features = jet_features, mask = mask, training = False, v= x_sv)
        
        l = self.loss_clr.forward2(output[0].unsqueeze(1).unsqueeze(1), torch.argmax(label, dim=1))
        
        torch.cuda.empty_cache()
        return l.item(),output
        
    def _run_batch_train(self, istep, x_pf, x_sv, jet_features, label):
        self.model.train(True)
        x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
        x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
        
        for param in self.model.parameters():
            param.grad = None
        
        
        x_pf = x_pf.to(self.gpu_id)
        x_sv = x_sv.to(self.gpu_id)
        jet_features = jet_features.to(self.gpu_id)
        label = label.to(self.gpu_id)
        
        output = self.model(x_pf,jet_features = jet_features, mask = mask, training = True, v= x_sv)
            
        self.optimizer.zero_grad()
        
        l = self.loss_flat.forward2(output[0].unsqueeze(1).unsqueeze(1), torch.argmax(label, dim=1))
        l.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        return l.item()

    def _run_epoch_train(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        if self.global_rank == 0:
            print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        loss_training = []
        loss_validation = []
        self.model.train(True)
        
        for istep, (x_pf,jet_features,label,v) in enumerate(self.train_data):
            if istep != len(self.train_data) - 1:
                lt = self._run_batch_train(istep, x_pf, v, jet_features, label)
                loss_training.append(lt)
        epoch_train_loss = np.mean(loss_training)
        self.loss_vals_training.append(epoch_train_loss)
        self.scheduler.step()
    
    def _save_snapshot(self, epoch):
        torch.save(self.model.state_dict(),"{}/epoch_{}_{}_loss_{}_{}.pth".format(self.outdir,epoch,self.name.replace(' ','_'),round(self.loss_vals_training[epoch],4),round(self.loss_vals_validation[epoch],4)))
        if self.global_rank == 0:
            print(f" Training snapshot saved")
    
    def train(self, max_epochs: int):
        self.model.train(True)
        np.random.seed(max_epochs)
        random.seed(max_epochs)
        
        model_dir = self.outdir
        os.system("mkdir -p ./"+model_dir)
        os.system("mkdir -p ./"+model_dir+ '/plots')
        
        if self.args.continue_training:
            load_matching_state_dict(self.model,self.args.mpath)
            start_epoch = self.args.mpath.split("/")[-1].split("epoch_")[-1].split("_")[0]
            start_epoch = int(start_epoch) + 1
            if self.global_rank == 0:
                print(f"Continuing training from epoch {start_epoch}...")
        else:
            if not self.args.prepath == None:
                if self.global_rank == 0:
                    print('loaded pretrained model')
                load_matching_state_dict(self.model,self.args.prepath)
            start_epoch = 0
        end_epoch = max_epochs
     
        if start_epoch < max_epochs:
            for epoch in range(self.epochs_run, max_epochs):
                epoch = epoch + start_epoch
                if epoch<max_epochs:
                    self.lr_history.append(self.optimizer.param_groups[0]['lr'])
                    self._run_epoch_train(epoch)
                    self._run_epoch_val(epoch)
                    out_str = f'Train: {np.round(self.loss_vals_training[epoch],decimals = 6)} \n Val: {np.round(self.loss_vals_validation[epoch],decimals = 6)}'
                    if self.global_rank == 0:
                        print(out_str)
                            
                        if (epoch % self.save_every == 0 or epoch == max_epochs-1):
                            self._save_snapshot(epoch)                        
                    
                        
#         self.run_inference(self.val_data,self.args)
        torch.cuda.empty_cache()
    
            
    def run_inference(self, val_loader,args):
        
        with torch.no_grad():
            print("Calculating Embeddings on test data")
            inputs = []
            embed = []
            testingLabels = []

            batch_size = self.args.batchsize

            for istep, (x_pf,jet_features,label,v) in enumerate(val_loader):
                #model.eval()
                label = torch.argmax(label,dim=1)
                x_pf = x_pf.to(self.gpu_id)
                v = v.to(self.gpu_id)
                jet_features = jet_features.to(self.gpu_id)
                label = label.to(self.gpu_id)
                testingLabels.append(label.cpu().detach().numpy())
                
                
                output = self.model(x_pf,jet_features = jet_features,training = False, v= v,tan_space = True)
                
                inputs.append(x_pf.cpu().detach().numpy())
                
                embed.append(output[0].cpu().detach().numpy())
        
        embed = np.concatenate(embed, axis = 0)
            
        self.outdir = self.outdir + '/plots'
        
        inputs= np.vstack(inputs)[0:100000]
        testingLabels = np.concatenate(testingLabels,axis=0)[0:100000]
        
#         np.savez(self.outdir + '/inputs', inputs = inputs, labels=testingLabels)
#         np.savez(self.outdir + '/labels', labels=testingLabels)
        
            
            

