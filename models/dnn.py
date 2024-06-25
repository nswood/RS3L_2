import numpy as np
import torch
import torch.nn as nn
import itertools
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Defines the interaction matrices
import sys

import torch.nn.functional as F
import tqdm
import math



class DNN(nn.Module):
    def __init__(self, name, n_inputs, n_targets):
        super(DNN, self).__init__()
        #self.flat = torch.flatten()
        self.name = name
        self.dropout = nn.Dropout(p=0.25)
        self.b0 = nn.BatchNorm1d(n_inputs).cuda()
        self.f0 = nn.Linear(n_inputs, 50).cuda()
        self.f1 = nn.Linear(50, 40).cuda()
        self.f1b = nn.Linear(40, 40).cuda()
        self.b2 = nn.BatchNorm1d(40).cuda()
        self.f2 = nn.Linear(40, 10).cuda()
        self.b3 = nn.BatchNorm1d(10).cuda()
        self.f3 = nn.Linear(10, 5).cuda()
        self.b5 = nn.BatchNorm1d(5).cuda()
        #self.f4 = nn.Linear(50, 10).cuda()
        self.f5 = nn.Linear(5, n_targets).cuda()
        self.activation = torch.nn.ReLU()
        if n_targets == 2 or n_targets == 1:
            self.lastactivation = torch.nn.Sigmoid()
        elif n_targets > 2:
            self.lastactivation = torch.nn.Softmax(dim=1)
        else:
            raise ValueError("I don't understand n_targets "+str(n_targets))
    def forward(self, x): 
        #print("before flat",x.shape)
        #print("before flat",x[0])
        x = torch.flatten(x,start_dim=1)
        #print("after flat",x.shape)
        #print("after flat",x[1])
        x = self.b0(x)
        x = self.activation(self.f0(x))
        x = self.activation(self.f1(x))
        x = self.activation(self.f1b(x))
        x = self.activation(self.f2(x))
        x = self.b3(x)
        x = self.activation(self.f3(x))
        x = self.b5(x)
        x = self.f5(x)
        #return x
        return(self.lastactivation(x))

