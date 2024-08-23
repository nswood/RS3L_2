import numpy as np
import torch
import torch.nn as nn
import itertools
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Defines the interaction matrices
from torch.nn import Sequential, Linear

import sys


import torch.nn.functional as F
#from transformers.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer, gelu_new
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfAttention, prune_linear_layer#, gelu_new
import tqdm
from transformers.activations import gelu_new
import math
VERBOSE=False


# Import necessary model layers for transformer
from model_layers.modules import ClassAttention, OskarAttention, OskarLayer, OskarLayerGroup, OskarTransformer


class Transformer(nn.Module):
    def __init__(self, config, name, softmax, sigmoid,sv_branch=False,pretrain = False):
        super().__init__()
        #print(config)
        self.relu = gelu_new #nn.ReLU() 
        self.tanh = nn.Tanh()
        self.gpu_id = 'cuda'
        self.config = config
        self.device = device
        self.name = name
        config.output_attentions = False
        config.output_hidden_states = False
        config.num_hidden_groups = 1
        config.inner_group_num = 1
        config.layer_norm_eps = 1e-12
        config.hidden_dropout_prob = 0
        config.attention_probs_dropout_prob = 0
        config.hidden_act = "gelu_new"
        self.softmax = softmax #config
        self.sigmoid = sigmoid
        self.input_bn = nn.BatchNorm1d(config.feature_size)
        self.pretrain = pretrain
        self.hybrid = config.hybrid
        
        self.embedder = nn.Linear(config.feature_size, config.embedding_size)
        if sv_branch: 
            self.input_bn_sv = nn.BatchNorm1d(config.feature_sv_size)
            
            self.embedder_sv = nn.Linear(config.feature_sv_size, config.embedding_size)
            #Had to change config.n_out_nodes*2 to config.n_out_nodes in first input to first layer
            

            self.final_embedder_sv = nn.ModuleList([
                nn.Linear(int(config.n_out_nodes), int(config.n_out_nodes*2)),
                nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes*4)),
                nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*4)),
                nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*2)),
                nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes)),
                nn.Linear(int(config.n_out_nodes), int(config.n_out_nodes)),
                nn.Linear(int(config.n_out_nodes), config.nclasses),
                ])
            self.embed_bn_sv = nn.BatchNorm1d(config.embedding_size)
            self.encoders_sv = nn.ModuleList()
            for i in range(config.num_encoders):
                
                if i == 0:
                    print('should be first')
                    self.encoders_sv.append(OskarTransformer(config,True,False))
                else:
                    self.encoders_sv.append(OskarTransformer(config,False,False))
                
            
            
            self.decoders_sv = nn.ModuleList([
                                           nn.Linear(int(config.hidden_size), int(config.hidden_size)),
                                           nn.Linear(int(config.hidden_size), int(config.hidden_size)),
                                           nn.Linear(int(config.hidden_size), int(config.n_out_nodes))
                                           ])
    
    
    
            self.decoder_bn_sv = nn.ModuleList([nn.BatchNorm1d(int(config.hidden_size)) for _ in self.decoders_sv[:-1]])
           
        
        self.final_embedder = nn.ModuleList([
                                            nn.Linear(config.n_out_nodes, int(config.n_out_nodes/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes/2), int(config.n_out_nodes/2)),
                                            nn.Linear(int(config.n_out_nodes/2), int(config.n_out_nodes*4)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes*4), int(config.n_out_nodes*2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes*2), int(config.n_out_nodes/2)),
                                            nn.ReLU(),
                                            nn.Linear(int(config.n_out_nodes/2), config.nclasses),
                                        ])
        self.first_three_modules = self.final_embedder[:3]
        self.remaining_modules = self.final_embedder[3:]
        
        size = config.nparts
        if config.sv:
            size = size + 5
        if config.replace_mean:  
            self.pre_final = nn.ModuleList([
                                               nn.Linear(int(size), int(size/2)),
                                               nn.Linear(int(size/2), int(size/4)),
                                               nn.Linear(int(size/4), int(1))
                                               ])
            


        self.embed_bn = nn.BatchNorm1d(config.embedding_size)

        self.encoders = nn.ModuleList()
        for i in range(config.num_encoders):
                
            if i == 0:

                self.encoders.append(OskarTransformer(config,True,False))
            else:
                self.encoders.append(OskarTransformer(config,False,False))
        
                           
        self.decoders = nn.ModuleList([
                               nn.Linear(config.hidden_size, config.hidden_size),
                               nn.Linear(config.hidden_size, config.hidden_size),
                               nn.Linear(config.hidden_size, config.n_out_nodes)
                               ])
       
        self.decoder_bn = nn.ModuleList([nn.BatchNorm1d(config.hidden_size) for _ in self.decoders[:-1]])
        #self.pooling = torch.mean()
        self.tests = nn.ModuleList(
                    [
                      nn.Linear(config.feature_size, 1, bias=False),
                      # nn.Linear(config.feature_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, config.hidden_size),
                      # nn.Linear(config.hidden_size, 1)
                    ]
                    )

        self.config = config
        print(self.decoders)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, sv=None,  mask=None, sv_mask=None):
        
        if mask is None:
            mask = torch.ones(x.size()[:-1], device=self.device)
        if len(mask.shape) == 3:
            attn_mask = mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
        else:
            attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]

        #print(x)
        attn_mask = (1 - attn_mask) * -1e9
        if self.config.mname is not None:
            attn_mask = attn_mask.to(self.device)
        head_mask = [None] * self.config.num_hidden_layers
	
	# Embed x
        
        x = self.input_bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        h = self.embedder(x)
        
        h = torch.relu(h)
        h = self.embed_bn(h.permute(0, 2, 1)).permute(0, 2, 1)
        
	# If sv not None, embed sv and concat to x
        if sv is not None:
            
            
            if sv_mask is None:
                
                
                x1 = x.shape[1]
                sv1 = sv.shape[1]
                
                #Used trick to get right shape via hstack to fix mask size bug
#                 concat_mask_size = torch.rand(config.embedding_size, (x1+sv1)).size()
#                 Used trick to get right shape via hstack to fix mask size bug
                concat_mask_size = torch.hstack([x[:,:,0],sv[:,:,0]]).size()
#                 print(concat_mask_size)
                
                sv_mask = torch.ones(concat_mask_size, device=self.device)
                
            if len(sv_mask.shape) == 3:
                attn_sv_mask = sv_mask.unsqueeze(1) # [B, P, P] -> [B, 1, P, P]
            else:
                attn_sv_mask = sv_mask.unsqueeze(1).unsqueeze(2) # [B, P] -> [B, 1, P, 1]
            attn_sv_mask = (1 - attn_sv_mask) * -1e9
            head_sv_mask = [None] * self.config.num_hidden_layers

            x = self.input_bn_sv(sv.permute(0, 2, 1)).permute(0, 2, 1)
            
            j = self.embedder_sv(x)
            j = torch.relu(j)
            j = self.embed_bn_sv(j.permute(0, 2, 1)).permute(0, 2, 1)
           
	    #Now j is hte embedded version of sv and we should stack 
            h = torch.cat((h,j),dim=1)
	    
	    #Shape now is 105xembedding size	
           
            #h is now concatented x and sv, so we process normally to final step	
            for e in self.encoders_sv:
            #print(h,attn_mask,head_mask)
            	h = e(h, attn_sv_mask, head_mask)[0]
            h = self.decoders_sv[0](h)
            h = self.relu(h)
            h = self.decoder_bn_sv[0](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders_sv[1](h)
            h = self.relu(h)
            h = self.decoder_bn_sv[1](h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.decoders_sv[2](h)
            #print("before h.shape",h.shape)
#             print(h.size())
            
            #============================================
            if self.config.gromov:
                return h
            if self.config.replace_mean:
                h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
                h = self.pre_final[0](h)
                h = self.pre_final[1](h)
                h = self.pre_final[2](h)
                h = torch.squeeze(h,dim =2)
                
            else:
                h = torch.mean(h,dim=1)
           
           
            

            

            for module in self.first_three_modules:
                 h =module(h)
            if self.pretrain:
                return h
            elif self.hybrid: 
                cspace = h
            for module in self.remaining_modules:
                h =module(h)


        else:
	    # shape is now 100 x embedding size 
            
            for e in self.encoders:
            #print(h,attn_mask,head_mask)
            	h = e(h, attn_mask, head_mask)[0]
            h = self.decoders[0](h)
            h = self.relu(h)
            h = self.decoder_bn[0](h.permute(0, 2, 1)).permute(0, 2, 1)

            h = self.decoders[1](h)
            h = self.relu(h)
            h = self.decoder_bn[1](h.permute(0, 2, 1)).permute(0, 2, 1)
            
            
            h = self.decoders[2](h)
            
            
            if self.config.gromov:
                return h
            if self.config.replace_mean:
                h = torch.reshape(h,(h.shape[0],h.shape[2],h.shape[1]))
                h = self.pre_final[0](h)
                h = self.pre_final[1](h)
                h = self.pre_final[2](h)
                h = torch.squeeze(h,dim =2)
            else:
                h = torch.mean(h,dim=1)
           
        
#             h = nn.BatchNorm1d(self.config.n_out_nodes).to(self.gpu_id)(h)
            
            for module in self.first_three_modules:
                 h =module(h)
            if self.pretrain:
                return h
            elif self.hybrid: 
                cspace = h
            for module in self.remaining_modules:
                h =module(h)
            
            
      
        if self.softmax:
            h = nn.Softmax(dim=1)(h)
        if self.sigmoid:
            h = nn.Sigmoid()(h)
        #sys.exit(1)
        if  self.hybrid:
            return h, cspace
        return h

