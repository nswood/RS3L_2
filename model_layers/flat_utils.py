import sys
import matplotlib.pyplot as plt
import geoopt
from geoopt.layers.stereographic import Distance2StereographicHyperplanes
from geoopt.manifolds.stereographic.math import arsinh, artanh,artan_k
from typing import List, Optional, Tuple, Union
    
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.nn.init as init
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from frechetmean import Poincare
from riemannian_batch_norm import RiemannianBatchNorm

import torch.nn.functional as F
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfAttention#, prune_linear_layer#, gelu_new
import tqdm
from transformers.activations import gelu_new
import math
from scipy.special import beta

VERBOSE = False

class ClassAttention(BertSelfAttention):
    def __init__(self, config,query_size, isSV=False,scale = 1,):
        super().__init__(config)
        self.isSV = isSV
        self.config = config
        self.softmax_fn = nn.Softmax(dim=-1)

        self.output_attentions = config.output_attentions
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(scale * config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if isSV:
            self.hidden_size = int(config.hidden_size/2)
            self.attention_head_size = int(config.hidden_size/2) // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.dense = nn.Linear(int(config.hidden_size/2), int(config.hidden_size/2))
            self.LayerNorm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
        else:
            self.hidden_size = config.hidden_size*scale
            self.attention_head_size = self.hidden_size // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#             self.dense = nn.Linear(self.hidden_size, self.hidden_size)
#             self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        self.query = nn.Linear(query_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        

    def forward(self, input_ids,  class_tokens, attention_mask=None, head_mask=None,first = False):
        #print(input_ids) 
        mixed_query_layer = self.query(class_tokens)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        
#         query_layer = self.transpose_for_scores(mixed_query_layer)
        if first:
            query_layer = mixed_query_layer.view(-1,self.num_attention_heads,1,self.attention_head_size)
            query_layer = query_layer.repeat(1,1,self.config.nparts,1)
        else:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        #print(query_layer, key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = (query_layer.unsqueeze(-2)-key_layer.unsqueeze(-2).transpose(2, 3)).norm(dim = -1, p = -2)**(0.5)
        attention_scores = -1* attention_scores
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(input_ids.shape[0], self.num_attention_heads, attention_scores.shape[-1], attention_scores.shape[-1])
            attention_scores = attention_scores + attention_mask


        attention_probs = self.softmax_fn(attention_scores)
        if VERBOSE:
            # print(attention_probs[0, :8, :8])
            print(torch.max(attention_probs), torch.min(attention_probs))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if False:
#         if head_mask is not None:
            attention_probs = attention_probs + head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        
        # Should find a better way to do this
#         w = (
#             self.dense.weight.t()
#             .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
#             .to(context_layer.dtype)
#         )
#         b = self.dense.bias.to(context_layer.dtype)

#         projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        #return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)
        return (context_layer, attention_probs)




class OskarAttention(BertSelfAttention):
    def __init__(self, config, isSV=False,scale = 1):
        super().__init__(config)
        self.isSV = isSV
        self.softmax_fn = nn.Softmax(dim=-1)

        self.output_attentions = config.output_attentions
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(scale * config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if isSV:
            self.hidden_size = int(config.hidden_size/2)
            self.attention_head_size = int(config.hidden_size/2) // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.dense = nn.Linear(int(config.hidden_size/2), int(config.hidden_size/2))
            self.LayerNorm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
        else:
            self.hidden_size = config.hidden_size*scale
            self.attention_head_size = self.hidden_size // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#             self.dense = nn.Linear(self.hidden_size, self.hidden_size)
#             self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        #print(input_ids) 
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
       
        # Take the dot product between "query" and "key" to get the raw attention scores.
        #print(query_layer, key_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = (query_layer.unsqueeze(-2)-key_layer.unsqueeze(-2).transpose(2, 3)).norm(dim = -1, p = -2)**(0.5)
        attention_scores = -1* attention_scores
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(input_ids.shape[0], self.num_attention_heads, attention_scores.shape[-1], attention_scores.shape[-1])
            
            attention_scores = attention_scores + attention_mask


        attention_probs = self.softmax_fn(attention_scores)
        if VERBOSE:
            # print(attention_probs[0, :8, :8])
            print(torch.max(attention_probs), torch.min(attention_probs))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if False:
#         if head_mask is not None:
            attention_probs = attention_probs + head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # Should find a better way to do this
#         w = (
#             self.dense.weight.t()
#             .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
#             .to(context_layer.dtype)
#         )
#         b = self.dense.bias.to(context_layer.dtype)

#         projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        #return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)
        return (context_layer, attention_probs)



class OskarLayer(nn.Module):
    def __init__(self, config, isSV,scale):
        super().__init__()

        self.config = config
        if isSV:
            self.full_layer_layer_norm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
            self.attention = OskarAttention(config, True,scale)
            self.ffn = nn.Linear(int(scale*config.hidden_size/2), scalec*config.intermediate_size)
            self.ffn_output = nn.Linear(scale*config.intermediate_size, int(config.hidden_size/2))
        else:
            self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.attention = OskarAttention(config,scale = 1)
#             self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
#             self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        try:
            self.activation = ACT2FN[config.hidden_act]#nn.GLU()#
        except KeyError:
            self.activation = config.hidden_act

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
#         ffn_output = self.ffn(attention_output[0])
#         ffn_output = self.activation(ffn_output)
#         ffn_output = self.ffn_output(ffn_output)
#         hidden_states = attention_output[0]
        hidden_states = self.full_layer_layer_norm(hidden_states + attention_output[0])

        return (hidden_states,) #+ attention_output[1:]  # add attentions if we output them


class OskarLayerGroup(nn.Module):
    def __init__(self, config, isSV=False,scale = 1):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([OskarLayer(config,isSV,scale) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask)
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class OskarTransformer(nn.Module):
    def __init__(self, config,first = False,  isSV=False,scale = 1):
        super().__init__()
        self.config = config
        self.first = first
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.hidden_size
        if isSV:
            self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, int(self.output_hidden_states/2))
        else:
            if first:
                self.embedding_hidden_mapping_in = nn.Linear(scale *config.embedding_size, scale * config.hidden_size)
            else:
                self.embedding_hidden_mapping_in = nn.Linear(scale *config.hidden_size, scale *config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([OskarLayerGroup(config,isSV,scale) for _ in range(config.num_hidden_groups)])
        
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        
        #Problem here with matching
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        
        all_attentions = ()
        
        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
    
    