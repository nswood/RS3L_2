import sys
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union
    
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.nn.init as init
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.nn.functional as F
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfAttention#, prune_linear_layer#, gelu_new
import tqdm
from transformers.activations import gelu_new
import math
from scipy.special import beta

VERBOSE = False

'''

-------------------------
Generic Modules
-------------------------

'''

class Linear(nn.Module):
    def __init__(self, n_inputs, n_targets):
        super(Linear, self).__init__()
        self.f1 = nn.Linear(n_inputs, n_targets).cuda()
        self.activation = torch.nn.Sigmoid()
    def forward(self, x): 
        x = self.f1(x)
        return(self.activation(x))

    
'''

-------------------------
Transformer Modules
-------------------------

'''
    
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
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        self.query = nn.Linear(query_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        

    def forward(self, input_ids,  class_tokens, attention_mask=None, head_mask=None,first = False):
        #print(input_ids) 
        mixed_query_layer = self.query(class_tokens)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)
        
        if first:
            query_layer = mixed_query_layer.view(-1,self.num_attention_heads,1,self.attention_head_size)
            query_layer = query_layer.repeat(1,1,self.config.nparts,1)
        else:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = -1* attention_scores
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(input_ids.shape[0], self.num_attention_heads, attention_scores.shape[-1], attention_scores.shape[-1])
            attention_scores = attention_scores + attention_mask


        attention_probs = self.softmax_fn(attention_scores)
        if VERBOSE:
            # print(attention_probs[0, :8, :8])
            print(torch.max(attention_probs), torch.min(attention_probs))

        attention_probs = self.dropout(attention_probs)

        if False:
#         if head_mask is not None:
            attention_probs = attention_probs + head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        context_layer = context_layer.contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return (context_layer, attention_probs)




class OskarAttention(BertSelfAttention):
    def __init__(self, config, isSV=False):
        super().__init__(config)
        self.isSV = isSV
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads

        if isSV:
            self.hidden_size = int(config.hidden_size/2)
            self.attention_head_size = int(config.hidden_size/2) // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.dense = nn.Linear(int(config.hidden_size/2), int(config.hidden_size/2))
            self.LayerNorm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
        else:
            self.hidden_size = config.hidden_size
            self.attention_head_size = config.hidden_size // config.num_attention_heads
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()
        self.attention_band = config.attention_band
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        #print(input_ids) 
        mixed_query_layer = self.query(input_ids)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.attention_band is not None:
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = key_layer.permute(0, 2, 1, 3)
            value_layer = value_layer.permute(0, 2, 1, 3)

            attn_band = self.attention_band
            if attention_mask is not None:
                attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
                remove_from_windowed_attention_mask = (attention_mask != 0)
            query_layer /= math.sqrt(self.attention_head_size)
            query_layer = query_layer.float().contiguous()
            key_layer = key_layer.float().contiguous()
            if False:
                attention_scores = diagonaled_mm_tvm(
                        query_layer, key_layer,
                        attn_band,
                        1, False, 0, False # dilation, is_t1_diag, padding, autoregressive
                    )
            else:
                attention_scores = sliding_chunks_matmul_qk(
                        query_layer, key_layer,
                        attn_band, padding_value=0
                )
            mask_invalid_locations(attention_scores, attn_band, 1, False)
            if attention_mask is not None:
                remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                float_mask = remove_from_windowed_attention_mask.type_as(query_layer).masked_fill(remove_from_windowed_attention_mask, -10000.0)
                float_mask = float_mask.repeat(1, 1, 1, 1) # don't think I need this
                ones = float_mask.new_ones(size=float_mask.size())
                if False:
                    d_mask = diagonaled_mm_tvm(ones, float_mask, attn_band, 1, False, 0, False)
                else:
                    d_mask = sliding_chunks_matmul_qk(ones, float_mask, attn_band, padding_value=0)
                attention_scores += d_mask

            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
            attention_probs = self.dropout(attention_probs)

            value_layer = value_layer.float().contiguous()
            if False:
                context_layer = diagonaled_mm_tvm(attention_probs, value_layer, attn_band, 1, True, 0, False)
            else:
                context_layer = sliding_chunks_matmul_pv(attention_probs, value_layer, attn_band)

        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            #print(query_layer, key_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                
                #Current bug here, attention_mask is not 105xEmbed, only 100xEmbed
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if VERBOSE:
                # print(attention_probs[0, :8, :8])
                print(torch.max(attention_probs), torch.min(attention_probs))

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3)

        context_layer = context_layer.contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class OskarLayer(nn.Module):
    def __init__(self, config, isSV):
        super().__init__()

        self.config = config
        if isSV:
            self.full_layer_layer_norm = nn.LayerNorm(int(config.hidden_size/2), eps=config.layer_norm_eps)
            self.attention = OskarAttention(config, True)
            self.ffn = nn.Linear(int(config.hidden_size/2), config.intermediate_size)
            self.ffn_output = nn.Linear(config.intermediate_size, int(config.hidden_size/2))
        else:
            self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.attention = OskarAttention(config)
            self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
            self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        try:
            self.activation = ACT2FN[config.hidden_act]
        except KeyError:
            self.activation = config.hidden_act

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask, head_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class OskarLayerGroup(nn.Module):
    def __init__(self, config, isSV=False):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([OskarLayer(config,isSV) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index])
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
    def __init__(self, config,first = False,  isSV=False):
        super().__init__()
        self.config = config
        self.first = first
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.hidden_size
        if isSV:
            self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, int(self.output_hidden_states/2))
        else:
            if first:
                self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
            else:
                self.embedding_hidden_mapping_in = nn.Linear(config.hidden_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([OskarLayerGroup(config,isSV) for _ in range(config.num_hidden_groups)])
        
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
    
    
    
'''

-------------------------
Graph Modules
-------------------------

'''

import numpy as np
import torch
import torch.nn as nn

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)

class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        #print(torch.where(~torch.isfinite(x),1,0).nonzero())
        return self.conv(x)
