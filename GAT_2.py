# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation 
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen, 2020
# ------------------------------------------------------------

import math
from numpy.core.fromnumeric import repeat
import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class IntrativeSelfattention(nn.Module):
    def __init__(self, embed_size, h, is_share, drop=None) -> None:
        super(IntrativeSelfattention, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear] 
        else:
            # self.linears = clones(nn.Linear(embed_size, embed_size), 3)
            self.query = nn.Linear(embed_size, embed_size)
            self.key = nn.Linear(embed_size, embed_size)
            self.value = nn.Linear(embed_size, embed_size)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)
    def transpose_for_scores1(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        nbatches = q.size(0)
        nobjects = q.size(1)
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        query_head = self.transpose_for_scores1(query)
        # node_query = query_head.reshape(nbatches * self.h, query_head.size()[2], query_head.size()[3])
        key_head = self.transpose_for_scores1(key)
        # node_key = key_head.reshape(nbatches * self.h, key_head.size()[2], key_head.size()[3])
        value_head = self.transpose_for_scores1(value)



        scores = torch.matmul(query_head, key_head.transpose(-2, -1)) \
             / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0,  -np.inf)
            # scores = scores.masked_fill(mask == 0, -1e9)
        

        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value_head)
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return x

class IGSAN1(nn.Module):       
    def __init__(self, num_layers, embed_size, h=1, is_share=False, drop=None):
        super(IGSAN1, self).__init__()
        self.num_layers = num_layers
        self.bns = clones(nn.BatchNorm1d(embed_size), num_layers)
        self.dropout = clones(nn.Dropout(drop), num_layers)
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.att_layers = clones(IntrativeSelfattention(embed_size, h, is_share, drop=drop), num_layers)

        # self.fc_in = nn.Linear(embed_size, embed_size)
        # self.bn_in = nn.BatchNorm1d(embed_size)
        # self.dropout_in = nn.Dropout(0.2)

        # self.fc_int = nn.Linear(embed_size, embed_size)

        # self.fc_out = nn.Linear(embed_size, embed_size)
        # self.bn_out = nn.BatchNorm1d(embed_size)
        # self.dropout_out = nn.Dropout(0.2)
    def forward(self, q, k, v, mask=None):
        ''' imb_emb -- (bs, num_r, dim), pos_emb -- (bs, num_r, num_r, dim) '''
        bs, num_r, emb_dim = q.size()

        
        # 1st layer
        x = self.att_layers[0](q, k, v, mask)    #(bs, r, d)
        # attention_output = self.fc_in(attention_output)
        # attention_output = self.dropout_in(attention_output)
        # attention_output = self.bn_in((attention_output + q).permute(0, 2, 1)).permute(0, 2, 1)
        # intermediate_output = self.fc_int(attention_output)
        # intermediate_output = F.relu(intermediate_output)
        # intermediate_output = self.fc_out(intermediate_output)
        # intermediate_output = self.dropout_out(intermediate_output)
        # graph_output = self.bn_out((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)
        x = (self.bns[0](x.view(bs*num_r, -1))).view(bs, num_r, -1) 
        agsa_emb = q + self.dropout[0](x)

        # # 2nd~num_layers
        # for i in range(self.num_layers - 1):
        #     x = self.att_layers[i+1](agsa_emb, mask) #(bs, r, d)
        #     x = (self.bns[i+1](x.view(bs*num_r, -1))).view(bs, num_r, -1) 
        #     agsa_emb = agsa_emb + self.dropout[i+1](x)

        return agsa_emb
