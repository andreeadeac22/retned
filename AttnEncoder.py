from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch import index_select

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import os
import time

class AttnEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(input_dim, hid_dim)

        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()

    def forward(self, src, src_mask):

        #src = [batch size, src sent len]
        #src_mask = [batch size, src sent len]

        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).type(torch.LongTensor)
        if torch.cuda.is_available():
            pos = pos.cuda()


        tok_emb = self.tok_embedding(src)
        pos_emb = self.pos_embedding(pos)

        src = self.do((tok_emb * self.scale) + pos_emb)
        #src = self.do((tok_emb * self.scale))

        #src = [batch size, src sent len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        #print("src.shape", src.shape)

        return src
