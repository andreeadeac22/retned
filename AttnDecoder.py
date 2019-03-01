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

class AttnDecoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])

        self.fc = nn.Linear(hid_dim, output_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.softmax = nn.Softmax()

    def forward(self, trg, src, trg_mask, src_mask):

        #trg = [batch_size, trg sent len]
        #src = [batch_size, src sent len]
        #trg_mask = [batch size, trg sent len]
        #src_mask = [batch size, src sent len]

        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).type(torch.LongTensor).to(self.device)

        tok_emb = self.tok_embedding(trg)
        pos_emb = self.pos_embedding(pos)
        #print("pos_emb", pos_emb.shape)

        trg = self.do((tok_emb * self.scale) + pos_emb)
        #trg = self.do((tok_emb * self.scale))

        #print("trg.shape", trg.shape)

        #trg = [batch size, trg sent len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        output_trg = self.fc(trg)
        return self.softmax(output_trg)
