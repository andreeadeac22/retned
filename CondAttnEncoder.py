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

class CondAttnEncoder(nn.Module):
    def __init__(self, comment_input_dim, code_input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.comment_input_dim = comment_input_dim
        self.code_input_dim = code_input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.word_embedding = nn.Embedding(comment_input_dim, hid_dim)
        self.tok_embedding = nn.Embedding(code_input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(input_dim, hid_dim)

        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
                                     for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim]))
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()

    def forward(self, src_x, src_xprime, src_yprime, src_mask):

        #src = [batch size, src sent len]
        #src_mask = [batch size, src sent len]

        src = torch.cat((src_x, src_xprime, src_yprime), dim=1)



        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).type(torch.LongTensor)
        if torch.cuda.is_available():
            pos = pos.cuda()


        tok_emb = self.tok_embedding(src_yprime)
        wordx_emb = self.word_embedding(src_x)
        wordxprime_emb = self.word_embedding(src_xprime)

        all_emb = torch.cat((wordx_emb, wordxprime_emb, tok_emb), dim=1)

        pos_emb = self.pos_embedding(pos)

        src = self.do((all_emb * self.scale) + pos_emb)

        for layer in self.layers:
            src = layer(src, src_mask)

        #print("src.shape", src.shape)

        return src
