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

class AttnEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src sent len, hid dim]
        #src_mask = [batch size, src sent len]

        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))

        src = self.ln(src + self.do(self.pf(src)))

        return src
