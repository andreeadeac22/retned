import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import os
import time

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        print("Input dim ", input_dim)
        print("Emb dim ", emb_dim)
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src):
        #print("RNNEncoder")
        #src = [src sent len, batch size]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        #embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        #print("outputs shape ", outputs.shape)
        #print("hidden shape", hidden.shape)
        #print("cell shape", cell.shape)

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell
