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

class CondRNNEncoder(nn.Module):
    def __init__(self, comment_input_dim, code_input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.comment_input_dim = comment_input_dim
        self.code_input_dim = code_input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        print("Emb dim ", emb_dim)
        self.word_embedding = nn.Embedding(comment_input_dim, hid_dim)
        self.tok_embedding = nn.Embedding(code_input_dim, hid_dim)
        self.rnn = nn.LSTM(hid_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, src_x, src_xprime, src_yprime):
        #print("RNNEncoder")
        #src = [src sent len, batch size]
        tok_emb = self.tok_embedding(src_yprime)
        wordx_emb = self.word_embedding(src_x)
        wordxprime_emb = self.word_embedding(src_xprime)

        tok_embedded = self.dropout(tok_emb)
        wordx_embedded = self.dropout(wordx_emb)
        wordxprime_embedded = self.dropout(wordxprime_emb)

        embedded = torch.cat((wordx_embedded, wordxprime_embedded, tok_embedded), dim=1)

        #embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        #print("outputs shape ", outputs.shape)
        #print("hidden shape", hidden.shape)
        #print("cell shape", cell.shape)

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        #outputs are always from the top hidden layer

        return hidden, cell, outputs[-1]
