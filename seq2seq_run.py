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

from data_processing import *
from constants import *
from RNNEncoder import *
from RNNDecoder import *
from Seq2Seq import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, valid_data, test_data = split_data()

INPUT_DIM = src_vocab_size
OUTPUT_DIM = trg_vocab_size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = RNNEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = RNNDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('The model has {0:9d} trainable parameters'.format(count_parameters(model)))

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()


def train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()

    for j in range(0, batch_size, batch_size):
        interval = [x for x in range(j, min(train_data.shape[0], j + batch_size))]
        interval = torch.LongTensor(interval)
        if torch.cuda.is_available():
            interval = interval.cuda()
        batch = Variable(index_select(train_data, 0, interval))
        src = batch[:, :max_comment_len]
        trg = batch[:, max_comment_len+1:]

        onehot_trg = torch.FloatTensor(batch_size, max_code_len, trg_vocab_size)
        if torch.cuda.is_available():
            onehot_trg = onehot_trg.cuda()

        for i in range(trg.shape[0]):
            for j in range(max_code_len):
                if trg[i][j] > 0:
                    onehot_trg[i][j][trg[i][j]] = 1

        src = torch.transpose(src, 0, 1)
        trg = torch.transpose(trg, 0, 1)
        onehot_trg = torch.transpose(onehot_trg, 0, 1)

        optimizer.zero_grad()
        output = model(src, trg)

        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = torch.transpose(output, 0, 1)
        onehot_trg = torch.transpose(onehot_trg, 0, 1)

        loss = criterion(output, onehot_trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_data)


def evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0
    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()
    with torch.no_grad():
        for j in range(0, train_data.shape[0], batch_size):
            interval = [x for x in range(j, min(train_data.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if torch.cuda.is_available():
                interval = interval.cuda()
            batch = Variable(index_select(train_data, 0, interval))
            src = batch[:, :max_comment_len]
            trg = batch[:, max_comment_len+1:]

            onehot_trg = torch.FloatTensor(batch_size, max_code_len, trg_vocab_size)
            if torch.cuda.is_available():
                onehot_trg = onehot_trg.cuda()

            for i in range(trg.shape[0]):
                for j in range(max_code_len):
                    if trg[i][j] > 0:
                        onehot_trg[i][j][trg[i][j]] = 1

            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)
            onehot_trg = torch.transpose(onehot_trg, 0, 1)

            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = torch.transpose(output, 0, 1)
            onehot_trg = torch.transpose(onehot_trg, 0, 1)

            loss = criterion(output, onehot_trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_model.pt')

best_valid_loss = float('inf')

if not os.path.isdir('models'):
    os.makedirs('models')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model, train_data, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_data, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('| Epoch: {0:3d} | Time: {1:5d}m {2:5d}s| Train Loss: {3:.3f} | Train PPL: {4:7.3f} | Val. Loss: {5:.3f} | Val. PPL: {6:7.3f} |'.format(epoch+1, epoch_mins, epoch_secs, train_loss, math.exp(train_loss), valid_loss, math.exp(valid_loss)))

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss = evaluate(model, test_iterator, criterion)
print('| Test Loss: {0:.3f} | Test PPL: {1:7.3f} |'.format(test_loss, math.exp(test_loss)))
