"""
Input: x, x', y'
Output: y
Attentive encoder - decoder
"""

from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import index_select
import argparse

from nltk.translate.bleu_score import sentence_bleu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spacy

import random
import math
import os
import time

from annoy import AnnoyIndex

from data_processing import *
from attn_constants import *
from AttnEncoder import *
from AttnDecoder import *
from AttnEncoderLayer import *
from AttnDecoderLayer import *
from SelfAttention import *
from PositionwiseFeedforward import *
from AttnSeq2Seq import *
from NoamOpt import *


def train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    batch_num = 0
    #encoded_train_data = torch.zeros(train_data.shape[0], HID_DIM)
    #print("encoded_train_data.shape", encoded_train_data.shape)

    #train_losses = []

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()
        #encoded_train_data = encoded_train_data.cuda()

    for j in range(0, train_data.shape[0], batch_size): #TODO: replace batch_size with train_data.shape[0]
        if j+ batch_size < train_data.shape[0]:
            batch_num +=1
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
                for k in range(max_code_len):
                    if trg[i][k] > 0:
                        onehot_trg[i][k][trg[i][k]] = 1

            #src = torch.transpose(src, 0, 1)
            #trg = torch.transpose(trg, 0, 1)
            #onehot_trg = torch.transpose(onehot_trg, 0, 1)

            optimizer.zero_grad()
            
            #output, encoded = model(src, trg)
            output = model(src, trg)

            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]

            output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
            trg = torch.reshape(trg, (batch_size*max_code_len,))

            #print("trg.shape", trg.shape)
            #print("output.shape", output.shape)

            loss = criterion(output, trg)

            #train_losses += [loss.item()]

            #print("Train loss", loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    #return epoch_loss / batch_num, encoded_train_data, train_losses
    #return epoch_loss / batch_num, encoded_train_data
    return epoch_loss / batch_num



def evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0

    #encoded_valid_data = torch.zeros(valid_data.shape[0], HID_DIM)
    #print("encoded_valid_data.shape", encoded_valid_data.shape)

    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()
        #encoded_valid_data = encoded_valid_data.cuda()

    #validation_losses = []

    with torch.no_grad():
        batch_num = 0
        for j in range(0, valid_data.shape[0], batch_size):
            if j+ batch_size < valid_data.shape[0]:
                batch_num +=1
                interval = [x for x in range(j, min(valid_data.shape[0], j + batch_size))]
                interval = torch.LongTensor(interval)
                if torch.cuda.is_available():
                    interval = interval.cuda()
                batch = Variable(index_select(valid_data, 0, interval))
                src = batch[:, :max_comment_len]
                trg = batch[:, max_comment_len+1:]

                onehot_trg = torch.FloatTensor(batch_size, max_code_len, trg_vocab_size)
                if torch.cuda.is_available():
                    onehot_trg = onehot_trg.cuda()

                for i in range(trg.shape[0]):
                    for k in range(max_code_len):
                        if trg[i][k] > 0:
                            onehot_trg[i][k][trg[i][k]] = 1

                #src = torch.transpose(src, 0, 1)
                #trg = torch.transpose(trg, 0, 1)
                #onehot_trg = torch.transpose(onehot_trg, 0, 1)


                #output, encoded = model(src, trg, 0)


                output = model(src, trg)

                # output shape is code_len, batch, trg_vocab_size

                #encoded = encoded.squeeze(0)

                #print("output.shape ", output.shape)
                #print("encoded.shape ", encoded.shape)

                #for cpj in range(encoded.shape[0]):
                #    encoded_valid_data[j+cpj] = encoded[cpj]

                #trg = [trg sent len, batch size]
                #output = [trg sent len, batch size, output dim]

                output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
                trg = torch.reshape(trg, (batch_size*max_code_len,))

                #output = torch.transpose(output, 0, 1)
                #onehot_trg = torch.transpose(onehot_trg, 0, 1)

                loss = criterion(output, trg)
                #print("Valid loss", loss.item())
                #validation_losses += [loss]
                epoch_loss += loss.item()
                print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    #return epoch_loss / batch_num, encoded_valid_data, validation_losses
    #return epoch_loss / batch_num, encoded_valid_data
    return epoch_loss / batch_num



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def create_annoy_index(fn, latent_space_vectors, num_trees=30):
    # Mapping comment to annoy id, so that we can then find similar comments easily
    # size is number of training samples -- this changes for train/valid/test! how does this influence
    # num_trees is a hyperparameter
    fn_annoy = fn + '.annoy'

    ann = AnnoyIndex(latent_space_vectors.shape[1]) #HID_DIM
    for i in range(latent_space_vectors.shape[0]):
        ann.add_item(i, latent_space_vectors[i])

    ann.build(num_trees)
    ann.save(fn_annoy)

    return ann



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_model', default="RNNEncoder")
    parser.add_argument('--dec_model', default="RNNDecoder")
    parser.add_argument('--input_dim', default=src_vocab_size)
    parser.add_argument('--output_dim', default=trg_vocab_size)
    parser.add_argument('--enc_emb_dim', default=256)
    parser.add_argument('--dec_emb_dim', default=256)
    parser.add_argument('--hid_dim', default=512)
    parser.add_argument('--n_layers', default=2)
    parser.add_argument('--enc_dropout', default=0.5)

    SEED = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    train_data, valid_data, test_data = split_data()

    enc = AttnEncoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, AttnEncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    dec = AttnDecoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, AttnDecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    pad_idx = 0
    model = AttnSeq2Seq(enc, dec, pad_idx, device).to(device)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    print('The model has {0:9d} trainable parameters'.format(count_parameters(model)))

    #optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    if not os.path.isdir('models'):
        os.makedirs('models')

    valid_losses = []
    train_losses = []
    times = []


    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_data, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_data, criterion)

        train_losses += [train_loss]
        valid_losses += [valid_loss]

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        times += [end_time - start_time]
        print('| Epoch: {0:3d} | Time: {1:5d}m {2:5d}s| Train Loss: {3:.3f} | Train PPL: {4:7.3f} | Val. Loss: {5:.3f} | Val. PPL: {6:7.3f} |'.format(epoch+1, epoch_mins, epoch_secs, train_loss, math.exp(train_loss), valid_loss, math.exp(valid_loss)))


    plt.title("Loss vs Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    print("train_losses", train_losses)
    plt.plot(range(1,N_EPOCHS+1),train_losses,label="Train")
    plt.plot(range(1,N_EPOCHS+1),valid_losses,label="Validation")

    plt.legend()
    plt.savefig("results/attn_loss_epochs")

    with open("results/attn_train_losses.pickle", 'wb') as f:
        pickle.dump(train_losses, f)

    with open("results/attn_valid_losses.pickle", 'wb') as g:
        pickle.dump(valid_losses, g)

    with open("results/attn_times.pickle", 'wb') as h:
        pickle.dump(times, h)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    #test_loss, enc_test_vect, test_losses = evaluate(model, test_data, criterion)
    test_loss = evaluate(model, test_data, criterion)
    print('| Test Loss: {0:.3f} | Test PPL: {1:7.3f} |'.format(test_loss, math.exp(test_loss)))




if __name__== "__main__":
    main()
