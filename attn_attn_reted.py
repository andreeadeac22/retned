
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch import index_select
import argparse

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

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
from reted_constants import *
from AttnEncoder import *
from AttnDecoder import *
from CondAttnEncoder import *
from editor import *
from AttnEncoderLayer import *
from AttnDecoderLayer import *
from SelfAttention import *
from PositionwiseFeedforward import *
from NoamOpt import *
from AttnSeq2Seq import *



def ret_train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    batch_num = 0
    encoded_train_data = torch.zeros(train_data.shape[0], max_comment_len*HID_DIM)

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()
        encoded_train_data = encoded_train_data.cuda()

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

            optimizer.zero_grad()
            output, encoded = model(src, trg)
            # output shape is code_len, batch, trg_vocab_size

            encoded = torch.reshape(encoded, (batch_size, max_comment_len*HID_DIM))

            for cpj in range(encoded.shape[0]):
                encoded_train_data[j+cpj] = encoded[cpj]

            output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
            trg = torch.reshape(trg, (batch_size*max_code_len,))

            loss = criterion(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    return epoch_loss / batch_num, encoded_train_data


def ed_train(model, train_data, sim_train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    batch_num = 0

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()
        encoded_train_data = encoded_train_data.cuda()

    for j in range(0, train_data.shape[0], batch_size): #TODO: replace batch_size with train_data.shape[0]
        if j+ batch_size < train_data.shape[0]:
            batch_num +=1
            interval = [x for x in range(j, min(train_data.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if torch.cuda.is_available():
                interval = interval.cuda()
            batch = Variable(index_select(train_data, 0, interval))
            x= batch[:, :max_comment_len]
            xprime = batch[:, max_comment_len+max_code_len+1 : max_comment_len*2+max_code_len]
            yprime = batch[:, max_comment_len*2+max_code_len+1:]

            # src = torch.cat((x, xprime, yprime), dim=1)
            # print("src.shape", src.shape)

            trg = batch[:, max_comment_len+1:max_comment_len+max_code_len] # y


            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)

            optimizer.zero_grad()
            output, encoded = model(x, xprime, yprime, trg)
            # output shape is code_len, batch, trg_vocab_size

            encoded = encoded.squeeze(0)


            for cpj in range(encoded.shape[0]):
                encoded_train_data[j+cpj] = encoded[cpj]

            output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
            trg = torch.reshape(trg, (batch_size*max_code_len,))

            loss = criterion(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    return epoch_loss / batch_num, encoded_train_data


def ret_evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0

    encoded_valid_data = torch.zeros(valid_data.shape[0], HID_DIM)
    #print("encoded_valid_data.shape", encoded_valid_data.shape)

    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()
        encoded_valid_data = encoded_valid_data.cuda()

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

                src = torch.transpose(src, 0, 1)
                trg = torch.transpose(trg, 0, 1)
                onehot_trg = torch.transpose(onehot_trg, 0, 1)


                output, encoded = model(src, trg, 0)
                # output shape is code_len, batch, trg_vocab_size

                encoded = encoded.squeeze(0)

                #print("output.shape ", output.shape)
                #print("encoded.shape ", encoded.shape)

                for cpj in range(encoded.shape[0]):
                    encoded_valid_data[j+cpj] = encoded[cpj]

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
    return epoch_loss / batch_num, encoded_valid_data


def ed_evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0

    encoded_valid_data = torch.zeros(valid_data.shape[0], HID_DIM)
    #print("encoded_valid_data.shape", encoded_valid_data.shape)

    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()
        encoded_valid_data = encoded_valid_data.cuda()

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
                x= batch[:, :max_comment_len]
                xprime = batch[:, max_comment_len+max_code_len+1 : max_comment_len*2+max_code_len]
                yprime = batch[:, max_comment_len*2+max_code_len+1:]
                trg = batch[:, max_comment_len+1:]

                #src = torch.transpose(src, 0, 1)
                trg = torch.transpose(trg, 0, 1)
                onehot_trg = torch.transpose(onehot_trg, 0, 1)


                output, encoded = model(x, xprime, yprime, trg, 0)
                # output shape is code_len, batch, trg_vocab_size

                encoded = encoded.squeeze(0)

                #print("output.shape ", output.shape)
                #print("encoded.shape ", encoded.shape)

                for cpj in range(encoded.shape[0]):
                    encoded_valid_data[j+cpj] = encoded[cpj]

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
    return epoch_loss / batch_num, encoded_valid_data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_loss(filename, train_losses, valid_losses):
    plt.title("Loss vs Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    print("train_losses", train_losses)
    plt.plot(range(1,N_EPOCHS+1),train_losses,label="Train")
    plt.plot(range(1,N_EPOCHS+1),valid_losses,label="Validation")

    plt.legend()
    plt.savefig("results/loss_epochs")


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


def train_valid_model(filename, which_train, which_evaluate, model, train_data, valid_data, optimizer, criterion):
    best_valid_loss = float('inf')

    valid_losses = []
    train_losses = []
    times = []

    for epoch in range(1):
        start_time = time.time()

        train_loss, enc_train_vect = which_train(model, train_data, optimizer, criterion, CLIP)
        valid_loss, enc_valid_vect= which_evaluate(model, valid_data, criterion)
        train_losses += [train_loss]
        valid_losses += [valid_loss]

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        times += [end_time - start_time]
        print('| Epoch: {0:3d} | Time: {1:5d}m {2:5d}s| Train Loss: {3:.3f} | Train PPL: {4:7.3f} | Val. Loss: {5:.3f} | Val. PPL: {6:7.3f} |'.format(epoch+1, epoch_mins, epoch_secs, train_loss, math.exp(train_loss), valid_loss, math.exp(valid_loss)))

    with open("results/ret_attn_train_losses.pickle", 'wb') as f:
        pickle.dump(train_losses, f)
    with open("results/ret_attn_valid_losses.pickle", 'wb') as g:
        pickle.dump(valid_losses, g)
    with open("results/ret_attn_times.pickle", 'wb') as h:
        pickle.dump(times, h)
    plot_loss(filename = "results/ret_losses", train_losses=train_losses, valid_losses=valid_losses)
    return enc_train_vect, enc_valid_vect



def test_model(filename, which_evaluate, model, test_data, criterion):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    #test_loss, enc_test_vect, test_losses = evaluate(model, test_data, criterion)
    test_loss, enc_test_vect = which_evaluate(model, test_data, criterion)
    print('| Test Loss: {0:.3f} | Test PPL: {1:7.3f} |'.format(test_loss, math.exp(test_loss)))
    return enc_test_vect


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


    train_data, valid_data, test_data = split_data()


    ############################### RETRIEVER #################################

    ret_enc = AttnEncoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, AttnEncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    ret_dec = AttnDecoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, AttnDecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    ret_pad_idx = 0
    ret_model = AttnSeq2Seq(ret_enc, ret_dec, ret_pad_idx, device).to(device)


    print('The model has {0:9d} trainable parameters'.format(count_parameters(ret_model)))

    ret_optimizer = optim.Adam(ret_model.parameters())
    ret_criterion = nn.CrossEntropyLoss()

    if not os.path.isdir('models'):
        os.makedirs('models')

    enc_train_vect, enc_valid_vect = train_valid_model(filename="ret", which_train=ret_train, which_evaluate=ret_evaluate, model=ret_model, train_data=train_data, valid_data=valid_data, optimizer=ret_optimizer, criterion=ret_criterion)
    enc_test_vect = test_model(filename="ret", which_evaluate= ret_evaluate, model=ret_model, test_data=test_data, criterion=ret_criterion)


    ######################## NEAREST NEIGHBOUR #################################


    ann = create_annoy_index("AttnEncAttnDec", enc_train_vect)

    wordlist2comment_dict = pickle.load(open("wordlist2comment.pickle", "rb"))
    word2idcommentvocab_dict = pickle.load(open("word2idcommentvocab.pickle", "rb"))

    sim_train_data = torch.zeros_like(train_data)
    sim_valid_data = torch.zeros_like(valid_data)

    for training_sample_id in range(train_data.shape[0]):
        training_sample_comment = train_data[training_sample_id][:max_comment_len]
        training_sample_code = train_data[training_sample_id][max_comment_len+1:]

        annoy_vect = ann.get_item_vector(training_sample_id)

        sim_vect_id = ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == training_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_train_data[training_sample_id] = train_data[sim_vect_id]

    for valid_sample_id in range(valid_data.shape[0]):
        valid_sample_comment = valid_data[valid_sample_id][:max_comment_len]
        valid_sample_code = valid_data[valid_sample_id][max_comment_len+1:]

        annoy_vect = ann.get_item_vector(valid_sample_id)

        sim_vect_id = ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == valid_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_valid_data[valid_sample_id] = valid_data[sim_vect_id]

    new_valid_data = torch.cat((valid_data, sim_valid_data), dim=1)

    ############################### EDITOR #################################


    ed_enc = CondAttnEncoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, AttnEncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    ed_dec = AttnDecoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, AttnDecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    ed_pad_idx = 0
    ed_model = Editor(ed_enc, ed_dec, ed_pad_idx, device).to(device)


    for p in ed_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print('The model has {0:9d} trainable parameters'.format(count_parameters(ed_model)))

    ed_optimizer = optim.Adam(ed_model.parameters())
    ed_criterion = nn.CrossEntropyLoss()

    output_train_vect, output_valid_vect = train_valid_model(filename= "ed", which_train=ed_train, which_evaluate=ed_evaluate, model=ed_model, train_data=new_train_data, valid_data=new_valid_data, optimizer=ed_optimizer, criterion=ed_criterion)
    output_test_vect = test_model(filename="ed", which_evaluate=ed_evaluate, model=ed_model, test_data=test_data, criterion=ed_criterion)



if __name__== "__main__":
    main()
