
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
from RNNEncoder import *
from RNNDecoder import *
from Seq2Seq import *

import os

SAVE_DIR = 'models'


def ret_train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    batch_num = 0
    encoded_train_data = torch.zeros(train_data.shape[0], HID_DIM)

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()
        encoded_train_data = encoded_train_data.cuda()

    for j in range(0, batch_size, batch_size): #TODO: replace batch_size with train_data.shape[0]
        if j+ batch_size < train_data.shape[0]:
            batch_num +=1
            interval = [x for x in range(j, min(train_data.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if torch.cuda.is_available():
                interval = interval.cuda()
            batch = Variable(index_select(train_data, 0, interval))
            src = batch[:, :max_comment_len]
            trg = batch[:, max_comment_len:]

            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)

            optimizer.zero_grad()
            output, encoded = model(src, trg)
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


def ed_train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    batch_num = 0

    if torch.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()

    for j in range(0, batch_size, batch_size): #TODO: replace batch_size with train_data.shape[0]
        if j+ batch_size < train_data.shape[0]:
            batch_num +=1
            interval = [x for x in range(j, min(train_data.shape[0], j + batch_size))]
            interval = torch.LongTensor(interval)
            if torch.cuda.is_available():
                interval = interval.cuda()
            batch = Variable(index_select(train_data, 0, interval))

            x= batch[:, :max_comment_len]
            xprime = batch[:, max_comment_len+max_code_len: max_comment_len*2+max_code_len]
            yprime = batch[:, max_comment_len*2+max_code_len:]

            trg = batch[:, max_comment_len:max_comment_len+max_code_len] # y

            optimizer.zero_grad()
            output = model(x, xprime, yprime, trg)
            # output shape is code_len, batch, trg_vocab_size

            output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
            trg = torch.reshape(trg, (batch_size*max_code_len,))

            loss = criterion(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    return epoch_loss / batch_num, output


def ret_evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0

    encoded_valid_data = torch.zeros(valid_data.shape[0], HID_DIM, device=device)
    #print("encoded_valid_data.shape", encoded_valid_data.shape)

    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()
        encoded_valid_data = encoded_valid_data.cuda()

    with torch.no_grad():
        batch_num = 0
        for j in range(0, batch_size, batch_size):
            if j+ batch_size < valid_data.shape[0]:
                batch_num +=1
                interval = [x for x in range(j, min(valid_data.shape[0], j + batch_size))]
                interval = torch.LongTensor(interval)
                if torch.cuda.is_available():
                    interval = interval.cuda()
                batch = Variable(index_select(valid_data, 0, interval))
                src = batch[:, :max_comment_len]
                trg = batch[:, max_comment_len:]


                src = torch.transpose(src, 0, 1)
                trg = torch.transpose(trg, 0, 1)

                output, encoded = model(src, trg, 0)
                # output shape is code_len, batch, trg_vocab_size

                encoded = encoded.squeeze(0)


                for cpj in range(encoded.shape[0]):
                    encoded_valid_data[j+cpj] = encoded[cpj]

                #trg = [trg sent len, batch size]
                #output = [trg sent len, batch size, output dim]

                output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
                trg = torch.reshape(trg, (batch_size*max_code_len,))

                loss = criterion(output, trg)
                epoch_loss += loss.item()
                print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    return epoch_loss / batch_num, encoded_valid_data


def ed_evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0

    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()

    ref_code = valid_data[:, max_comment_len:max_comment_len+max_code_len]
    candidate_code = torch.zeros_like(ref_code)

    with torch.no_grad():
        batch_num = 0
        for j in range(0, batch_size, batch_size):
            if j+ batch_size < valid_data.shape[0]:
                batch_num +=1
                interval = [x for x in range(j, min(valid_data.shape[0], j + batch_size))]
                interval = torch.LongTensor(interval)
                if torch.cuda.is_available():
                    interval = interval.cuda()
                batch = Variable(index_select(valid_data, 0, interval))

                x= batch[:, :max_comment_len]
                xprime = batch[:, max_comment_len+max_code_len: max_comment_len*2+max_code_len]
                yprime = batch[:, max_comment_len*2+max_code_len:]

                trg = batch[:, max_comment_len:max_comment_len+max_code_len] # y


                output = model(x, xprime, yprime, trg)
                # output shape is code_len, batch, trg_vocab_size

                #trg = [trg sent len, batch size]
                #output = [trg sent len, batch size, output dim]

                for cpj in range(batch_size):
                    candidate_code[j+cpj] = torch.argmax(output[j+cpj], dim=1)

                output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
                trg = torch.reshape(trg, (batch_size*max_code_len,))

                loss = criterion(output, trg)
                epoch_loss += loss.item()
                print("Batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
    return epoch_loss / batch_num, candidate_code


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
    #print("train_losses", train_losses)
    plt.plot(range(1,N_EPOCHS+1),train_losses,label="Train")
    plt.plot(range(1,N_EPOCHS+1),valid_losses,label="Validation")

    plt.legend()
    plt.savefig(filename)


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

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, enc_train_vect = which_train(model, train_data, optimizer, criterion, CLIP)
        valid_loss, enc_valid_vect_candidates = which_evaluate(model, valid_data, criterion)
        train_losses += [train_loss]
        valid_losses += [valid_loss]

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            MODEL_SAVE_PATH = os.path.join(SAVE_DIR, filename + '_model.pt')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        times += [end_time - start_time]
        print('| Epoch: {0:3d} | Time: {1:5d}m {2:5d}s| Train Loss: {3:.3f} | Train PPL: {4:7.3f} | Val. Loss: {5:.3f} | Val. PPL: {6:7.3f} |'.format(epoch+1, epoch_mins, epoch_secs, train_loss, math.exp(train_loss), valid_loss, math.exp(valid_loss)))

    with open("results/" + filename + "_train_losses.pickle", 'wb') as f:
        pickle.dump(train_losses, f)
    with open("results/" + filename + "_valid_losses.pickle", 'wb') as g:
        pickle.dump(valid_losses, g)
    with open("results/" + filename + "_attn_times.pickle", 'wb') as h:
        pickle.dump(times, h)
    plot_loss(filename = "results/" + filename + "_losses", train_losses=train_losses, valid_losses=valid_losses)
    return enc_train_vect, enc_valid_vect_candidates



def test_model(filename, which_evaluate, model, test_data, criterion):
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, filename + '_model.pt')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    #test_loss, enc_test_vect, test_losses = evaluate(model, test_data, criterion)
    test_loss, enc_test_vect_candidates = which_evaluate(model, test_data, criterion)
    print('| Test Loss: {0:.3f} | Test PPL: {1:7.3f} |'.format(test_loss, math.exp(test_loss)))
    return enc_test_vect_candidates


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

    ret_enc = RNNEncoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    ret_dec = RNNDecoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    ret_model = Seq2Seq(ret_enc, ret_dec, cuda_device).to(cuda_device)


    print('The model has {0:9d} trainable parameters'.format(count_parameters(ret_model)))

    ret_optimizer = optim.Adam(ret_model.parameters())
    ret_criterion = nn.CrossEntropyLoss()

    if not os.path.isdir('models'):
        os.makedirs('models')

    enc_train_vect, enc_valid_vect = train_valid_model(filename="ret", which_train=ret_train, which_evaluate=ret_evaluate, model=ret_model, train_data=train_data, valid_data=valid_data, optimizer=ret_optimizer, criterion=ret_criterion)
    enc_test_vect = test_model(filename="ret", which_evaluate= ret_evaluate, model=ret_model, test_data=test_data, criterion=ret_criterion)


    ######################## NEAREST NEIGHBOUR #################################


    train_ann = create_annoy_index("AttnEncAttnDecTrain", enc_train_vect)
    valid_ann = create_annoy_index("AttnEncAttnDecValid", enc_valid_vect)
    test_ann = create_annoy_index("AttnEncAttnDecTest", enc_test_vect)


    wordlist2comment_dict = pickle.load(open("wordlist2comment.pickle", "rb"))
    word2idcommentvocab_dict = pickle.load(open("word2idcommentvocab.pickle", "rb"))

    sim_train_data = torch.zeros_like(train_data)
    sim_valid_data = torch.zeros_like(valid_data)
    sim_test_data = torch.zeros_like(test_data)

    for training_sample_id in range(train_data.shape[0]):
        training_sample_comment = train_data[training_sample_id][:max_comment_len]
        training_sample_code = train_data[training_sample_id][max_comment_len+1:]

        annoy_vect = train_ann.get_item_vector(training_sample_id)

        sim_vect_id = train_ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == training_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_train_data[training_sample_id] = train_data[sim_vect_id]

    new_train_data = torch.cat((train_data, sim_train_data), dim=1)
    #print("new_train_data ", new_train_data.shape)

    for valid_sample_id in range(valid_data.shape[0]):
        valid_sample_comment = valid_data[valid_sample_id][:max_comment_len]
        valid_sample_code = valid_data[valid_sample_id][max_comment_len+1:]

        annoy_vect = valid_ann.get_item_vector(valid_sample_id)

        sim_vect_id = valid_ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == valid_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_valid_data[valid_sample_id] = valid_data[sim_vect_id]

    new_valid_data = torch.cat((valid_data, sim_valid_data), dim=1)

    for test_sample_id in range(test_data.shape[0]):
        test_sample_comment = test_data[test_sample_id][:max_comment_len]
        test_sample_code = test_data[test_sample_id][max_comment_len+1:]

        annoy_vect = test_ann.get_item_vector(test_sample_id)

        sim_vect_id = test_ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == test_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_test_data[test_sample_id] = test_data[sim_vect_id]

    new_test_data = torch.cat((test_data, sim_test_data), dim=1)

    ############################### EDITOR #################################


    ed_enc = CondAttnEncoder(src_vocab_size, trg_vocab_size, hid_dim, n_layers, n_heads, pf_dim, AttnEncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    ed_dec = AttnDecoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, AttnDecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    ed_pad_idx = 0
    ed_model = Editor(ed_enc, ed_dec, ed_pad_idx, device).to(device)


    for p in ed_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    print('The model has {0:9d} trainable parameters'.format(count_parameters(ed_model)))

    ed_optimizer = optim.Adam(ed_model.parameters())
    ed_criterion = nn.CrossEntropyLoss()

    output_train_vect, output_valid_vect_candidates = train_valid_model(filename= "ed", which_train=ed_train, which_evaluate=ed_evaluate, model=ed_model, train_data=new_train_data, valid_data=new_valid_data, optimizer=ed_optimizer, criterion=ed_criterion)
    #print("Test model")
    output_test_vect_candidates = test_model(filename="ed", which_evaluate=ed_evaluate, model=ed_model, test_data=new_test_data, criterion=ed_criterion)
    output_test_vect_reference = test_data[:, max_comment_len:]

    token_dict = pickle.load(open("codevocab.pickle", "rb"))


    all_refs = []
    all_cands = []
    all_bleu_scores = []
    for j in range(test_data.shape[0]):
        ref = []
        cand = []
        for i in range(max_code_len):
            ref_el = output_test_vect_reference[j][i].item()
            cand_el = output_test_vect_candidates[j][i].item()
            if ref_el > 0:
                if ref_el in token_dict:
                    ref += [token_dict[ref_el]]
                if cand_el in token_dict:
                    cand += [token_dict[cand_el]]
        bleu = sentence_bleu([ref], cand)
        all_bleu_scores += [bleu]
        all_refs += [ref]
        all_cands += [cand]

    bleu_eval = {}
    bleu_eval["scores"] = all_bleu_scores
    bleu_eval["references"] = all_refs
    bleu_eval["candidates"] = all_cands
    pickle.dump(bleu_eval, open("results/bleu_evaluation_results.pickle", "wb"))

if __name__== "__main__":
    main()
