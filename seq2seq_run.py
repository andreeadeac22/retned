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
from constants import *
from RNNEncoder import *
from RNNDecoder import *
from Seq2Seq import *


SAVE_DIR = 'models'


def train(model, train_data, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    batch_num = 0
    encoded_train_data = torch.zeros(train_data.shape[0], ret_HID_DIM)

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
            trg = batch[:, max_comment_len:]

            src = torch.transpose(src, 0, 1)
            trg = torch.transpose(trg, 0, 1)

            optimizer.zero_grad()
            output, encoded = model(src, trg)
            # output shape is code_len, batch, trg_vocab_size

            #encoded = torch.reshape(encoded, (ret_N_LAYERS * batch_size, ret_HID_DIM))

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



def evaluate(model, valid_data, criterion):
    model.eval()
    epoch_loss = 0

    encoded_valid_data = torch.zeros(valid_data.shape[0], ret_HID_DIM, device=cuda_device)
    #print("encoded_valid_data.shape", encoded_valid_data.shape)

    if torch.cuda.is_available():
        model.cuda()
        valid_data = valid_data.cuda()
        encoded_valid_data = encoded_valid_data.cuda()

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
                trg = batch[:, max_comment_len:]


                src = torch.transpose(src, 0, 1)
                trg = torch.transpose(trg, 0, 1)

                output, encoded = model(src, trg, 0)
                # output shape is code_len, batch, trg_vocab_size

                #encoded = torch.reshape(encoded, (ret_N_LAYERS * batch_size, ret_HID_DIM))


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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_loss(train_losses, valid_losses):
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


def train_valid_model(model, train_data, valid_data, optimizer, criterion):
    best_valid_loss = float('inf')

    valid_losses = []
    train_losses = []
    times = []

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, enc_train_vect = train(model, train_data, optimizer, criterion, CLIP)
        valid_loss, enc_valid_vect= evaluate(model, valid_data, criterion)
        train_losses += [train_loss]
        valid_losses += [valid_loss]

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'ret_only_model.pt')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        times += [end_time - start_time]
        print('| Epoch: {0:3d} | Time: {1:5d}m {2:5d}s| Train Loss: {3:.3f} | Train PPL: {4:7.3f} | Val. Loss: {5:.3f} | Val. PPL: {6:7.3f} |'.format(epoch+1, epoch_mins, epoch_secs, train_loss, math.exp(train_loss), valid_loss, math.exp(valid_loss)))

    with open("results/rnn_train_losses.pickle", 'wb') as f:
        pickle.dump(train_losses, f)
    with open("results/rnn_valid_losses.pickle", 'wb') as g:
        pickle.dump(valid_losses, g)
    with open("results/rnn_times.pickle", 'wb') as h:
        pickle.dump(times, h)
    plot_loss(train_losses, valid_losses)

    return enc_train_vect, enc_valid_vect



def test_model(model, test_data, criterion):
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'ret_only_model.pt')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    #test_loss, enc_test_vect, test_losses = evaluate(model, test_data, criterion)
    test_loss, enc_test_vect = evaluate(model, test_data, criterion)
    print('| Test Loss: {0:.3f} | Test PPL: {1:7.3f} |'.format(test_loss, math.exp(test_loss)))
    return enc_test_vect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_ret', action='store_true')
    parser.add_argument('--fourthofdata', action='store_true')
    parser.add_argument('--halfdata', action='store_true')
    parser.add_argument('--threefourthsofdata', action='store_true')
    opt = parser.parse_args()

    train_data, valid_data, test_data = split_data(opt)

    ret_enc = RNNEncoder(ret_INPUT_DIM, ret_ENC_EMB_DIM, ret_HID_DIM, ret_N_LAYERS, ret_ENC_DROPOUT)
    ret_dec = RNNDecoder(ret_OUTPUT_DIM, ret_DEC_EMB_DIM, ret_HID_DIM, ret_N_LAYERS, ret_DEC_DROPOUT)

    model = Seq2Seq(ret_enc, ret_dec, cuda_device).to(cuda_device)


    print('The model has {0:9d} trainable parameters'.format(count_parameters(model)))

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if not os.path.isdir('models'):
        os.makedirs('models')

    enc_train_vect, enc_valid_vect = train_valid_model(model=model, train_data=train_data, valid_data=valid_data, optimizer=optimizer, criterion=criterion)
    enc_test_vect = test_model(model=model, test_data=test_data, criterion=criterion)


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

    for valid_sample_id in range(valid_data.shape[0]):
        valid_sample_comment = valid_data[valid_sample_id][:max_comment_len]
        valid_sample_code = valid_data[valid_sample_id][max_comment_len+1:]

        annoy_vect = valid_ann.get_item_vector(valid_sample_id)
        sim_vect_id = train_ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == valid_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_valid_data[valid_sample_id] = train_data[sim_vect_id]

    for test_sample_id in range(test_data.shape[0]):
        test_sample_comment = test_data[test_sample_id][:max_comment_len]
        test_sample_code = test_data[test_sample_id][max_comment_len+1:]

        annoy_vect = test_ann.get_item_vector(test_sample_id)
        sim_vect_id = train_ann.get_nns_by_vector(annoy_vect, 1)

        if sim_vect_id == test_sample_id:
            print("Same id for training vect and similar vect")
            exit(0)

        sim_test_data[test_sample_id] = train_data[sim_vect_id]

    output_test_vect_reference = test_data[:, max_comment_len:]
    output_test_vect_candidates = sim_test_data[:, max_comment_len:]

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

    print("Average BLEU score is ", sum(all_bleu_scores)/len(all_bleu_scores))
    pickle.dump(bleu_eval, open("results/bleu_evaluation_results.pickle", "wb"))


if __name__== "__main__":
    main()
