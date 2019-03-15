from __future__ import print_function
import numpy as np

from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch import index_select
import argparse
import logging
import csv

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
from CondRNNEncoder import *
from RNNEncoder import *
from RNNDecoder import *
from Seq2Seq import *
from RNNEditor import *

from rnn_attn_reted import ret_train, ret_evaluate, count_parameters, epoch_time, plot_loss, create_annoy_index, train_valid_model, test_model, vis_tsne

import os

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
	datefmt="%Y-%m-%d %H:%M:%S")

SAVE_DIR = 'models'


def ed_train(opt, model, train_data, optimizer, criterion, clip):
	if opt.dataset_name == 'r252':
		ret_INPUT_DIM = r252_src_vocab_size
		ret_OUTPUT_DIM = r252_trg_vocab_size
		ed_input_dim  = r252_src_vocab_size
		ed_output_dim = r252_trg_vocab_size
		max_comment_len = r252_max_comment_len
		max_code_len = r252_max_code_len
		src_vocab_size = r252_src_vocab_size
		trg_vocab_size = r252_trg_vocab_size
	if opt.dataset_name == 'hstone':
		ret_INPUT_DIM = hstone_src_vocab_size
		ret_OUTPUT_DIM = hstone_trg_vocab_size
		ed_input_dim  = hstone_src_vocab_size
		ed_output_dim = hstone_trg_vocab_size
		max_comment_len = hstone_max_comment_len
		max_code_len = hstone_max_code_len
		src_vocab_size = hstone_src_vocab_size
		trg_vocab_size = hstone_trg_vocab_size
	model.train()
	epoch_loss = 0

	batch_num = 0

	if torch.cuda.is_available():
		model.cuda()
		train_data = train_data.cuda()

	for j in range(0, train_data.shape[0], batch_size): #TODO: replace batch_size with train_data.shape[0]
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

			x = torch.transpose(x, 0, 1)
			xprime = torch.transpose(xprime, 0, 1)
			yprime = torch.transpose(yprime, 0, 1)
			trg = torch.transpose(trg, 0, 1)

			output = model(x, xprime, yprime, trg)
			# output shape is code_len, batch, trg_vocab_size

			output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
			trg = torch.reshape(trg, (batch_size*max_code_len,))

			loss = criterion(output, trg)

			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
			optimizer.step()
			epoch_loss += loss.item()
			print("Ed batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
	return epoch_loss / batch_num, output


def ed_evaluate(model, valid_data, criterion):
	if opt.dataset_name == 'r252':
		ret_INPUT_DIM = r252_src_vocab_size
		ret_OUTPUT_DIM = r252_trg_vocab_size
		ed_input_dim  = r252_src_vocab_size
		ed_output_dim = r252_trg_vocab_size
		max_comment_len = r252_max_comment_len
		max_code_len = r252_max_code_len
		src_vocab_size = r252_src_vocab_size
		trg_vocab_size = r252_trg_vocab_size
	if opt.dataset_name == 'hstone':
		ret_INPUT_DIM = hstone_src_vocab_size
		ret_OUTPUT_DIM = hstone_trg_vocab_size
		ed_input_dim  = hstone_src_vocab_size
		ed_output_dim = hstone_trg_vocab_size
		max_comment_len = hstone_max_comment_len
		max_code_len = hstone_max_code_len
		src_vocab_size = hstone_src_vocab_size
		trg_vocab_size = hstone_trg_vocab_size
	model.eval()
	epoch_loss = 0

	if torch.cuda.is_available():
		model.cuda()
		valid_data = valid_data.cuda()

	ref_code = valid_data[:, max_comment_len:max_comment_len+max_code_len]
	candidate_code = torch.zeros_like(ref_code)

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
				xprime = batch[:, max_comment_len+max_code_len: max_comment_len*2+max_code_len]
				yprime = batch[:, max_comment_len*2+max_code_len:]

				trg = batch[:, max_comment_len:max_comment_len+max_code_len] # y

				x = torch.transpose(x, 0, 1)
				xprime = torch.transpose(xprime, 0, 1)
				yprime = torch.transpose(yprime, 0, 1)
				trg = torch.transpose(trg, 0, 1)

				output = model(x, xprime, yprime, trg)
				# output shape is code_len, batch, trg_vocab_size

				#trg = [trg sent len, batch size]
				#output = [trg sent len, batch size, output dim]

				for cpj in range(batch_size):
					candidate_code[j+cpj] = torch.argmax(output[cpj], dim=1)

				output = torch.reshape(output, (batch_size*max_code_len, trg_vocab_size))
				trg = torch.reshape(trg, (batch_size*max_code_len,))

				loss = criterion(output, trg)
				epoch_loss += loss.item()
				print("Ed batch: {0:3d} | Loss: {1:.3f}".format(batch_num, loss.item()))
	return epoch_loss / batch_num, candidate_code


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--resume_ret', action='store_true')
	parser.add_argument('--fourthofdata', action='store_true')
	parser.add_argument('--halfdata', action='store_true')
	parser.add_argument('--threefourthsofdata', action='store_true')
	parser.add_argument('--dataset_name', default='r252')

	opt = parser.parse_args()

	if opt.dataset_name == 'r252':
		ret_INPUT_DIM = r252_src_vocab_size
		ret_OUTPUT_DIM = r252_trg_vocab_size
		ed_input_dim  = r252_src_vocab_size
		ed_output_dim = r252_trg_vocab_size
		max_comment_len = r252_max_comment_len
		max_code_len = r252_max_code_len
		src_vocab_size = r252_src_vocab_size
		trg_vocab_size = r252_trg_vocab_size
	if opt.dataset_name == 'hstone':
		ret_INPUT_DIM = hstone_src_vocab_size
		ret_OUTPUT_DIM = hstone_trg_vocab_size
		ed_input_dim  = hstone_src_vocab_size
		ed_output_dim = hstone_trg_vocab_size
		max_comment_len = hstone_max_comment_len
		max_code_len = hstone_max_code_len
		src_vocab_size = hstone_src_vocab_size
		trg_vocab_size = hstone_trg_vocab_size

	############################### RETRIEVER #################################

	ret_enc = RNNEncoder(ret_INPUT_DIM, ret_ENC_EMB_DIM, ret_HID_DIM, ret_N_LAYERS, ret_ENC_DROPOUT)
	ret_dec = RNNDecoder(ret_OUTPUT_DIM, ret_DEC_EMB_DIM, ret_HID_DIM, ret_N_LAYERS, ret_DEC_DROPOUT)

	ret_model = Seq2Seq(ret_enc, ret_dec, cuda_device).to(cuda_device)

	print('The model has {0:9d} trainable parameters'.format(count_parameters(ret_model)))

	ret_optimizer = optim.Adam(ret_model.parameters())
	ret_criterion = nn.CrossEntropyLoss()

	if not os.path.isdir('models'):
		os.makedirs('models')

	if opt.resume_ret:
		with open("results/" + opt.dataset_name + "ret" + "_data.pickle", "rb") as k:
			data = pickle.load(k)
		train_data = data["train"]
		valid_data = data["valid"]
		test_data = data["test"]
		print("valid data", valid_data)
		MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "ret" + '_model.pt')
		ret_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
		with open("results/" + opt.dataset_name + "ret" + "_latent_space_vect.pickle", "rb") as j:
			latent_space_vects = pickle.load(j)
			enc_train_vect = latent_space_vects["train"]
			enc_valid_vect = latent_space_vects["valid"]
	else:
		train_data, valid_data, test_data = split_data(opt)
		data = {}
		data["train"] = train_data
		data["valid"] = valid_data
		data["test"] = test_data

		with open("results/" + opt.dataset_name + "ret" + "_data.pickle", "wb") as k:
			pickle.dump(data, k)
			enc_train_vect, enc_valid_vect = train_valid_model(opt=opt, filename="ret", which_train=ret_train, which_evaluate=ret_evaluate, model=ret_model, train_data=train_data, valid_data=valid_data, optimizer=ret_optimizer, criterion=ret_criterion)

	enc_test_vect = test_model(opt=opt, filename="ret", which_evaluate= ret_evaluate, model=ret_model, test_data=test_data, criterion=ret_criterion)

	######################## NEAREST NEIGHBOUR #################################


	train_ann = create_annoy_index("AttnEncAttnDecTrain", enc_train_vect)
	valid_ann = create_annoy_index("AttnEncAttnDecValid", enc_valid_vect)
	test_ann = create_annoy_index("AttnEncAttnDecTest", enc_test_vect)

	wordlist2comment_dict = pickle.load(open(opt.dataset_name + "wordlist2comment.pickle", "rb"))
	word2idcommentvocab_dict = pickle.load(open(opt.dataset_name + "word2idcommentvocab.pickle", "rb"))

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
		sim_vect_id = train_ann.get_nns_by_vector(annoy_vect, 1)

		if sim_vect_id == valid_sample_id:
			print("Same id for training vect and similar vect")
			exit(0)

		sim_valid_data[valid_sample_id] = train_data[sim_vect_id]

	new_valid_data = torch.cat((valid_data, sim_valid_data), dim=1)

	for test_sample_id in range(test_data.shape[0]):
		test_sample_comment = test_data[test_sample_id][:max_comment_len]
		test_sample_code = test_data[test_sample_id][max_comment_len+1:]

		annoy_vect = test_ann.get_item_vector(test_sample_id)
		sim_vect_id = train_ann.get_nns_by_vector(annoy_vect, 1)

		if sim_vect_id == test_sample_id:
			print("Same id for training vect and similar vect")
			exit(0)

		sim_test_data[test_sample_id] = train_data[sim_vect_id]

	new_test_data = torch.cat((test_data, sim_test_data), dim=1)

	############################### TSNE #################################

	#tsne_test_sample = enc_test_vect[0]
	num_tsne_train_data = 100
	which_tsne_test_sample = random.randint(0,enc_test_vect.shape[0])

	annoy_tsne_test_vect = test_ann.get_item_vector(which_tsne_test_sample)
	tsne_data = enc_train_vect[:num_tsne_train_data]
	tsne_data_add = torch.zeros(11, enc_test_vect.shape[1], device=cuda_device)
	tsne_data_add[0] = enc_test_vect[which_tsne_test_sample]

	nr = 1
	for id in train_ann.get_nns_by_vector(annoy_tsne_test_vect, 10):
		tsne_data_add[nr] = enc_train_vect[id]
		nr +=1

	tsne_data = torch.cat((tsne_data, tsne_data_add), dim=0)

	colour_labels = []
	for i in range(num_tsne_train_data):
		colour_labels += ["#0099cc"] #train
	colour_labels += ["#e60b42"] #test
	for i in range(10):
		colour_labels += ["#f09a00"] #nearest neighbours

	vis_tsne(data=tsne_data, labels=colour_labels, name="10nearest")

	############################### EDITOR #################################

	ed_enc = CondRNNEncoder(src_vocab_size, trg_vocab_size, ret_ENC_EMB_DIM, ret_HID_DIM, ret_N_LAYERS, ret_ENC_DROPOUT)
	ed_dec = RNNDecoder(ret_OUTPUT_DIM, ret_DEC_EMB_DIM, ret_HID_DIM, ret_N_LAYERS, ret_DEC_DROPOUT)

	ed_model = RNNEditor(ed_enc, ed_dec, cuda_device).to(cuda_device)


	for p in ed_model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)

	print('The model has {0:9d} trainable parameters'.format(count_parameters(ed_model)))

	ed_optimizer = optim.Adam(ed_model.parameters())
	ed_criterion = nn.CrossEntropyLoss()

	output_train_vect, output_valid_vect_candidates = train_valid_model(opt=opt, filename= "ed", which_train=ed_train, which_evaluate=ed_evaluate, model=ed_model, train_data=new_train_data, valid_data=new_valid_data, optimizer=ed_optimizer, criterion=ed_criterion)
	#print("Test model")
	output_test_vect_candidates = test_model(opt=opt, filename="ed", which_evaluate=ed_evaluate, model=ed_model, test_data=new_test_data, criterion=ed_criterion)
	output_test_vect_reference = test_data[:, max_comment_len:]

	token_dict = pickle.load(open(opt.dataset_name +"codevocab.pickle", "rb"))

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
	pickle.dump(bleu_eval, open("results/" + opt.dataset_name + "bleu_evaluation_results.pickle", "wb"))

if __name__== "__main__":
	main()
