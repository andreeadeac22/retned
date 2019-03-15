import os
import torch

r252_max_code_len = 150
r252_max_comment_len = 296

hstone_max_code_len = 430
hstone_max_comment_len = 50

#src_vocab_size = 14976 -- unfiltered
r252_src_vocab_size = 8278 + 2  # after removing words with freq=1
#trg_vocab_size = 21550
r252_trg_vocab_size = 11394 + 2 #after removing tokens with freq=1


hstone_src_vocab_size = 1704 + 2
hstone_trg_vocab_size = 727 + 2


ret_ENC_EMB_DIM = 128
ret_DEC_EMB_DIM = 128
ret_HID_DIM = 64
ret_N_LAYERS = 2
ret_ENC_DROPOUT = 0.5
ret_DEC_DROPOUT = 0.5


ed_n_layers = 3
ed_n_heads = 2
ed_dropout = 0.1
ed_hid_dim = 64 # was 512
ed_pf_dim = 32 # was 2048?

patience = 4

batch_size = 16

N_EPOCHS = 20
CLIP = 1

result_csv_file = "results/valid_loss.csv"

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda_device is ", cuda_device)
