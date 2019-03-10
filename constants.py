import os
import torch

max_code_len = 150

max_comment_len = 296

#src_vocab_size = 14976 -- unfiltered
src_vocab_size = 8278 + 2  # after removing words with freq=1

#trg_vocab_size = 21550
trg_vocab_size = 11394 + 2 #after removing tokens with freq=1


ret_INPUT_DIM = src_vocab_size
ret_OUTPUT_DIM = trg_vocab_size
ret_ENC_EMB_DIM = 128
ret_DEC_EMB_DIM = 128
ret_HID_DIM = 64
ret_N_LAYERS = 2
ret_ENC_DROPOUT = 0.5
ret_DEC_DROPOUT = 0.5


ed_input_dim  = src_vocab_size
ed_output_dim = trg_vocab_size
ed_n_layers = 3
ed_n_heads = 2
ed_dropout = 0.3
ed_hid_dim = 64 # was 512
ed_pf_dim = 32 # was 2048?

patience = 4

batch_size = 128

N_EPOCHS = 30
CLIP = 1

result_csv_file = "results/valid_loss.csv"

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda_device is ", cuda_device)
