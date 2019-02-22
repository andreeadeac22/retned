import os
import torch

max_code_len = 150

max_comment_len = 296

#src_vocab_size = 14976 -- unfiltered
src_vocab_size = 8278 + 2  # after removing words with freq=1

#trg_vocab_size = 21550
trg_vocab_size = 11394 + 2 #after removing tokens with freq=1

batch_size = 64

INPUT_DIM = src_vocab_size
OUTPUT_DIM = trg_vocab_size
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 64
N_LAYERS = 1
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3


N_EPOCHS = 20
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_model.pt')

cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda_device is ", cuda_device)
