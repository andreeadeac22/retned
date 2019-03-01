import os
import torch

#src_vocab_size = 14976 -- unfiltered
src_vocab_size = 8278 + 2  # after removing words with freq=1

#trg_vocab_size = 21550
trg_vocab_size = 11394 + 2 #after removing tokens with freq=1hid_dim = 512

input_dim  = src_vocab_size
n_layers = 3
n_heads = 2
pf_dim = 32
dropout = 0.1

output_dim = trg_vocab_size
hid_dim = 64 # was 512
n_layers = 3
n_heads = 2
pf_dim = 32 # was 2048?
dropout = 0.1

BATCH_SIZE = 16 

N_EPOCHS = 20

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'reted_model.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda_device is ", device)
