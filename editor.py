import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

class Editor(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_masks(self, src, trg):

        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))

        trg_mask = trg_pad_mask & trg_sub_mask

        return src_mask, trg_mask

    def forward(self, src_x, src_xprime, src_yprime, trg):

        #src = [batch size, src sent len]
        #trg = [batch size, trg sent len]

        src = torch.cat((src_x, src_xprime, src_yprime), dim=1)

        src_mask, trg_mask = self.make_masks(src, trg)

        enc_src = self.encoder(src_x, src_xprime, src_yprime, src_mask)

        #enc_src = [batch size, src sent len, hid dim]

        out = self.decoder(trg, enc_src, trg_mask, src_mask)

        #out = [batch size, trg sent len, output dim]

        return out
