import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch import index_select

class Editor(nn.Module):
    def __init__(self):
        super(Editor, self).__init__()
        # encoder-decoder
         for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        #TODO: adapt
        """

        :param m: Layer type
        :return: void. Performs parameter initialisation
        """
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
            m.bias.data.fill_(0.0)

    def forward(self, input):
        return input
