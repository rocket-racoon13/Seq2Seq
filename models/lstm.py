import torch
from torch import nn
import torch.nn.functional as F


class LSTMScratch(nn.Module):
    def __init__(
        self,
        args,
        input_size
    ):
        super(LSTMScratch, self).__init__()
        self.input_size = input_size # num_of_features (aka, vocab_size)
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.nonlinearity = args.nonlinearity
        self.bias = args.bias
        
        self.x2i = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.x2f = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.x2g = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.x2o = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.h2i = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.h2f = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.h2g = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.h2o = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single RNN cell
        """
        i2h = torch.sigmoid(self.x2i(x) + self.h2i(h_prev))
        f2h = torch.sigmoid(self.x2f(x) + self.h2f(h_prev))
        g2h = torch.tanh(self.x2g(x) + self.h2g(h_prev))
        o2h = torch.sigmoid(self.x2o(x) + self.h2o(h_prev))
        cell_state = torch.multiply(f2h, c_prev) +\
                     torch.multiply(i2h, g2h)
        hidden_state = torch.multiply(o2h, torch.tanh(cell_state))
        return hidden_state, cell_state
    
    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(
            batch_size,
            self.hidden_size
        ))
