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
        
        self.x2r = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.x2u = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.x2n = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias)
        self.h2r = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.h2u = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.h2n = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        
    def forward(self, x, h_prev):
        """
        Forward pass for a single RNN cell
        """
        r2h = torch.sigmoid(self.x2r(x) + self.h2r(h_prev))
        u2h = torch.sigmoid(self.x2u(x) + self.h2u(h_prev))
        n2h = torch.tanh(self.x2n(x) + torch.multiply(r2h, self.h2n(h_prev)))
        hidden_state = torch.multiply((1-u2h), n2h) + torch.multiply(u2h, h_prev)
        return hidden_state
    
    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(
            batch_size,
            self.hidden_size
        ))
