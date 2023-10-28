import torch
from torch import nn


class RNNScratch(nn.Module):
    def __init__(
        self,
        args,
        input_size
    ):
        super(RNNScratch, self).__init__()
        self.input_size = input_size # num_of_features (aka, vocab_size)
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.nonlinearity = args.nonlinearity
        self.bias = args.bias
        
        self.i2h = nn.Linear(self.embedding_size, self.hidden_size, bias=self.bias) # Stores W_ih & b_i
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias) # Stores W_hh & b_h
        self.h2o = nn.Linear(self.hidden_size, self.input_size, bias=self.bias) # Stores W_ho & b_o
        
    def forward(self, x, h_prev):
        """
        Forward pass for a single RNN cell
        """
        weighted_input = self.i2h(x)   # (batch_sz, hidden_sz)
        hidden_state = self.h2h(h_prev)   # (batch_sz, hidden_sz)
        hidden_state = torch.tanh(weighted_input + hidden_state)   # (batch_sz, hidden_sz)
        output = self.h2o(hidden_state)   # (batch_sz, input_sz)
        return output, hidden_state
    
    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(
            batch_size,
            self.hidden_size
        ))
