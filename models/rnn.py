import torch
from torch import nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True
    ):
        super(RNN, self).__init__()
        # self.args = args
        self.input_size = input_size # num_of_features (aka, vocab_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(self.input_size, self.hidden_size, bias=bias) # Stores W_ih & b_i
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=bias) # Stores W_hh & b_h
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=bias) # Stores W_ho & b_o
        
    def forward(self, x, hidden_state) -> tuple(torch.Tensor, torch.Tensor):
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = torch.tanh(x + hidden_state)
        output = self.h2o(hidden_state)
        return output, hidden_state
    
    def initialize_h_0(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, requires_grad=False)
    
    
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)