import torch
from torch import nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        embedding_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True
    ):
        super(RNN, self).__init__()
        # self.args = args
        self.input_size = input_size # num_of_features (aka, vocab_size)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        
        self.embed_layer = nn.Embedding(self.input_size, self.embedding_size, 0)
        self.i2h = nn.Linear(self.embedding_size, self.hidden_size, bias=bias) # Stores W_ih & b_i
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=bias) # Stores W_hh & b_h
        self.h2o = nn.Linear(self.hidden_size, self.input_size, bias=bias) # Stores W_ho & b_o
        
    def forward(self, x, h_prev):
        embedding = self.embed_layer(x) # (batch_sz, seq_len, embed_sz)
        print(embedding.shape)
        weighted_input = self.i2h(embedding) # (batch_sz, input_sz, hidden_sz)
        hidden_state = self.h2h(h_prev) # (hidden_sz, input_sz)
        print(hidden_state.shape, h_prev.shape)
        hidden_state = torch.tanh(weighted_input + hidden_state) # (batch_sz, hidden
        output = self.h2o(hidden_state) # (batch_sz, hidden_sz,
        return output, hidden_state
    
    def initialize_h_0(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size))


if __name__ == "__main__":
    seq_len = 4
    batch_size = 2
    vocab_size = 5
    hidden_size = 3
    embed_size = 6
    h_prev = torch.zeros(hidden_size)
    a = torch.randint(0, 4, (batch_size, seq_len)) # (2, 4)
    print(a)
    print(a[:,0])