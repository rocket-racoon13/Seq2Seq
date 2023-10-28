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
        
        self.encoder = nn.Embedding(self.input_size, self.embedding_size, 0)
        self.i2h = nn.Linear(self.embedding_size, self.hidden_size, bias=bias) # Stores W_ih & b_i
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, bias=bias) # Stores W_hh & b_h
        self.h2o = nn.Linear(self.hidden_size, self.input_size, bias=bias) # Stores W_ho & b_o
        
    def forward(self, x, h_prev):
        print(x.shape)
        embedding = self.encoder(x)   # (batch_sz, input_sz, embedding_sz)
        print(embedding)
        print(embedding.shape)
        weighted_input = self.i2h(embedding)   # (batch_sz, hidden_sz)
        hidden_state = self.h2h(h_prev)   # (batch_sz, hidden_sz)
        print(weighted_input.shape, hidden_state.shape)
        hidden_state = torch.tanh(weighted_input + hidden_state)   # (batch_sz, hidden_sz)
        output = self.h2o(hidden_state)   # (batch_sz, input_sz)
        return output, hidden_state
    
    def initialize_h_0(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size))


if __name__ == "__main__":
    batch_size = 2
    vocab_size = 5
    hidden_size = 3
    embed_size = 6
    h_prev = torch.zeros(hidden_size)
    a = [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0]]
    a = torch.LongTensor(a)
    rnn = RNN(vocab_size, hidden_size, embed_size)
    output, hidden_state = rnn(a, rnn.initialize_h_0(batch_size))
    print(output.shape, hidden_state.shape)