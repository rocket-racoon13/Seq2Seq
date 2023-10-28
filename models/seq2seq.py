import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(
        self,
        args,
        input_size
    ):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.bias = args.bias
        self.bidirectional = args.bidirectional
        self.directions = 2 if args.bidirectional else 1
        
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.fc = nn.Linear(self.hidden_size, self.input_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_output, (hidden_state, cell_state) = self.rnn(embedded)
        y_pred = self.fc(rnn_output)
        return y_pred
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden_state (or cell_state) for the beginning sequence.
        Implemented in nn.LSTM, so the current code is not used.
        """
        return nn.init.kaiming_uniform_(torch.empty(
            self.num_layers*self.directions,
            batch_size,
            self.hidden_size
        ))