import numpy as np
from model_utils import *


class LSTM:
    def __init__(self, vocab_size, hidden_size, lr):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.W_xf = np.zeros((self.hidden_size, self.vocab_size))
        self.W_xi = np.zeros((self.hidden_size, self.vocab_size))
        self.W_xg = np.zeros((self.hidden_size, self.vocab_size))
        self.W_xo = np.zeros((self.hidden_size, self.vocab_size))
        self.W_hf = np.zeros((self.hidden_size, self.hidden_size))
        self.W_hi = np.zeros((self.hidden_size, self.hidden_size))
        self.W_hg = np.zeros((self.hidden_size, self.hidden_size))
        self.W_ho = np.zeros((self.hidden_size, self.hidden_size))
        self.W_v = np.zeros((self.vocab_size, self.hidden_size))
        
        self.b_f = np.zeros((self.hidden_size, 1))
        self.b_i = np.zeros((self.hidden_size, 1))
        self.b_g = np.zeros((self.hidden_size, 1))
        self.b_o = np.zeros((self.hidden_size, 1))
        self.b_v = np.zeros((self.vocab_size, 1))
    
    def forward(self, inputs, hidden_state, cell_state):
        f_dict, i_dict, g_dict, o_dict = {}, {}, {}, {}
        hidden_states, cell_states, outputs = {}, {}, {}
        hidden_states[-1] = hidden_state
        cell_states[-1] = cell_state
        
        for t in range(len(inputs)):
            f_dict[t] = sigmoid(
                np.dot(self.W_xf, inputs[t]) +\
                np.dot(self.W_hf, hidden_states[t-1]) +\
                self.b_f
            )
            i_dict[t] = sigmoid(
                np.dot(self.W_xi, inputs[t]) +\
                np.dot(self.W_hi, hidden_states[t-1]) +\
                self.b_i
            )
            g_dict[t] = tanh(
                np.dot(self.W_xg, inputs[t]) +\
                np.dot(self.W_hg, hidden_states[t-1]) +\
                self.b_g
            )
            cell_states[t] = np.multiply(cell_states[t-1], f_dict[t]) +\
                np.multiply(i_dict[t], g_dict[t])
            o_dict[t] = sigmoid(
                np.dot(self.W_xo, inputs[t]) +\
                np.dot(self.W_ho, hidden_states[t-1]) +\
                self.b_o
            )
            hidden_states[t] = np.multiply(o_dict[t], tanh(cell_states[t]))
            output = softmax(
                np.dot(self.W_v, hidden_states[t]) + self.b_v
            )
            outputs[t] = output
            
        return f_dict, i_dict, g_dict, o_dict, cell_states, hidden_states, outputs
        
    def backward(self):
        pass
    
    def optimize(self):
        pass