import numpy as np


class RNN:
    def __init__(self):
        self.hidden_size = 100
        self.seq_length = 25
        self.learning_rate = 1e-1
        self.vocab_size = None
        
        self.W_xh = np.random.randn(self.hidden_size, self.vocab_size)*0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size)*0.01
        self.W_hy = np.random.randn(self.vocab_size, self.hidden_size)*0.01
        self.b_h = np.zeros((self.hidden_size, 1))
        self.b_y = np.zeros((self.hidden_size, 1))
        
        self.xs = {}
        self.hs = {}
        self.ys = {}
        self.ps = {}
    
    def forward(self, inputs, targets, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.W_xh, xs[t]) + np.dot(self.W_hh, hs[t-1]) + self.b_h)
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])
        
        parameters = (xs, hs, ys, ps)
        return parameters, loss
    
    def backward(self, inputs, targets, parameters):
        xs, hs, ys, ps = parameters
        
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)        
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])