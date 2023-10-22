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
        self.W_y = np.zeros((self.vocab_size, self.hidden_size))
        
        self.b_f = np.zeros((self.hidden_size, 1))
        self.b_i = np.zeros((self.hidden_size, 1))
        self.b_g = np.zeros((self.hidden_size, 1))
        self.b_o = np.zeros((self.hidden_size, 1))
        self.b_y = np.zeros((self.vocab_size, 1))
    
    def forward(self, inputs, hidden_state, cell_state):
        state_dict = {}
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
                np.dot(self.W_y, hidden_states[t]) + self.b_y
            )
            outputs[t] = output
        
        # pack states
        state_dict["f_dict"] = f_dict
        state_dict["i_dict"] = i_dict
        state_dict["g_dict"] = g_dict
        state_dict["o_dict"] = o_dict
        state_dict["cell_states"] = cell_states
        state_dict["hidden_states"] = hidden_states
        
        return state_dict, outputs
        
    def backward(self, inputs, outputs, targets, state_dict):
        # unpack states
        f_dict = state_dict["f_dict"]
        i_dict = state_dict["i_dict"]
        g_dict = state_dict["g_dict"]
        o_dict = state_dict["o_dict"]
        cell_states= state_dict["cell_states"]
        hidden_states = state_dict["hidden_states"]
    
        # initialize gradients
        d_W_xf = np.zeros_like(self.W_xf)
        d_W_xi = np.zeros_like(self.W_xi)
        d_W_xg = np.zeros_like(self.W_xg)
        d_W_xo = np.zeros_like(self.W_xo)
        d_W_hf = np.zeros_like(self.W_hf)
        d_W_hi = np.zeros_like(self.W_hi)
        d_W_hg = np.zeros_like(self.W_hg)
        d_W_ho = np.zeros_like(self.W_ho)
        d_W_y = np.zeros_like(self.W_y) # (vocab_size, hidden_size)
        
        d_b_f = np.zeros_like(self.b_f)
        d_b_i = np.zeros_like(self.b_i)
        d_b_g = np.zeros_like(self.b_g)
        d_b_o = np.zeros_like(self.b_o)
        d_b_y = np.zeros_like(self.b_y)
        
        d_h_next = np.zeros_like(hidden_states[0])
        d_c_next = np.zeros_like(cell_states[0])
        
        loss = 0
        for t in reversed(range(len(outputs))):
            # cross-entropy loss
            loss += -np.mean(np.log(outputs[t]) * targets[t])
            c_prev = cell_states[t-1]
            
            # dL/dy
            d_y = np.copy(outputs[t])
            d_y[np.argmax(targets[t])] -= 1
            
            # dL/dW_y
            d_W_y += np.dot(d_y, hidden_states[t].T) # (vocab_size, hidden_size)
            d_b_y += d_y # (vocab_size, 1)
            
            # dL/dh (derivative of hidden_state)
            d_h = np.dot(self.W_y.T, d_y) + d_h_next # (hidden_size, 1)
            
            # dL/do (derivative of output gate)
            d_o = d_h * tanh(cell_states[t]) # (hidden_size, 1)
            d_o = d_o * sigmoid(o_dict[t], derivative=True)
            d_W_ho += np.dot(d_o, hidden_states[t-1].T) # (hidden_size, hidden_size)
            d_W_xo += np.dot(d_o, inputs[t].T) # (hidden_size, vocab_size)
            d_b_o += d_o
            
            # dL/dc (derivative of current cell state)
            d_c = np.copy(d_c_next)
            d_c += d_h * o_dict[t] * tanh(tanh(cell_states[t]), derivative=True)

            # dL/dg (derivative of g)
            d_g = d_c * i_dict[t] # (hidden_size, 1)
            d_g = d_g * tanh(g_dict[t], derivative=True)
            d_W_hg += np.dot(d_g, hidden_states[t-1].T) # (hidden_size, hidden_size)
            d_W_xg += np.dot(d_g, inputs[t].T) # (hidden_size, vocab_size)
            d_b_g += d_g
            
            # dL/di (derivative of input gate)
            d_i = d_c * g_dict[t]
            d_i = d_i * sigmoid(i_dict[t], derivative=True)
            d_W_hi += np.dot(d_i, hidden_states[t-1].T) # (hidden_size, hidden_size)
            d_W_xi += np.dot(d_i, inputs[t].T) # (hidden_size, vocab_size)
            d_b_i += d_i
            
            # dL/df (derivative of forget gate)
            d_f = d_c * c_prev
            d_f = d_f * sigmoid(f_dict[t], derivative=True)
            d_W_hf += np.dot(d_f, hidden_states[t-1].T) # (hidden_size, hidden_size)
            d_W_xf += np.dot(d_f, inputs[t].T) # (hidden_size, vocab_size)
            d_b_f += d_f
            
            # compute d_h_prev, d_c_prev
            d_h_prev = np.dot(self.W_hf, d_f) +\
                        np.dot(self.W_hi.T, d_i) +\
                        np.dot(self.W_hg.T, d_g) +\
                        np.dot(self.W_ho.T, d_o)
            d_c_prev = d_c * f_dict[t]
            
        gradients = (
            d_W_hf, d_W_hg, d_W_hi, d_W_ho, 
            d_W_xf, d_W_xg, d_W_xi, d_W_xo, d_W_y,
            d_b_f, d_b_g, d_b_i, d_b_o, d_b_y
        )
        
        gradients = clip_gradient_norm(gradients)
        return loss, gradients
        
    def optimize(self, gradients):
        (d_W_hf, d_W_hg, d_W_hi, d_W_ho, 
         d_W_xf, d_W_xg, d_W_xi, d_W_xo, d_W_y,
         d_b_f, d_b_g, d_b_i, d_b_o, d_b_y) = gradients
        
        self.W_xf -= self.learning_rate * d_W_xf
        self.W_xi -= self.learning_rate * d_W_xi
        self.W_xg -= self.learning_rate * d_W_xg
        self.W_xo -= self.learning_rate * d_W_xo
        self.W_hf -= self.learning_rate * d_W_hf
        self.W_hi -= self.learning_rate * d_W_hi
        self.W_hg -= self.learning_rate * d_W_hg
        self.W_ho -= self.learning_rate * d_W_ho
        self.W_y -= self.learning_rate * d_W_y
        
        self.b_f -= self.learning_rate * d_b_f
        self.b_i -= self.learning_rate * d_b_i
        self.b_g -= self.learning_rate * d_b_g
        self.b_o -= self.learning_rate * d_b_o
        self.b_y -= self.learning_rate * d_b_y