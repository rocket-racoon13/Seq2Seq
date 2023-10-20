import numpy as np
from model_utils import *


class RNN:
    def __init__(self, tokenizer, hidden_size=100, lr=1e-3):
        self.vocab_size = len(tokenizer["token_to_idx"])
        self.hidden_size = hidden_size
        self.learning_rate = lr
        
        self.U = np.zeros((self.hidden_size, self.vocab_size))
        self.V = np.zeros((self.hidden_size, self.hidden_size))
        self.W = np.zeros((self.vocab_size, self.hidden_size))
        self.b_hidden = np.zeros((self.hidden_size, 1))
        self.b_out = np.zeros((self.vocab_size, 1))
        
        for param in [self.U, self.V, self.W]:
            param = init_orthogonal(param)
    
    def forward(self, inputs, hidden_state):
        outputs, hidden_states = {}, {}
        hidden_states[-1] = hidden_state
        for t in range(len(inputs)):
            hidden_state = tanh(np.dot(self.U, inputs[t]) + np.dot(self.V, hidden_state[t-1]) + self.b_hidden)
            out = softmax(np.dot(self.W, hidden_state) + self.b_out)
            hidden_states[t] = hidden_state
            outputs[t] = out
        return outputs, hidden_states
    
    def clip_gradient_norm(grads, max_norm=0.25):
        """
        Clips gradients to have a maximum norm of `max_norm`.
        This is to prevent the exploding gradients problem.
        """ 
        # Set the maximum of the norm to be of type float
        max_norm = float(max_norm)
        total_norm = 0
        
        # Calculate the L2 norm squared for each gradient and add them to the total norm
        for grad in grads:
            grad_norm = np.sum(np.power(grad, 2))
            total_norm += grad_norm
        
        total_norm = np.sqrt(total_norm)
        
        # Calculate clipping coeficient
        clip_coef = max_norm / (total_norm + 1e-6)
        
        # If the total norm is larger than the maximum allowable norm, then clip the gradient
        if clip_coef < 1:
            for grad in grads:
                grad *= clip_coef
        
        return grads
    
    def backward(self, inputs, outputs, hidden_states, targets):
        d_U = np.zeros_like(self.U)
        d_V = np.zeros_like(self.V)
        d_W = np.zeros_like(self.W)
        d_b_hidden = np.zeros_like(self.b_hidden)
        d_b_out = np.zeros_like(self.b_out)
        
        d_h_next = np.zeros_like(hidden_states[0])
        loss = 0
        
        for t in reversed(range(len(outputs))):
            loss += -np.mean(np.log(outputs[t]+1e-12) * targets[t])
            
            # Backpropagate into output (derivative of cross-entropy)
            # output = softmax(y)
            # dL/dy = dL/dSm * dSm/dy = Sm - 1 (for target index)
            d_y = outputs[t].copy()
            d_y[np.argmax(targets[t])] -= 1 # target 위치의 loss를 줄이는 것
            
            # Backpropagate into W
            # dL/dW = dL/dy * h[t]
            d_W += np.dot(d_y, hidden_states[t].T)
            d_b_out += d_y
            
            # Backpropagate into h
            # dL/dh = W * dL/dy
            d_h = np.dot(self.W.T, d_y) + d_h_next
            
            # Backpropagate into h_raw
            # dL/dh_raw = dL/dh * dh/dh_raw
            d_h_raw = tanh(hidden_states[t], derivative=True) * d_h
            
            # Backpropagate into U
            # dL/dU = dL/dh_raw * x[t]
            d_U = np.dot(d_h_raw, inputs[t].T)
            
            # Backpropagate into V
            # dL/dV = dL/dh_raw * h[t-1]
            d_V = np.dot(d_h_raw, hidden_states[t-1].T)
            d_h_next = np.dot(self.V.T, d_h_raw)
            
        gradients = d_U, d_V, d_W, d_b_hidden, d_b_out
        gradients = self.clip_gradient_norm(gradients)
        return loss, gradients
            
    def optimize(self, gradients):
        params = [
            self.U,
            self.V,
            self.W,
            self.b_hidden,
            self.b_out
        ]
        for param, grad in zip(params, gradients):
            param -= self.learning_rate * grad
            
        return params