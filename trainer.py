from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from data_utils import *


class Trainer:
    def __init__(
        self,
        args,
        train_ds,
        eval_ds,
        model
    ):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.model = model
        self.training_loss = []
        self.validation_loss = []
        
    def train(self):
        for i in tqdm(range(self.num_epochs)):
            train_loader = DataLoader(self.train_ds, self.batch_size, shuffle=True)
            epoch_training_loss = 0
            epoch_validation_loss = 0
            
            for step, batch in enumerate(train_loader):
                input, target = batch
                
            
            
            for inputs, targets in training_set:
                
                inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
                targets_one_hot = one_hot_encode_sequence(targets, vocab_size)
                
                hidden_state = np.zeros_like(hidden_state)
                
                outputs, hidden_states = self.model.forward(
                    inputs_one_hot, hidden_state
                )
                loss, gradients = self.model.backward(
                    inputs_one_hot, outputs, hidden_states, targets_one_hot
                )
                
                if np.isnan(loss):
                    raise ValueError('Gradients have vanished!')
                
                params = self.model.optimize(gradients)
                
                epoch_training_loss += loss
                
    def valid(self):
        pass