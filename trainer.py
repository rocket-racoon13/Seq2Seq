from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import custom_collator


class Trainer:
    def __init__(
        self,
        args,
        train_ds,
        valid_ds,
        model,
        optimizer,
        scheduler,
        device
    ):
        self.args = args
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        
        self.device = device
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.steps = 0
        
    def train(self):
        self.model.train()
        
        for epoch_i in tqdm(range(self.num_epochs)):
            train_loader = DataLoader(
                self.train_ds,
                self.args.batch_size,
                shuffle=True,
                collate_fn=custom_collator
            )
            
            epoch_training_loss = 0
            for step, (inputs, targets) in enumerate(self.train_ds, 1):
                # initialize hidden_state[t-1]
                hidden_state = np.zeros((self.hidden_size, 1))
                outputs, hidden_states = self.model.forward(
                    inputs, hidden_state
                )
                loss, gradients = self.model.backward(
                    inputs, outputs, hidden_states, targets
                )
                epoch_training_loss += loss
                self.model.optimize(gradients)
                self.steps += 1
                
                if step % self.args.logging_steps == 0:
                    avg_loss = epoch_training_loss / step
                    print(f"Epoch: {epoch_i:3d} Batch: {step:2d} Loss: {avg_loss:4.4f}")
                    
                if self.steps % self.args.saving_steps == 0:
                    pass
            
            self.training_loss.append(epoch_training_loss / len(self.train_ds))
            self.eval()
            
    def eval(self):
        # valid_loader = DataLoader(self.valid_ds, self.batch_size, shuffle=False)
        validation_loss = 0
        for (inputs, targets) in self.valid_ds:
            hidden_state = np.zeros((self.hidden_size, 1))
            outputs, hidden_states = self.model.forward(
                inputs, hidden_state
            )
            loss, _ = self.model.backward(
                inputs, outputs, hidden_states, targets
            )
            validation_loss += loss
            
        print(f"Validation Loss: {validation_loss/len(self.valid_ds):4.4f}")