import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import custom_collator


class Tester:
    def __init__(
        self,
        args,
        test_ds,
        model,
        device
    ):
        self.args = args    
        self.test_ds = test_ds

        self.device = device
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        
        self.test_loss = 0
        
    def test(self):
        test_loader = DataLoader(
            self.test_ds,
            self.args.batch_size,
            shuffle=False,
            collate_fn=custom_collator
        )
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = [b.to(self.device) for b in batch]
                input, target = batch
                
                y_pred = self.model(input)
                loss = self.loss_func(y_pred, target)
                self.test_loss += loss
            
        print(f"=== Test Average Loss: {self.loss / len(test_loader):.4f}")