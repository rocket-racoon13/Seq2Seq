from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
    ):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # X -> Using LongTensor (indexing) for nn.Embedding
        X = torch.LongTensor(self.inputs[idx])
        y = torch.FloatTensor(self.targets[idx])
        
        return X, y
    
    
def custom_collator(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pads variable length input to given max_len.
    Collates list of input, target into a separate, stacked torch.Tensor.
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = pad_sequences(inputs)   # variable length input padding
    targets = torch.stack(targets, dim=0)
    
    
def pad_sequences(sequences: List[torch.Tensor], max_len=None, padding="pre"):
    batch_size = len(sequences)
    if max_len is None:
        longest_len = max([len(seq) for seq in sequences])
    else:
        longest_len = max_len
    padded_seq = torch.zeros((batch_size, longest_len), dtype=torch.long)
    
    for i in range(batch_size):
        seq = sequences[i]
        if padding == "pre":
            padded_seq[i,-len(seq):] = seq
        elif padding == "post":
            padded_seq[i,:len(seq)] = seq
    return padded_seq