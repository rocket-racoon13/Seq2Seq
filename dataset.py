import numpy as np
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
        X = np.array(self.inputs[idx])
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = np.array(self.targets[idx])
        y = y.reshape(y.shape[0], y.shape[1], 1)
        return X, y