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
        X = self.inputs[idx]
        y = self.targets[idx]
        return X, y