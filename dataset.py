from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path="data/simple.txt",
        one_hot_encode=True
    ):
        with open(f"{file_path}", encoding="utf-8-sig") as f_in:
            self.data = f_in.read().splitlines() # list of str
        self.inputs = [list(sequence) for sequence in self.data]
        self.targets = [list(sequence)[1:] + ["EOS"] for sequence in self.data]
        
        if one_hot_encode:
            
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        X = self.inputs[idx]
        y = self.targets[idx]
        return X, y