import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):

    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_length]
        y_val = self.y[idx + self.seq_length]

        return torch.tensor(X_seq, dtype=torch.float32), \
            torch.tensor(y_val, dtype=torch.float32)
