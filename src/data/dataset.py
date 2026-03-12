import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):

    def __init__(self, X, y, window):
        self.X = X
        self.y = y
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window]
        y_val = self.y[idx + self.window]

        return torch.tensor(X_seq, dtype=torch.float32), \
            torch.tensor(y_val, dtype=torch.float32)
