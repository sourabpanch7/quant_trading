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


class StockGNNDataset(Dataset):

    def __init__(self, X_seq, y_seq, stock_ids):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_seq, dtype=torch.float32)
        stock_ids = [int(stock_id) for stock_id in stock_ids]
        self.stock_ids = torch.tensor(stock_ids, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.stock_ids[idx],
            self.y[idx]
        )
