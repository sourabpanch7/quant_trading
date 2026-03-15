import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class StockPriceHybridModel(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size=64
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.gcn1 = GCNConv(hidden_size, 64)
        self.gcn2 = GCNConv(64, 32)

        self.fc = nn.Linear(32, 1)

    def forward(self, x, stock_ids, edge_index):
        lstm_out, _ = self.lstm(x)

        h = lstm_out[:, -1, :]

        g = torch.relu(self.gcn1(h, edge_index))
        g = torch.relu(self.gcn2(g, edge_index))
        node_features = g[stock_ids]
        out = self.fc(node_features)

        return out.squeeze()
