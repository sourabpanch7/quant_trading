import torch.nn as nn


class StockPriceModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(StockPriceModel, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        return self.fc(last_output).squeeze()
