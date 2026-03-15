import mlflow.pyfunc
import torch
import numpy as np
from src.utils.utility import get_device
from src.model.lstm_gnn import StockPriceHybridModel

np.random.seed(42)


class LSTMPyFuncModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model, seq_length, n_features):
        self.model = model.half()
        self.seq_length = seq_length
        self.n_features = n_features
        self.device = get_device()

    def predict(self, context, model_input):
        # Convert dataframe to numpy
        X = model_input.values

        # reshape flattened input → LSTM format
        X = X.reshape(-1, self.seq_length, self.n_features)

        X_tensor = torch.tensor(
            X,
            dtype=torch.float16
        ).to(self.device)

        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_tensor)

        return preds.cpu().numpy().flatten()


class LSTMGNNPyFuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.device = get_device()
        self.edge_index = torch.load(
            context.artifacts["edge_index"]
        ).to(self.device)
        checkpoint = torch.load(context.artifacts["model_path"],map_location=self.device)
        input_size = checkpoint["input_size"]
        hidden_dim = checkpoint["hidden_dim"]
        num_stocks = checkpoint["num_stocks"]
        self.model = StockPriceHybridModel(input_size,hidden_dim,num_stocks).to
        self.model = self.model.half()
        self.model.eval()


    def predict(self, context, model_input):

        X = model_input["X"]
        stock_ids = model_input["stock_id"]

        X = torch.tensor(X, dtype=torch.float16)
        stock_ids = torch.tensor(stock_ids)

        with torch.no_grad():

            preds = self.model(
                X,
                stock_ids,
                self.edge_index
            )

        return preds.cpu().numpy()