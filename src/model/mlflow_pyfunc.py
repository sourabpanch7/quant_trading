import mlflow.pyfunc
import torch
import numpy as np
from src.utils.utility import get_device

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
