import logging
import torch
import numpy as np
import pandas as pd
import joblib
import mlflow.pytorch
from mlflow.data.numpy_dataset import from_numpy
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from src.data.dataset import StockDataset
from src.model.lstm_model import StockPriceModel
from src.model.early_stopping import EarlyStopping
from src.model.mlflow_pyfunc import LSTMPyFuncModel
from src.utils.utility import read_full_data, create_or_set_experiment, get_device

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:

        combined_df = read_full_data()

        combined_df["target"] = combined_df.groupby("stock_id")["Close"].shift(-1) / combined_df["Close"] - 1

        combined_df = combined_df.dropna()

        features = combined_df.drop(columns=["target", "Date", "stock_id"])

        target = combined_df["target"].values

        X = features.values
        y = target

        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)

        splits = list(tscv.split(X))

        train_indices = np.concatenate([splits[i][1] for i in range(len(splits) - 2)])
        val_indices = splits[-2][1]
        test_indices = splits[-1][1]

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_val = X[val_indices]
        y_val = y[val_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_val = scaler.transform(X_val)

        X_test = scaler.transform(X_test)

        joblib.dump(scaler, "resources/outputs/artifacts/scaler.pkl")

        device = get_device()

        seq_length = 30
        batch_size = 64

        train_dataset = StockDataset(X_train, y_train, seq_length)
        val_dataset = StockDataset(X_val, y_val, seq_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        input_dim = X_train.shape[1]

        model = StockPriceModel(input_dim)
        model.to(device)

        criterion = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )

        epochs = 100
        early_stopping = EarlyStopping(patience=8)

        run_timestamp = create_or_set_experiment()

        with mlflow.start_run(run_name=f"training_{run_timestamp}"):
            mlflow.log_params({
                "model": "LSTM",
                "split_method": "TimeSeriesSplit",
                "n_splits": n_splits,
                "seq_length": seq_length,
                "batch_size": batch_size,
                "hidden_dim": 64,
                "num_layers": 2,
                "optimizer": "Adam",
                "lr": 0.001,
                "loss": "MSE",
                "target": "1_day_forward_return",
                "early_stopping_patience": 8
            })

            for epoch in range(epochs):

                model.train()
                train_losses = []

                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()

                    X_batch = X_batch.to(device)
                    preds = model(X_batch)

                    y_batch = y_batch.to(device)
                    loss = criterion(preds, y_batch)

                    loss.backward()

                    optimizer.step()

                    train_losses.append(loss.item())

                train_loss = np.mean(train_losses)

                model.eval()

                val_losses = []

                with torch.no_grad():

                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)

                        preds_val = model(X_batch)

                        loss = criterion(preds_val, y_batch)

                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss
                    },
                    step=epoch
                )

                early_stopping(val_loss)

                logging.info(f"Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss:.6f}")

                if early_stopping.stop:
                    logging.info("Early stopping triggered")
                    break

            pyfunc_model = LSTMPyFuncModel(
                model=model,
                seq_length=30,
                n_features=X_train.shape[1],
                device=device
            )

            seq_length = 30
            n_features = X_train.shape[1]

            input_example_array = np.random.randn(1, seq_length * n_features)

            columns = [
                f"f_{t}_{f}"
                for t in range(seq_length)
                for f in range(n_features)
            ]

            input_example = pd.DataFrame(
                input_example_array,
                columns=columns
            )

            mlflow.pyfunc.log_model(
                artifact_path="stock_price_model",
                python_model=pyfunc_model,
                registered_model_name="Stock_Price_Model",
                input_example=input_example
            )

            mlflow.log_artifact("resources/outputs/artifacts/scaler.pkl")
            mlflow.log_dict(features.columns, "resources/outputs/artifacts/feature_names.json")

            numpy_dataset_train = from_numpy(features=X_train, targets=y_train, name="train_data")
            numpy_dataset_test = from_numpy(features=X_test, targets=y_test, name="test_data")
            numpy_dataset_val = from_numpy(features=X_val, targets=y_val, name="val_data")

            mlflow.log_input(dataset=numpy_dataset_train, context="training")
            mlflow.log_input(dataset=numpy_dataset_test, context="test")
            mlflow.log_input(dataset=numpy_dataset_val, context="validation")

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("TRAINING DONE")
