import logging
import joblib
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from src.data.dataset import StockGNNDataset
from src.model.lstm_gnn import StockPriceHybridModel
from src.model.early_stopping import EarlyStopping
from src.model.mlflow_pyfunc import LSTMGNNPyFuncModel
from src.model.graph_utils import build_stock_graph
from src.utils.utility import read_config, create_or_set_experiment, get_device, create_gnn_sequences

if __name__ == "__main__":
    np.random.seed(42)
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        config = read_config('resources/config/config.json')
        combined_df = pd.read_csv(config["feature_engineered_data_path"])

        # combined_df["target"] = combined_df.groupby("stock_id")["Close"].shift(-1) / combined_df["Close"] - 1

        combined_df = combined_df.dropna()

        combined_df = combined_df.sort_values(by=['stock_id', 'Date']).reset_index(drop=True)

        combined_df.to_csv(config["op_path"], index=False)
        features = combined_df.drop(columns=["target", "Date", "stock_id"])
        feature_columns = features.columns.tolist()

        stock_ids = combined_df['stock_id'].astype(int).unique().tolist()
        # stock_ids = np.array(stock_ids)

        device = get_device()

        edge_index = build_stock_graph(combined_df, threshold=0.2)

        edge_index = edge_index.to(device)

        n_splits = config["n_splits"]
        tscv = TimeSeriesSplit(n_splits=n_splits)

        seq_length = config["seq_length"]

        scaler = StandardScaler()

        run_timestamp = create_or_set_experiment()

        with mlflow.start_run(run_name=f"training_{run_timestamp}"):
            mlflow.log_params({
                "model": "lstm_gnn",
                "split_method": "TimeSeriesSplit",
                "n_splits": n_splits,
                "seq_length": seq_length,
                "batch_size": config["batch_size"],
                "hidden_dim": config["hidden_dim"],
                "num_layers": config["num_layers"],
                "optimizer": "Adam",
                "lr": config["learning_rate"],
                "loss": "Huber",
                "target": "1_day_forward_return",
                "early_stopping_patience": config["early_stopping_patience"],
                "graph_threshold": config["graph_threshold"]
            })
            for fold, (train_idx, val_idx) in enumerate(tscv.split(features)):
                logging.info(f"Training fold {fold + 1}")

                df_train = combined_df.iloc[train_idx]
                df_val = combined_df.iloc[val_idx]

                scaler.fit(df_train[feature_columns])

                joblib.dump(scaler,config["scaler_path"])

                # Create sequences AFTER splitting
                X_train_seq, stock_train_seq, y_train_seq, train_dates, train_prices = create_gnn_sequences(
                    df_train,
                    feature_columns,
                    seq_length,
                    scaler
                )

                X_val_seq, stock_val_seq, y_val_seq, val_dates, val_prices = create_gnn_sequences(
                    df_val,
                    feature_columns,
                    seq_length,
                    scaler
                )

                train_dataset = StockGNNDataset(
                    X_train_seq,
                    y_train_seq,
                    stock_train_seq
                )

                val_dataset = StockGNNDataset(
                    X_val_seq,
                    y_val_seq,
                    stock_val_seq
                )

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=config["batch_size"],
                    shuffle=True
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=config["batch_size"],
                    shuffle=False
                )

                input_size = len(feature_columns)

                model = StockPriceHybridModel(
                    input_size=input_size,
                    hidden_dim=config["hidden_dim"]
                )
                model.to(device)

                # criterion = torch.nn.MSELoss()

                criterion = torch.nn.HuberLoss()

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config["learning_rate"],
                    weight_decay=1e-5
                )

                epochs = config.get("epochs", 50)
                early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 8))

                for epoch in range(epochs):

                    model.train()

                    train_loss = 0

                    for X_batch, stock_batch, y_batch in train_loader:
                        X_batch = X_batch.to(device)
                        stock_batch = stock_batch.to(device)
                        y_batch = y_batch.to(device)

                        preds = model(
                            X_batch,
                            stock_batch,
                            edge_index
                        )

                        loss = criterion(preds, y_batch)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()

                    train_loss /= len(train_loader)
                    model.eval()

                    val_loss = 0

                    with torch.no_grad():

                        for X_batch, stock_batch, y_batch in val_loader:
                            X_batch = X_batch.to(device)
                            stock_batch = stock_batch.to(device)
                            y_batch = y_batch.to(device)
                            preds_val = model(
                                X_batch,
                                stock_batch,
                                edge_index
                            )

                            loss = criterion(preds_val, y_batch)

                            val_loss += loss.item()

                    val_loss /= len(val_loader)

                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss
                        },
                        step=epoch
                    )

                    early_stopping(val_loss)

                    logging.info(f"Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss:.6f}")

                    torch.save({"model_state_dict": model.state_dict()
                                   , "optimizer_state_dict": optimizer.state_dict()
                                   , "val_loss": val_loss}, config["model_path"])

                    if early_stopping.stop:
                        logging.info("Early stopping triggered")
                        break

            n_features = X_train_seq.shape[2]

            input_example = pd.DataFrame({
                "sequence": [X_train_seq[0].tolist()],
                "stock_id": [int(stock_train_seq[0])]
            }
            )

            torch.save(edge_index.cpu(), config["edge_index_op_path"])
            mlflow.pyfunc.log_model(
                artifact_path="stock_price_gnn_model",
                python_model=LSTMGNNPyFuncModel(),
                registered_model_name="Stock_Price_GNN_Model",
                input_example=input_example,
                artifacts={
                    "model_path": config["model_path"],
                    "edge_index": config["edge_index_op_path"]
                }
            )

            mlflow.log_artifact(config["scaler_path"])
            mlflow.log_dict(feature_columns, config.get("feature_names_artifact_name", "feature_names.json"))

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("TRAINING DONE")
