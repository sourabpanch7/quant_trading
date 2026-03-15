import os
import logging
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import mlflow
import torch


def create_sequences(X, stock_ids, seq_length):
    X_seq = []
    stock_seq = []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        stock_seq.append(stock_ids[i + seq_length])

    return np.array(X_seq), np.array(stock_seq)


def load_pyfunc_model(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def prepare_inference_dataframe(X_seq, stock_seq):
    df = pd.DataFrame({
        "sequence": [x.tolist() for x in X_seq],
        "stock_id": stock_seq.astype(int)
    })
    return df


def run_inference(X_test, stock_ids, model, seq_length=30):
    # Create sequences
    X_seq, stock_seq = create_sequences(
        X_test,
        stock_ids,
        seq_length
    )

    # Prepare dataframe for pyfunc model
    inference_df = prepare_inference_dataframe(
        X_seq,
        stock_seq
    )

    # Load MLflow model

    # Predict
    preds = model.predict(inference_df)

    return preds, X_seq, stock_seq


if __name__ == "__main__":
    np.random.seed(42)
    logging.getLogger().setLevel(level=logging.INFO)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    try:
        model_name = "Stock_Price_GNN_Model"

        model_metadata = client.get_latest_versions(model_name)
        latest_model_version = model_metadata[0].version

        model_uri = client.get_model_version_download_uri(name=model_name, version=latest_model_version)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)

        # Example loading test data
        test_data = pd.read_csv('resources/outputs/outputs/data_complete.csv')
        test_data = test_data[test_data["stock_id"].isin('003', '032', '049', '068', '071')]
        stock_ids = test_data["stock_id"].astype(int).unique().tolist()
        X_test = test_data.drop(columns=["target", "Date", "stock_id"])

        # Run inference
        predictions, X_seq, stock_seq = run_inference(
            X_test,
            stock_ids,
            pyfunc_model,
            seq_length=30
        )
        print(predictions)
    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("INFERENCE DONE")
