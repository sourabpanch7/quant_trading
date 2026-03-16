import os
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
from concurrent.futures import ProcessPoolExecutor
from src.data.load_data import load_data
import torch


def get_device():
    if torch.backends.mps.is_available():
        logging.info("Using Apple Metal GPU (MPS)")
        return torch.device("mps")

    else:
        print("Using CPU")
        return torch.device("cpu")


def get_file_names(directory_path):
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            yield full_path


def read_full_data(file_path='resources/anonymized_data'):
    files = get_file_names(file_path)
    with ProcessPoolExecutor(max_workers=3) as executor:
        df_list = list(executor.map(load_data, files))

    return pd.concat(df_list, ignore_index=True)


def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])

    return np.array(X_seq), np.array(y_seq)


def read_config(file_path):
    with open(file_path, 'r') as fl:
        cnf = json.load(fl)

    return cnf


def create_or_set_experiment(exp_name="quant_algo_trading"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name=exp_name)
    run_timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return run_timestamp


def create_gnn_sequences(df, feature_cols, seq_length, scaler):
    X_seq = []
    stock_seq = []
    y_seq = []
    date_seq = []
    price_seq = []

    for stock_id, group in df.groupby("stock_id"):

        group = group.sort_values("Date")

        X = group[feature_cols]
        X = scaler.transform(X)
        # X = X.values
        y = group["target"].values
        dates = group["Date"].values
        prices = group["Close"].values

        for i in range(len(group) - seq_length):
            X_seq.append(X[i:i + seq_length])
            stock_seq.append(stock_id)
            y_seq.append(y[i + seq_length])
            date_seq.append(dates[i + seq_length])
            price_seq.append(prices[i + seq_length])

    return (
        np.array(X_seq),
        np.array(stock_seq),
        np.array(y_seq),
        np.array(date_seq),
        np.array(price_seq)
    )
