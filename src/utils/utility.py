import os
import json
import mlflow


def get_file_names(directory_path):
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            yield full_path


def read_config(file_path):
    with open(file_path, 'r') as fl:
        cnf = json.load(fl)

    return cnf


def create_or_set_experiment(exp_name="quant_trading_lstm"):
    mlflow.set_experiment(exp_name)
