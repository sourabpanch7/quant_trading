import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df['source_file'] = path
    df['load_timestamp'] = pd.Timestamp("now")

    return df
