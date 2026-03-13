import pandas as pd


def load_data(path):
    stock_id = path.split('/')[-1].split('_')[-1].split('.')[0]
    df = pd.read_csv(path)
    df['stock_id'] = stock_id

    return df
