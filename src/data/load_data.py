import pandas as pd
from src.data.kalman_filter import apply_kalman


def load_data(path):
    stock_id = path.split('/')[-1].split('_')[-1].split('.')[0]
    df = pd.read_csv(path)
    df['stock_id'] = stock_id
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].apply(apply_kalman,
                                                                                                          axis=0)
    return df
