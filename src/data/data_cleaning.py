import numpy as np
import pandas as pd


def add_features(df):

    df = df.sort_values(["stock_id", "Date"])

    # ---------- LOG RETURNS ----------
    df["log_return"] = (
        df.groupby("stock_id")["Close"]
        .transform(lambda x: np.log(x).diff())
    )

    # ---------- MOMENTUM FEATURES ----------
    df["ret_5"] = (
        df.groupby("stock_id")["Close"]
        .pct_change(5)
    )

    df["ret_10"] = (
        df.groupby("stock_id")["Close"]
        .pct_change(10)
    )

    df["ret_20"] = (
        df.groupby("stock_id")["Close"]
        .pct_change(20)
    )

    # ---------- VOLATILITY ----------
    df["volatility_10"] = (
        df.groupby("stock_id")["log_return"]
        .rolling(10)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["volatility_20"] = (
        df.groupby("stock_id")["log_return"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ---------- MOVING AVERAGES ----------
    df["ma_10"] = (
        df.groupby("stock_id")["Close"]
        .rolling(10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ma_20"] = (
        df.groupby("stock_id")["Close"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ---------- MEAN REVERSION ----------
    df["price_ma10_ratio"] = df["Close"] / df["ma_10"]
    df["price_ma20_ratio"] = df["Close"] / df["ma_20"]

    # ---------- VOLUME FEATURES ----------
    df["volume_ma10"] = (
        df.groupby("stock_id")["Volume"]
        .rolling(10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["volume_ratio"] = df["Volume"] / df["volume_ma10"]

    df = df.dropna().reset_index(drop=True)

    return df
