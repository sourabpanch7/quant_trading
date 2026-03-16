import numpy as np
import pandas as pd
import statsmodels.api as sm


def run_portfolio_backtest(df, transaction_cost=0.01):
    df['abs_signal'] = df.groupby("Date")["signal"].transform(lambda x: np.abs(x).sum())
    df["weight"] = df["signal"] / df["abs_signal"].replace(0, np.nan)
    df["weight"] = df["weight"].fillna(0)
    df["gross_return"] = df["weight"] * df["actual_return"]
    df["prev_weight"] = df.groupby("stock_id")["weight"].shift(1).fillna(0)
    df["turnover_component"] = np.abs(df["weight"] - df["prev_weight"])
    turnover = df.groupby("Date")["turnover_component"].sum().mean()
    df["transaction_cost"] = transaction_cost * df["turnover_component"]
    df["net_return"] = df["gross_return"] - df["transaction_cost"]
    daily_returns = df.groupby("Date")["net_return"].sum()
    return daily_returns, turnover


def calculate_sharpe_ratio(daily_returns, annualization_factor=252):
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    return np.sqrt(annualization_factor) * (mean_return / std_return)


def calculate_drawdown(daily_returns):
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown.mean()

    return max_drawdown, avg_drawdown


def calculate_spread(df, stock_a, stock_b):
    df_a = df[df["stock_id"] == stock_a][["Date", "Close"]]
    df_b = df[df["stock_id"] == stock_b][["Date", "Close"]]
    merged = pd.merge(df_a, df_b, on="Date", suffixes=("_a", "_b"))
    merged["spread"] = merged["Close_a"] - merged["Close_b"]
    merged["zscore"] = (merged["spread"] - merged["spread"].mean()) / merged["spread"].std()
    merged["stock_a"] = stock_a
    merged["stock_b"] = stock_b
    return merged

