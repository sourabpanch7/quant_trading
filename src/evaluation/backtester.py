import numpy as np


def backtest(returns, signals, transaction_cost=0.001):
    capital = 1_000_000
    portfolio = []
    prev_signal = 0
    for r, s in zip(returns, signals):
        cost = abs(s - prev_signal) * transaction_cost
        pnl = s * r
        capital *= (1 + pnl - cost)
        portfolio.append(capital)
        prev_signal = s
    return np.array(portfolio)
