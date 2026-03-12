import numpy as np
import statsmodels.api as sm


def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()


def max_drawdown(portfolio):
    peak = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - peak) / peak
    return drawdown.min()


def turnover(signals):
    return np.sum(np.abs(np.diff(signals)))


def calculate_spread(x, y):
    model = sm.OLS(x, sm.add_constant(y)).fit()
    beta = model.params[1]
    spread = x - beta * y
    return spread
