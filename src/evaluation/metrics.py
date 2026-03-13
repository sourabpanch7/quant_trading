import numpy as np
import statsmodels.api as sm


def sharpe_ratio(returns):
    return np.sqrt(252) * returns.mean() / returns.std()


def max_drawdown(portfolio):
    rolling_max = portfolio.cummax()
    drawdown = portfolio / rolling_max - 1
    return drawdown.min()


def turnover(signals):
    position_change = signals.diff().abs()
    turnover = position_change.sum() / len(signals)
    return turnover


def calculate_spread(x, y):
    model = sm.OLS(x, sm.add_constant(y)).fit()
    beta = model.params[1]
    spread = x - beta * y
    return spread
