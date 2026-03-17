import matplotlib.pyplot as plt
import seaborn as sns


def plot_equity_curve(daily_returns, fig_path):
    cumulative_returns = (1 + daily_returns).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns)

    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")

    plt.grid(True)

    plt.savefig(fig_path, format="png")


def plot_drawdown(daily_returns, fig_path):
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max

    plt.figure(figsize=(10, 6))
    plt.plot(drawdown)

    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")

    plt.grid(True)

    plt.savefig(fig_path, format="png")


def plot_predictions_vs_actual(df, fig_path):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["predicted_return"],
        df["actual_return"],
        alpha=0.3
    )

    plt.title("Predicted vs Actual Return")
    plt.xlabel("Predicted Return")
    plt.ylabel("Actual Return")

    plt.grid(True)

    plt.savefig(fig_path, format="png")


def plot_prediction_distribution(df, fig_path):
    plt.figure(figsize=(8, 6))
    plt.hist(df["predicted_return"].unique(), bins=50)

    plt.title("Distribution of Model Predictions")
    plt.xlabel("Predicted Return")
    plt.ylabel("Frequency")

    plt.savefig(fig_path, format="png")


def plot_daily_return_distribution(daily_returns, fig_path):
    plt.figure(figsize=(8, 6))
    plt.hist(daily_returns, bins=50)
    plt.title("Daily Portfolio Return Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.savefig(fig_path, format="png")


def plot_turnover(df, fig_path):
    df = df.sort_values(["stock_id", "Date"])
    df["prev_signal"] = df.groupby("stock_id")["signal"].shift(1)
    df["trade"] = abs(df["signal"] - df["prev_signal"])
    turnover_series = df.groupby("Date")["trade"].sum()
    plt.figure(figsize=(10, 6))

    plt.plot(turnover_series)

    plt.title("Portfolio Turnover Over Time")
    plt.xlabel("Date")
    plt.ylabel("Turnover")

    plt.savefig(fig_path, format="png")


def plot_stock_correlation(df, fig_path):
    pivot = df.pivot_table(index="Date", columns="stock_id", values="actual_return")
    corr = pivot.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Stock Return Correlation Heatmap")
    plt.savefig(fig_path, format="png")


def plot_spread(df, fig_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["spread"])
    plt.title("Spread Between Two 049 and 068")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.savefig(fig_path, format="png")


def plot_spread_zscore(df, fig_path):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["zscore"])
    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")
    plt.axhline(0)
    plt.title("Spread Z-score")
    plt.savefig(fig_path, format="png")


def plot_ic(df, fig_path):
    ic = df.groupby("Date").apply(
        lambda x: x["predicted_return"].corr(x["actual_return"])
    )
    plt.plot(ic)
    plt.title("Information Coefficient Over Time")

    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.savefig(fig_path, format="png")
