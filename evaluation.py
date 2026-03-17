import logging
import mlflow
import pandas as pd
import numpy as np
from src.utils.utility import create_or_set_experiment, get_file_names
from src.evaluation.metrics import calculate_sharpe_ratio, calculate_drawdown, run_portfolio_backtest, calculate_spread
from src.evaluation.stat_arb import run_stat_arb_strategy, find_stat_arb_pairs
from src.evaluation.visualisations import *


def calculate_metrics(df, fig_path='resources/outputs/evaluation_plots'):
    daily_returns, turnover = run_portfolio_backtest(df)
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd, avg_dd = calculate_drawdown(daily_returns)
    pairs = find_stat_arb_pairs(df, 0.5)
    plot_equity_curve(daily_returns, f'{fig_path}/equity_curve.png')
    plot_drawdown(daily_returns, f'{fig_path}/drawdown_curve.png')
    plot_predictions_vs_actual(df, f'{fig_path}/prediction_vs_actual.png')
    plot_prediction_distribution(df, f'{fig_path}/prediction_distribution.png')
    plot_daily_return_distribution(df, f'{fig_path}/daily_returns_distribution.png')
    plot_turnover(df, f'{fig_path}/turnover.png')
    plot_ic(df, f'{fig_path}/information_coefficient.png')
    plot_stock_correlation(df, f'{fig_path}/stock_correlation.png')

    metrics = {"sharpe_ratio": sharpe, "max_drawdown": max_dd, "average_drawdown": avg_dd}
    spreads = []
    for pair in pairs:
        spread = calculate_spread(pred_df, pair[0], pair[1])
        spreads.append(spread)

    spread_df = pd.concat(spreads)
    plot_spread(spread_df, f'{fig_path}/spread_viz.png')
    plot_spread_zscore(spread_df, f'{fig_path}/spread_zscore_viz.png')
    fin_df = run_stat_arb_strategy(spread_df)
    return metrics, fin_df


if __name__ == "__main__":
    np.random.seed(42)
    logging.getLogger().setLevel(level=logging.INFO)
    FIGURE_PATH = 'resources/outputs/evaluation_plots'
    try:
        pred_df = pd.read_csv('resources/outputs/outputs/pred_test.csv')

        dataset = mlflow.data.from_pandas(
            pred_df,
            source="resources/outputs/outputs/pred_test.csv",
            name="predictions"
        )
        metrics, fin_df = calculate_metrics(pred_df, FIGURE_PATH)
        strategy_fin_df = fin_df[fin_df['strategy_return'] > 0]
        strategy_fin_df["stock_a"] = strategy_fin_df["stock_a"].astype(str).str.zfill(3)
        strategy_fin_df["stock_b"] = strategy_fin_df["stock_b"].astype(str).str.zfill(3)
        pair_dict = strategy_fin_df[["stock_a", "stock_b"]].drop_duplicates().to_dict(orient='records')[0]
        metrics = {**metrics, **pair_dict}
        run_timestamp = create_or_set_experiment()
        with mlflow.start_run(run_name=f"evaluation_{run_timestamp}"):
            mlflow.log_metrics(metrics)
            mlflow.log_input(dataset, context="evaluation")
            mlflow.log_table(
                data=strategy_fin_df,
                artifact_file="positive_strategy_return.json"  # Name of the artifact file
            )
            for file in get_file_names(FIGURE_PATH):
                mlflow.log_artifact(file)

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg

    finally:
        logging.info("EVALUATION DONE")
