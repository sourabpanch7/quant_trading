import logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from src.utils.utility import create_or_set_experiment
from src.evaluation.metrics import sharpe_ratio, max_drawdown, turnover


def calculate_metrics(df):
    metrics = {}
    mse = mean_squared_error(df['true_return'], df['pred_return'])
    metrics['mse'] = mse
    rmse = root_mean_squared_error(df['true_return'], df['pred_return'])
    metrics['rmse'] = rmse
    r2 = r2_score(df['true_return'], df['pred_return'])
    metrics['r2'] = r2
    sharperatio = sharpe_ratio(df["strategy_return"])
    metrics['sharpe_ratio'] = sharperatio

    maxdrawdown = max_drawdown(df["equity_curve"])
    metrics['max_drawdown'] = maxdrawdown

    turn_over = turnover(df["signal"])
    metrics['turnover'] = turn_over
    return metrics


if __name__ == "__main__":
    np.random.seed(42)
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        pred_df = pd.read_csv('resources/outputs/outputs/y_pred_test.csv')
        logging.info(pred_df["pred_return"].describe())
        logging.info(pred_df["signal"].value_counts())

        logging.info(pred_df["signal"].diff().abs().sum())

        dataset = mlflow.data.from_pandas(
            pred_df,
            source="resources/outputs/outputs/y_pred_test.csv",
            name="predictions"
        )
        metrics = calculate_metrics(pred_df)
        run_timestamp = create_or_set_experiment()

        with mlflow.start_run(run_name=f"evaluation_{run_timestamp}"):
            mlflow.log_metrics(metrics)
            mlflow.log_input(dataset, context="evaluation")

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg

    finally:
        logging.info("EVALUATION DONE")
