import logging
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.evaluation.metrics import sharpe_ratio, max_drawdown, turnover
from src.utils.utility import create_or_set_experiment

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        pred_df = pd.read_csv('resources/outputs/outputs/y_pred_test.csv')
        logging.info(pred_df["pred_return"].describe())
        logging.info(pred_df["signal"].value_counts())

        logging.info(pred_df["signal"].diff().abs().sum())

        mse = mean_squared_error(pred_df['true_return'], pred_df['pred_return'])

        metrics = {}
        metrics['mse'] = mse

        sharpe_ratio = sharpe_ratio(pred_df["strategy_return"])
        metrics['sharpe_ratio'] = sharpe_ratio

        max_drawdown = max_drawdown(pred_df["equity_curve"])
        metrics['max_drawdown'] = max_drawdown

        turnover = turnover(pred_df["signal"])
        metrics['turnover'] = turnover

        logging.info(metrics)

        dataset = mlflow.data.from_pandas(
            pred_df,
            source="resources/outputs/outputs/y_pred_test.csv",
            name="predictions"
        )

        run_timestamp = create_or_set_experiment()

        with mlflow.start_run(run_name=f"evaluation_{run_timestamp}"):
            mlflow.log_metrics(metrics)
            mlflow.log_input(dataset, context="evaluation")

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg

    finally:
        logging.info("EVALUATION DONE")
