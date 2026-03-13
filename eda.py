import logging
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import mlflow
from src.data.load_data import load_data
from src.utils.utility import get_file_names
from src.eda.correlation_analysis import CorrelationCalculation

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        files = get_file_names('resources/anonymized_data')
        with ProcessPoolExecutor(max_workers=3) as executor:
            df_list = list(executor.map(load_data, files))

        combined_df = pd.concat(df_list, ignore_index=True)

        correlation_obj = CorrelationCalculation(df=combined_df,
                                                 img_path=r'/Users/sourabpanchanan/PycharmProjects/quant_trading/resources/outputs/eda_plots',
                                                 threshold=-0.16)

        correlation_obj.perform_eda()

        params = {
            "num_stocks": 100,
            "max_lag_tested": correlation_obj.lags,
            "pair_threshold": correlation_obj.threshold
        }

        metrics = {
            "optimal_lag": correlation_obj.optimal_lag,
            "max_lag_correlation": max(correlation_obj.lag_scores.values())
        }

        dataset = mlflow.data.from_pandas(
            combined_df,
            source="resources/anonymized_data",
            name="stock_trends"
        )

        is_pairs_df_empty = True

        if not correlation_obj.pairs_df.empty:
            correlation_obj.pairs_df.to_csv(
                "resources/outputs/analysis_outputs/leader_follower_pairs.csv",
                index=False)
            is_pairs_df_empty = False

        run_timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("quant_algo_trading_new")

        with mlflow.start_run(run_name=f"eda_{run_timestamp}"):

            mlflow.log_input(dataset, context="eda")

            # log parameters
            mlflow.log_params(params)

            # log metrics
            mlflow.log_metrics(metrics)

            # log plots
            mlflow.log_artifact("resources/outputs/eda_plots/followers_visualisation.png")
            mlflow.log_artifact("resources/outputs/eda_plots/lag_score_visualisation.png")
            mlflow.log_artifact("resources/outputs/eda_plots/lead_score_heatmap.png")
            mlflow.log_artifact("resources/outputs/eda_plots/leaders_visualisation.png")
            # log dataset outputs
            if not is_pairs_df_empty:
                mlflow.log_artifact("resources/outputs/analysis_outputs/leader_follower_pairs.csv")

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("DONE")
