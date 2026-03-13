import logging
import mlflow
from src.eda.correlation_analysis import CorrelationCalculation
from src.utils.utility import read_full_data, create_or_set_experiment

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:

        combined_df = read_full_data()

        correlation_obj = CorrelationCalculation(df=combined_df,
                                                 img_path=r'/Users/sourabpanchanan/PycharmProjects/quant_trading/resources/outputs/eda_plots',
                                                 threshold=0.16)

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
                "resources/outputs/outputs/leader_follower_pairs.csv",
                index=False)
            is_pairs_df_empty = False

        run_timestamp = create_or_set_experiment()

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
                mlflow.log_artifact("resources/outputs/outputs/leader_follower_pairs.csv")

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("EDA DONE")
