import logging
import mlflow
from src.eda.correlation_analysis import CorrelationCalculation
from src.eda.clustering_analaysis import ClusteringCalculation
from src.utils.utility import read_full_data, create_or_set_experiment, get_file_names

if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:

        combined_df = read_full_data()

        correlation_obj = CorrelationCalculation(df=combined_df,
                                                 img_path=r'resources/outputs/eda_plots',
                                                 threshold=0.16)

        correlation_obj.perform_eda()
        clustering_obj = ClusteringCalculation(df=combined_df,
                                               img_path=r'resources/outputs/eda_plots',
                                               artifact_path=r'resources/outputs/artifacts')

        clustering_obj.perform_clustering_analysis()

        params = {
            "num_stocks": 100,
            "max_lag_tested": correlation_obj.lags,
            "pair_threshold": correlation_obj.threshold,
        }

        metrics = {
            "optimal_lag": correlation_obj.optimal_lag,
            "max_lag_correlation": max(correlation_obj.lag_scores.values()),
            "optimal_k": clustering_obj.optimal_k,
            "best_silhouette_score": clustering_obj.silhouette_score
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
        clustering_obj.df.to_csv(
            "resources/inputs/engineered_data.csv",
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
            for file in get_file_names('resources/outputs/eda_plots/'):
                mlflow.log_artifact(file)

            # log dataset outputs
            if not is_pairs_df_empty:
                # mlflow.log_artifact("resources/outputs/outputs/leader_follower_pairs.csv")
                mlflow.log_table(
                    data=correlation_obj.pairs_df,
                    artifact_file="leader_follower_pairs.json"  # Name of the artifact file
                )

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("EDA DONE")
