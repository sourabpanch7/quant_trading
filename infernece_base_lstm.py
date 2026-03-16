import logging
import joblib
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import mlflow
from src.utils.utility import create_gnn_sequences


def prepare_inference_dataframe(X_seq, stock_seq):
    return pd.DataFrame({
        "sequence": [x.tolist() for x in X_seq],
        "stock_id": stock_seq.astype(int)
    })


def run_inference(df, feature_cols, seq_length, model, scaler):
    df = df.sort_values(["stock_id", "Date"]).reset_index(drop=True)

    X_seq, stock_seq, y_seq, dates, prices = create_gnn_sequences(
        df,
        feature_cols,
        seq_length, scaler
    )

    inference_df = prepare_inference_dataframe(X_seq, stock_seq)

    preds = model.predict(inference_df)

    preds = preds.flatten()

    pred_df = pd.DataFrame({
        "Date": dates,
        "stock_id": stock_seq,
        "Close": prices,
        "actual_return": y_seq,
        "predicted_return": preds
    })

    pred_df["signal"] = np.sign(pred_df["predicted_return"])
    pred_df["strategy_return"] = pred_df["signal"] * pred_df["actual_return"]
    pred_df = pred_df.sort_values(["stock_id", "Date"])

    return pred_df


def save_predictions(pred_df, preds_path):
    pred_df.to_csv(preds_path, index=False)


if __name__ == "__main__":
    np.random.seed(42)
    logging.getLogger().setLevel(level=logging.INFO)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    SEQ_LENGTH = 30
    PRED_PATH = 'resources/outputs/outputs/pred_test.csv'
    try:
        model_name = "Stock_Price_GNN_Model"

        model_metadata = client.get_latest_versions(model_name)
        latest_model_version = model_metadata[0].version

        model_uri = client.get_model_version_download_uri(name=model_name, version=latest_model_version)
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)

        scaler = joblib.load('resources/outputs/artifacts/scaler.pkl')
        test_data = pd.read_csv('resources/outputs/outputs/data_complete.csv')
        stock_ids = test_data["stock_id"].astype(int).unique().tolist()
        feature_cols = [col for col in test_data.columns if col not in ("target", "Date", "stock_id")]

        pred_df = run_inference(test_data, feature_cols, SEQ_LENGTH, pyfunc_model, scaler)

        save_predictions(pred_df, PRED_PATH)
    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg
    finally:
        logging.info("INFERENCE DONE")
