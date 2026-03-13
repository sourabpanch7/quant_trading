import logging
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd


def create_label_sequnce(y_test, seq_length):
    y_test_sequences = []

    for i in range(len(y_test) - seq_length):
        y_test_sequences.append(y_test[i + seq_length])
    return np.array(y_test_sequences)


def create_sequences(X, seq_length):
    sequences = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
    return np.array(sequences)


def prepare_pyfunc_input(X_seq):
    n_samples, seq_len, n_features = X_seq.shape

    flattened = X_seq.reshape(n_samples, seq_len * n_features)

    columns = [
        f"f_{t}_{f}"
        for t in range(seq_len)
        for f in range(n_features)
    ]

    return pd.DataFrame(flattened, columns=columns)


def run_inference(X_test, model_uri, seq_length=30):
    model = mlflow.pyfunc.load_model(model_uri)

    # Create sequences
    X_seq = create_sequences(X_test, seq_length)

    # Convert to dataframe for pyfunc
    input_df = prepare_pyfunc_input(X_seq)

    # print(input_df)
    # Predict
    preds = model.predict(input_df)

    return np.array(preds)


def create_evaluation_dataframe(predictions, y_test_sequences):
    predictions = predictions.flatten()

    preds = np.squeeze(predictions)
    y_test_seq = np.squeeze(y_test_sequences)

    min_len = min(len(preds), len(y_test_seq))

    preds = preds[:min_len]
    y_test_seq = y_test_seq[:min_len]

    pred_df = pd.DataFrame({
        "true_return": y_test_seq,
        "pred_return": preds
    })

    pred_df["signal"] = 0

    pred_df.loc[pred_df["pred_return"] > 0, "signal"] = 1
    pred_df.loc[pred_df["pred_return"] < 0, "signal"] = -1

    pred_df["strategy_return"] = pred_df["signal"] * pred_df["true_return"]
    pred_df["equity_curve"] = (
            1 + pred_df["strategy_return"]
    ).cumprod()
    pred_df["position_change"] = pred_df["signal"].diff().abs()
    return pred_df


if __name__ == "__main__":
    logging.getLogger().setLevel(level=logging.INFO)
    try:
        X_test = np.load("resources/outputs/datasets/X_test.npy")
        y_test = np.load("resources/outputs/datasets/y_test.npy")
        seq_length = 30

        y_test_sequences = create_label_sequnce(y_test, seq_length)

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        client = MlflowClient()

        model_name = "Stock_Price_Model"
        model_version = "latest"

        model_uri = client.get_model_version_download_uri(name=model_name, version='3')
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)

        predictions = run_inference(
            X_test,
            model_uri=model_uri,
            seq_length=seq_length
        )

        pred_df = create_evaluation_dataframe(predictions, y_test_sequences)
        pred_df.to_csv('resources/outputs/outputs/y_pred_test.csv', index=False)

    except Exception as err_msg:
        logging.error(str(err_msg))
        raise err_msg

    finally:
        logging.info("INFERENCE DONE")
