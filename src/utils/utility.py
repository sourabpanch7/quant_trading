import mlflow


def create_or_set_experiment(exp_name="quant_trading_lstm"):
    mlflow.set_experiment(exp_name)
