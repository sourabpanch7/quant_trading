import mlflow
import mlflow.pytorch
import torch.optim as optim
import torch.nn as nn


def train_model(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "Adam")

        for epoch in range(epochs):

            total_loss = 0

            for X, y in dataloader:
                optimizer.zero_grad()

                pred = model(X).squeeze()

                loss = criterion(pred, y)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        mlflow.pytorch.log_model(model, "model")
