import pandas as pd
import numpy as np


class DataCleaner:

    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.df = self.df.sort_values("Date")

        # Forward fill missing values
        self.df = self.df.fillna(method="ffill")

        # Remove extreme outliers
        z = np.abs((self.df["Close"] - self.df["Close"].mean()) / self.df["Close"].std())
        self.df = self.df[z < 4]

        # Log returns
        self.df["log_return"] = np.log(self.df["Close"] / self.df["Close"].shift(1))
