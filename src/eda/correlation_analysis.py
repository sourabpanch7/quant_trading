import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CorrelationCalculation:
    def __init__(self, df, img_path, lags=(1, 5, 20, 40, 60, 180), threshold=0.1):
        self.df = df
        self.lags = lags
        self.threshold = threshold
        self.img_path = img_path
        self.lag_corr_matrices = {}
        self.lag_scores = {}
        self.returns_matrix = None
        self.opt_matrix = None
        self.optimal_lag = None
        self.leaders = None
        self.followers = None
        self.pairs_df = pd.DataFrame()

    def create_pivoted_data(self):
        self.df = self.df.sort_values(["stock_id", "Date"])
        self.df = self.df.sort_values(["stock_id", "Date"])
        self.df["returns"] = self.df.groupby("stock_id")["Close"].pct_change()

        self.returns_matrix = self.df.pivot(
            index="Date",
            columns="stock_id",
            values="returns"
        )

    def create_lag_wise_matrix(self):
        for lag in self.lags:
            lag_corr = pd.DataFrame(
                index=self.returns_matrix.columns,
                columns=self.returns_matrix.columns
            )

            for stock_a in self.returns_matrix.columns:
                for stock_b in self.returns_matrix.columns:
                    corr = self.returns_matrix[stock_a].corr(
                        self.returns_matrix[stock_b].shift(-lag)
                    )

                    lag_corr.loc[stock_a, stock_b] = corr

            self.lag_corr_matrices[lag] = lag_corr.astype(float)

    def get_optimal_lag(self):

        for lag, matrix in self.lag_corr_matrices.items():
            # remove self correlations
            np.fill_diagonal(matrix.values, np.nan)

            score = np.nanmean(np.abs(matrix.values))

            self.lag_scores[lag] = score

        self.optimal_lag = max(self.lag_scores, key=self.lag_scores.get)

        logging.info(f"Lag Scores: {self.lag_scores}")
        logging.info(f"Optimal Lag:  {self.optimal_lag}")

        self.opt_matrix = self.lag_corr_matrices[self.optimal_lag]

    def get_leaders(self):
        leader_scores = self.opt_matrix.mean(axis=1)
        self.leaders = leader_scores.sort_values(ascending=False)

    def get_followers(self):
        follower_scores = self.opt_matrix.mean(axis=0)
        self.followers = follower_scores.sort_values(ascending=False)

    def get_strong_leader_followers_pairs(self):
        pairs = []
        for i in self.opt_matrix.index:
            for j in self.opt_matrix.columns:

                if i != j:
                    corr = self.opt_matrix.loc[i, j]

                    if np.abs(corr) > self.threshold:
                        pairs.append((i, j, corr))

        self.pairs_df = pd.DataFrame(pairs, columns=["Leader", "Follower", "LagCorr"])
        logging.info(self.pairs_df)
        self.pairs_df = self.pairs_df.sort_values("LagCorr", ascending=True)

    def plot_lag_score(self):
        lags = list(self.lag_scores.keys())
        scores = list(self.lag_scores.values())

        plt.figure(figsize=(8, 5))
        plt.plot(lags, scores, marker="o")

        plt.xlabel("Lag (Days)")
        plt.ylabel("Average Absolute Correlation")
        plt.title("Optimal Lag Detection")

        plt.grid(True)
        plt.savefig(f"{self.img_path}/lag_score_visualisation.png", format='png')

    def plot_lead_lag_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.opt_matrix,
            cmap="coolwarm",
            center=0
        )
        plt.title(f"Lead-Lag Correlation Matrix (Lag={self.optimal_lag})")

        plt.savefig(f"{self.img_path}/lead_score_heatmap.png", format='png')

    def plot_leader_scores(self):
        plt.figure(figsize=(10, 6))
        self.leaders.head(10).plot(kind="bar")
        plt.title("Top Leader Stocks")
        plt.ylabel("Leader Score")

        plt.savefig(f"{self.img_path}/leaders_visualisation.png", format='png')

    def plot_follower_scores(self):
        plt.figure(figsize=(10, 6))
        self.followers.head(10).plot(kind="bar")
        plt.title("Top Follower Stocks")
        plt.ylabel("Follower Score")

        plt.savefig(f"{self.img_path}/followers_visualisation.png", format='png')

    def perform_eda(self):
        self.create_pivoted_data()
        self.create_lag_wise_matrix()
        self.get_optimal_lag()
        self.plot_lag_score()
        self.plot_lead_lag_heatmap()
        self.get_leaders()
        self.plot_leader_scores()
        self.get_followers()
        self.plot_follower_scores()
        self.get_strong_leader_followers_pairs()
