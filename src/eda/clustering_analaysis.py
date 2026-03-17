import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram


class ClusteringCalculation:
    def __init__(self, df, img_path, artifact_path):
        self.df = df
        self.img_path = img_path
        self.artifact_path = artifact_path
        self.feature_df = None
        self.X = None
        self.scaler = None
        self.optimal_k = 0
        self.silhouette_score = 0
        self.kmeans = None

    def build_stock_features(self):
        stock_features = []
        for stock, g in self.df.groupby('stock_id'):
            features = {}

            features["stock_id"] = stock
            features["mean_return"] = g["log_return"].mean()
            features["volatility"] = g["log_return"].std()
            features["skewness"] = g["log_return"].skew()
            features["kurtosis"] = g["log_return"].kurtosis()

            features["avg_volume"] = g["Volume"].mean()
            features["volume_volatility"] = g["Volume"].std()

            features["momentum_5"] = g["Close"].pct_change(5).mean()
            features["momentum_20"] = g["Close"].pct_change(20).mean()
            stock_features.append(features)

        self.feature_df = pd.DataFrame(stock_features)

    def scale_features(self):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.feature_df.drop(columns=["stock_id"]))


    def plot_elbow_silhouette(self):
        inertia = []
        silhouette_scores = []
        K = range(2, 15)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X)
            inertia.append(kmeans.inertia_)
            score = silhouette_score(self.X, kmeans.labels_)
            silhouette_scores.append(score)
            if score > self.silhouette_score:
                self.silhouette_score = score
                self.optimal_k = k
                self.kmeans = kmeans

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method for Optimal K')

        plt.subplot(1, 2, 2)
        plt.plot(K, silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Average Silhouette Score')
        plt.title('Silhouette Method for Optimal K')

        plt.tight_layout()
        plt.savefig(f'{self.img_path}/clustering_analysis.png', format='png')
        joblib.dump(self.kmeans, f'{self.artifact_path}/kmeans.pkl')

    def predict_cluster(self):
        labels = self.kmeans.predict(self.X)
        self.feature_df["cluster"] = labels

    def plot_cluster_pca(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=self.feature_df["cluster"],
            cmap="CMRmap"
        )

        plt.title("Stock Clusters (PCA Projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(f'{self.img_path}/PCA.png', format='png')

    def plot_dendrogram(self):
        linked = linkage(self.X, method="ward")
        plt.figure(figsize=(12, 6))
        dendrogram(linked, labels=self.feature_df["stock_id"].values)

        plt.title("Hierarchical Clustering Dendrogram")

        plt.xlabel("Stock")
        plt.ylabel("Distance")
        plt.savefig(f'{self.img_path}/hierarchical_clustering.png', format='png')

    def plot_correlation_clusters(self):
        pivot = self.df.pivot_table(index="Date", columns="stock_id", values="log_return")
        corr = pivot.corr()
        sns.clustermap(corr, cmap="coolwarm", figsize=(12, 12))
        plt.title("Stock Correlation Clustering")
        plt.savefig(f'{self.img_path}/correlation_clustering.png', format='png')

    def plot_cluster_returns(self):
        df_cluster = self.df.merge(
            self.feature_df[["stock_id", "cluster"]]
            , on="stock_id"
        )

        cluster_returns = df_cluster.groupby(
            ["Date", "cluster"]
        )["log_return"].mean().unstack()

        cluster_returns.cumsum().plot(figsize=(10, 6))

        plt.title("Cumulative Returns by Cluster")
        plt.savefig(f'{self.img_path}/cluster_returns.png', format='png')
        df_cluster['target'] = df_cluster.pop('target')
        self.df = df_cluster

    def perform_clustering_analysis(self):
        self.build_stock_features()
        self.scale_features()
        self.plot_elbow_silhouette()
        self.predict_cluster()
        self.plot_cluster_pca()
        self.plot_dendrogram()
        self.plot_cluster_returns()
        self.plot_correlation_clusters()
