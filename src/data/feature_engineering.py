class FeatureEngineer:

    def __init__(self, df):
        self.df = df

    def engineer_momentum_features(self):
        self.df["mom_5"] = self.df["Close"].pct_change(5)
        self.df["mom_10"] = self.df["Close"].pct_change(10)
        self.df["mom_20"] = self.df["Close"].pct_change(20)

    def engineer_trend_features(self):
        self.df["sma_10"] = self.df["Close"].rolling(10).mean()
        self.df["sma_50"] = self.df["Close"].rolling(50).mean()
        self.df["ema_20"] = self.df["Close"].ewm(span=20).mean()

    def engineer_volatality_features(self):
        self.df["vol_20"] = self.df["log_return"].rolling(20).std()
        self.df["vol_60"] = self.df["log_return"].rolling(60).std()

    def engineer_liquidity_features(self):
        self.df["volume_change"] = self.df["Volume"].pct_change()
        self.df["volume_ma20"] = self.df["Volume"].rolling(20).mean()

    def engineer_microstructure_signals_features(self):
        self.df["hl_spread"] = (self.df["High"] - self.df["Low"]) / self.df["Close"]
        self.df["co_return"] = (self.df["Close"] - self.df["Open"]) / self.df["Open"]

    def create_target(self, n_day):
        self.df["target"] = self.df["Close"].shift(-n_day) / self.df["Close"] - 1

    def perform_feature_engineering(self, n_day=5):
        self.engineer_momentum_features()
        self.engineer_liquidity_features()
        self.engineer_volatality_features()
        self.engineer_microstructure_signals_features()
        self.engineer_trend_features()
        self.create_target(n_day=n_day)
