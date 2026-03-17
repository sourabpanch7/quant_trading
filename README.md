# Algorithmic Trading Pipeline with LSTM + GNN

## Overview

This project implements an **end-to-end quantitative trading research pipeline** for a universe of anonymized stocks. The system converts raw market data into a **portfolio trading strategy** through structured stages including:

1. Data cleaning and feature engineering
2. Exploratory data analysis (EDA) and statistical discovery
3. Predictive modeling using **LSTM + Graph Neural Networks (GNN)**
4. Experiment tracking with **MLflow**
5. Backtesting and portfolio performance evaluation
6. Statistical arbitrage discovery

The goal of the system is to **maximize risk-adjusted returns** by learning both **temporal patterns (LSTM)** and **cross-asset relationships (GNN)**.

---


# Key Features

## Data Engineering

The pipeline performs robust preprocessing of raw financial data:

* Missing value handling
* Forward fill of time series gaps
* Outlier detection and removal
* Log return calculation
* Feature normalization

### Feature Engineering

Several financial features are generated:

* Log returns
* Rolling volatility
* Momentum indicators
* Volume z-scores
* Rolling correlations
* Lag features

Optional denoising techniques include:

* **Kalman filtering** for price smoothing
* Noise-reduced return signals

---

# Exploratory Data Analysis

The EDA module investigates market structure using:

### Correlation Analysis

* Rolling correlations
* Lag correlations
* Leader–follower discovery

### Clustering Analysis

* KMeans clustering of stocks
* Hierarchical clustering
* PCA cluster visualization
* Correlation heatmaps

### Visualizations

The following plots are generated:

* Equity curve
* Drawdown curve
* Prediction vs actual returns
* Portfolio return distribution
* Correlation heatmap
* Spread z-score for pairs trading
* Information coefficient (IC)

These analyses help identify **market regimes, co-moving assets, and trading opportunities**.

---

# Predictive Model

The predictive architecture combines **temporal and relational learning**.

## LSTM Component

Captures **temporal dependencies** in financial time series.

Input:

```
sequence_length × num_features
```

Output:

```
latent temporal representation
```

---

## Graph Neural Network Component

Captures **cross-asset dependencies**.

Stocks are represented as nodes in a graph.

Edges represent:

* correlation relationships
* cluster similarity
* statistical influence

Graph structure is encoded via:

```
edge_index
```

The GNN aggregates information across related assets to improve predictions.

---

## Final Model Architecture

```
Input Features
      │
Sequence Creation
      │
     LSTM
      │
Temporal Embedding
      │
      GNN
      │
Cross-Asset Aggregation
      │
Fully Connected Layer
      │
Predicted 1-Day Forward Return
```

---

# Target Variable

The model predicts:

```
1-day forward return
```

```
forward_return_t = (Close_{t+1} − Close_t) / Close_t
```

This allows predictions to be converted into **trading signals**.

---

# Training Pipeline

Training includes:

* Time series cross-validation using **TimeSeriesSplit**
* Early stopping using validation loss
* Experiment tracking with **MLflow**
* Automatic logging of:

  * parameters
  * metrics
  * artifacts
  * models
  * EDA visualizations

### Hardware Support

The training pipeline supports:

* CPU
* CUDA GPUs
* **Apple Metal Performance Shaders (MPS)**

---

# MLflow Integration

MLflow is used to track experiments locally.

The pipeline logs:

### Parameters

```
sequence_length
learning_rate
batch_size
hidden_dim
num_layers
```

### Metrics

```
training_loss
validation_loss
MSE
Sharpe ratio
max drawdown
turnover
```

### Artifacts

```
EDA plots
model weights
edge_index graph
evaluation reports
```

Models are saved as:

```
mlflow pyfunc models
```

which allows standardized inference.

---

# Inference Pipeline

The inference module:

1. Loads test dataset CSV
2. Creates time sequences
3. Loads trained MLflow model
4. Generates predictions
5. Stores predictions in CSV format

Prediction output includes all fields required for evaluation:

```
date
stock_index
Close
actual_return
predicted_return
signal
```

Signals are generated as:

```
signal = sign(predicted_return)
```

---

# Backtesting Engine

The backtest simulates realistic portfolio execution.

### Strategy Logic

* Position size proportional to prediction signal
* Portfolio weights normalized daily
* Transaction cost applied per trade

### Transaction Cost

```
10 bps per trade
```

---

# Evaluation Metrics

The following performance metrics are computed:

### Sharpe Ratio

Risk-adjusted return.

```
Sharpe = mean(return) / std(return)
```

Annualized with 252 trading days.

---

### Maximum Drawdown

Largest peak-to-trough decline.

Indicates downside risk.

---

### Portfolio Turnover

Measures trading frequency.

Higher turnover implies:

* higher transaction costs
* higher model instability

---

### Total Return

Portfolio equity growth over the test period.

---

# Statistical Arbitrage Module

In addition to directional trading, the project explores **relative value strategies**.

Techniques include:

* correlation discovery
* cointegration testing
* spread modeling
* z-score mean reversion signals

Pairs trading opportunities are detected via:

```
high correlation clusters
lag relationships
```

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running Training

```
python train_base_lstm.py
```

This will:

* train the model
* log experiments to MLflow
* save artifacts

---

# Running Inference

```
python infernece_base_lstm.py
```

This will produce:

```
resources/outputs/outputs/pred_test.csv
```

---

# Running Backtest

```
python evaluation.py
```

Outputs include:

* Sharpe ratio
* drawdown statistics
* turnover
* equity curve

---

# Future Improvements

Potential enhancements include:

* Transformer based time series models
* Dynamic lead–lag network discovery
* Reinforcement learning portfolio allocation
* Regime detection models
* Multi-horizon return prediction

---

# Acknowledgements

This project was developed as part of a quantitative research exercise.

**ChatGPT (OpenAI)** was used to assist with:

* code scaffolding
* architectural brainstorming
* documentation and README preparation

All modeling decisions, experiment design, and validation procedures were reviewed and curated by the project author.

---

# License

This project is intended for **educational and research purposes only**.

It should **not be used directly for live trading without additional validation and risk controls**.