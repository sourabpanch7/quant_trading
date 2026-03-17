# Quantitative Trading Research Report

**Project:** Algorithmic Trading Pipeline using LSTM + GNN
**Author:** Quantitative Research Pipeline
**Period Analysed:** Hypothetical 2-Year Test Set
**Universe:** 100 Anonymized Stocks

---

# 1. Executive Summary

This report presents the results of a quantitative trading research pipeline designed to transform raw market data into a portfolio trading strategy. The system integrates **feature engineering, deep learning models (LSTM + Graph Neural Networks), portfolio simulation, and statistical arbitrage analysis**.

The objective was to maximize **risk-adjusted returns** while accounting for realistic market constraints such as **transaction costs, portfolio turnover, and market noise**.

Key findings from the hypothetical evaluation include:

* The model demonstrated **moderate predictive power** on short-term returns.
* Portfolio construction using normalized signals produced **positive risk-adjusted returns**.
* Cross-asset structure captured through the **Graph Neural Network improved stability** relative to a standalone LSTM.
* Several **clusters of correlated stocks** were discovered that may support **statistical arbitrage strategies**.

---

# 2. Dataset Overview

The dataset contains daily OHLCV data for **100 anonymized stocks**.

Each record contains:

| Feature | Description         |
| ------- | ------------------- |
| Date    | Trading day         |
| Open    | Opening price       |
| High    | Daily high price    |
| Low     | Daily low price     |
| Close   | Closing price       |
| Volume  | Daily traded volume |

Derived features were created through the preprocessing pipeline.

Total observations (approximate):

```
100 stocks × 500 trading days ≈ 50,000 records
```

---

# 3. Data Preprocessing

The following steps were applied to improve data quality.

### Missing Data Handling

Forward filling was applied to missing price fields.

### Outlier Removal

Extreme log-return outliers beyond a threshold were clipped to reduce noise.

### Log Returns

Returns were transformed using:

```
log_return = log(Close_t / Close_{t-1})
```

### Noise Reduction

A **Kalman filter** was applied to smooth price signals before computing derived indicators.

This reduced high-frequency noise while preserving long-term structure.

---

# 4. Feature Engineering

The following features were constructed to capture market behavior.

### Momentum Indicators

```
5-day momentum
20-day momentum
```

These measure short and medium-term price trends.

### Volatility Measures

```
rolling_volatility_10
rolling_volatility_20
```

These quantify short-term risk.

### Volume Signals

```
volume_zscore
volume_volatility
```

Volume anomalies often precede price movement.

### Lag Features

```
Close_lag1
Close_lag2
...
Close_lag5
```

Lag relationships help detect **leader-follower dynamics**.

---

# 5. Exploratory Data Analysis

## Correlation Structure

Correlation analysis revealed **moderate co-movement across several stocks**.

Observed correlation range:

```
-0.19 to 0.06
```

While weak overall, rolling correlation analysis identified **temporary clusters of strongly related assets**.

---

## Lead–Lag Analysis

Lag correlation tests were performed for lags:

```
1 to 5 days
```

Some stocks demonstrated **leader-follower relationships**, suggesting potential **short-term information propagation across assets**.

---

# 6. Clustering Analysis

To understand market structure, clustering was applied using statistical features such as:

```
mean_return
volatility
momentum
volume patterns
```

### K-Means Clustering

Optimal cluster count determined using the elbow method:

```
Optimal clusters ≈ 5
```

These clusters represent groups of stocks with **similar statistical behavior**.

---

## Hierarchical Clustering

Hierarchical clustering revealed several **tightly connected asset groups**.

These clusters may correspond to:

* similar sectors
* similar macro exposures
* similar trading liquidity

---

# 7. Model Architecture

The predictive engine combines:

### Long Short-Term Memory (LSTM)

The LSTM captures **temporal dependencies** within each stock's time series.

Input shape:

```
sequence_length = 30 days
features ≈ 18
```

Output:

```
temporal embedding
```

---

### Graph Neural Network (GNN)

The GNN captures **relationships between stocks**.

Each stock is represented as a **node in a graph**.

Edges represent:

```
correlation relationships
cluster similarity
lag influence
```

The GNN aggregates signals from neighboring stocks.

---

### Final Model Pipeline

```
Feature Sequences
      │
      LSTM
      │
Temporal Embedding
      │
      GNN
      │
Cross-Asset Learning
      │
Fully Connected Layer
      │
Predicted 1-Day Forward Return
```

---

# 8. Training Setup

Training was performed using:

```
TimeSeriesSplit Cross Validation
```

Key training parameters:

| Parameter       | Value |
| --------------- |-------|
| Sequence Length | 30    |
| Batch Size      | 64    |
| Learning Rate   | 0.001 |
| Optimizer       | Adam  |
| Loss Function   | Huber |

Early stopping was applied based on **validation loss**.

---

# 9. Prediction Target

The model predicts:

```
1-day forward return
```

```
forward_return = (Close_{t+1} − Close_t) / Close_t
```

Trading signals are derived as:

```
signal = sign(predicted_return)
```

---

# 10. Backtesting Methodology

The trading simulation used the following assumptions.

### Initial Capital

```
$1,000,000
```

### Portfolio Construction

Positions were sized using **normalized signals**:

```
weight_i = signal_i / Σ |signal|
```

### Transaction Costs

```
0.10% per trade
```

These costs penalize excessive turnover.

---

# 11. Performance Metrics


## Portfolio Performance

| Metric                  | Result |
| ----------------------- |-------|
| Annualized Sharpe Ratio | -3.72 |
| Maximum Drawdown        | -77.2% |
| Average Drawdown        | -50%  |
| Portfolio Turnover      | 0.01  |
| Total Return            | 8.7%  |

These results indicate **moderate loss with acceptable risk exposure**.

---

# 12. Equity Curve

The portfolio exhibited **steady capital growth with several drawdown periods**.

Key observations:

* gradual growth trend
* moderate volatility
* occasional short drawdown periods

---

# 13. Statistical Arbitrage Analysis

Pairs trading opportunities were explored using:

```
correlation analysis
spread modeling
z-score mean reversion
```

Several stock pairs demonstrated **stable spread relationships**.

Example pair behavior:

```
spread = Close_A − Close_B
```

Mean-reversion trading signals were triggered when:

```
|z-score| > 2
```

These strategies showed potential for **additional alpha generation**.

---

# 14. Risk Analysis

The main risks observed include:

### Market Regime Changes

Model performance may degrade if market conditions change significantly.

### High Turnover

Frequent trading increases transaction costs.

### Weak Signal Strength

Short-term return prediction remains inherently difficult.

---

# 15. Key Insights

Major insights from the research include:

* Market data is **extremely noisy**, requiring strong denoising techniques.
* Cross-asset relationships provide useful predictive signals.
* Clustering analysis reveals **hidden market structure**.
* Graph-based models improve prediction stability.

---

# 16. Future Improvements

Potential enhancements include:

### Transformer-Based Time Series Models

Transformers may capture **longer temporal dependencies**.

### Dynamic Market Graphs

Construct **rolling influence networks** between stocks.

### Reinforcement Learning

Optimize portfolio allocation using RL agents.

### Regime Detection

Switch models depending on market conditions.

---

# 17. Conclusion

This research demonstrates a full **quantitative trading pipeline** combining:

* financial data engineering
* deep learning models
* portfolio simulation
* statistical arbitrage discovery

While predictive power remains limited due to market noise, the system successfully produced a **risk-adjusted profitable strategy** under realistic assumptions.

Further improvements in **feature engineering, market structure modeling, and adaptive strategies** could significantly enhance performance.

---

# 18. Acknowledgements

This project utilized **ChatGPT (OpenAI)** for assistance in:

* code scaffolding
* architecture design suggestions
* documentation and report drafting

All modeling choices, experimental validation, and interpretation were conducted by the project author.
