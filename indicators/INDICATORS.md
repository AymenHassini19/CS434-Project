# 📊 Technical Indicators Documentation

## 1. Project Context

This project focuses on predicting the future price of Nvidia (NVDA) stock using 1-hour OHLCV data (Open, High, Low, Close, Volume).

To enhance the dataset and extract meaningful information from raw price data, several technical indicators are computed. These indicators transform historical market data into structured quantitative features.

---

## 2. Overview of Technical Indicators

Technical indicators are mathematical functions derived from historical price and/or volume data.

They are commonly used in financial analysis to:

- Identify market trends  
- Measure price momentum  
- Estimate volatility  
- Evaluate trading activity  

It is important to clarify that technical indicators are **not deterministic predictors** of future price movements.  
They are **descriptive and indicative tools** that reflect past market behavior and highlight specific market conditions.  

They provide insight into the state of the market rather than guaranteed forecasts.

---

## 3. Classification of Indicators

The selected indicators are grouped into four primary categories:

1. Trend Indicators  
2. Momentum Indicators  
3. Volatility Indicators  
4. Volume-Based Indicators  

Each category captures a different structural characteristic of the market.

---

# 4. Trend Indicators

Trend indicators aim to identify the general direction of price movement over time.

## 4.1 Simple Moving Average (SMA)

**Type:** Trend  
**Input Data:** Close price  

The Simple Moving Average (SMA) is the arithmetic mean of closing prices over a fixed number of periods.

**Formula (conceptual):**  
SMA = (Sum of last n closing prices) / n

**Purpose:**

- Smooth short-term price fluctuations  
- Reduce market noise  
- Identify upward or downward trends  

---

## 4.2 Exponential Moving Average (EMA)

**Type:** Trend  
**Input Data:** Close price  

The Exponential Moving Average (EMA) assigns greater weight to recent prices, making it more responsive than SMA.

**Formula (conceptual):**  
EMA_t = alpha × Close_t + (1 − alpha) × EMA_t-1  

Where alpha is the smoothing factor.

**Purpose:**

- Capture short-term trends  
- React faster to recent price changes  

---

## 4.3 Moving Average Convergence Divergence (MACD)

**Type:** Trend + Momentum  
**Input Data:** Close price  

MACD is calculated as the difference between two exponential moving averages:

MACD = EMA(12) − EMA(26)

A signal line (typically EMA(9) of MACD) is used to detect momentum shifts.

**Purpose:**

- Identify changes in trend direction  
- Detect strengthening or weakening momentum  

---

## 4.4 Average Directional Index (ADX)

**Type:** Trend Strength  
**Input Data:** High, Low, Close  

The Average Directional Index (ADX) measures the strength of a trend without indicating its direction.

**Purpose:**

- Distinguish trending markets from ranging markets  
- Higher values indicate stronger trends  

---

# 5. Momentum Indicators

Momentum indicators measure the speed and magnitude of price movements.

## 5.1 Relative Strength Index (RSI)

**Type:** Momentum  
**Input Data:** Close price  

RSI evaluates the ratio between recent gains and recent losses over a defined period.

**Formula (conceptual):**  
RSI = 100 − (100 / (1 + RS))  
Where RS = Average Gain / Average Loss

**Interpretation:**

- RSI > 70 → Potential overbought condition  
- RSI < 30 → Potential oversold condition  

**Purpose:**

- Measure strength of price movement  
- Identify possible exhaustion of trends  

---

## 5.2 Rate of Change (ROC)

**Type:** Momentum  
**Input Data:** Close price  

The Rate of Change (ROC) measures the percentage change in price over a specified number of periods.

**Formula (conceptual):**  
ROC = (Close_t − Close_t-n) / Close_t-n

**Purpose:**

- Measure acceleration or deceleration in price  
- Identify increasing or decreasing momentum  

---

# 6. Volatility Indicators

Volatility indicators measure the degree of price fluctuation over time.

## 6.1 Bollinger Bands

**Type:** Volatility  
**Input Data:** Close price  

Bollinger Bands consist of three components:

- Middle Band: Simple Moving Average  
- Upper Band: SMA + (k × Standard Deviation)  
- Lower Band: SMA − (k × Standard Deviation)

**Purpose:**

- Identify periods of high or low volatility  
- Detect price expansion and contraction phases  

---

## 6.2 Average True Range (ATR)

**Type:** Volatility  
**Input Data:** High, Low, Close  

The Average True Range (ATR) measures the average trading range over a specified period.

**Purpose:**

- Quantify market volatility  
- Identify periods of increased price movement  

---

# 7. Volume-Based Indicators

Volume-based indicators incorporate trading volume to assess the strength of market participation.

## 7.1 On-Balance Volume (OBV)

**Type:** Volume Momentum  
**Input Data:** Close price, Volume  

OBV accumulates volume based on price direction:

- If the closing price increases, volume is added  
- If the closing price decreases, volume is subtracted  

**Purpose:**

- Detect accumulation or distribution  
- Confirm price trends  

---

## 7.2 Volume Weighted Average Price (VWAP)

**Type:** Volume + Price  
**Input Data:** Price, Volume  

VWAP is calculated as:

VWAP = Sum(Price × Volume) / Sum(Volume)

**Purpose:**

- Measure the average traded price weighted by volume  
- Reflect the relationship between price and trading activity  

---

# 8. Summary

The indicators described above transform raw OHLCV data into structured quantitative representations of:

- Market direction  
- Momentum strength  
- Volatility conditions  
- Trading participation  

They are not predictive guarantees but rather analytical tools that provide indicative insights into market behavior.  

These indicators serve as structured features for further quantitative analysis and modeling.