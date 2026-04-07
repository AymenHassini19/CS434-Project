
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas_market_calendars as mcal

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# 1) DATA LOADING
# ============================================================

def load_nvda_data(file_path: str) -> pd.DataFrame:
    """
    Load the user's hourly NVDA CSV.

    Expected file format:
        Datetime, Close, High, Low, Open, Volume
    with 3 header rows to skip.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df = pd.read_csv(file_path, skiprows=3, header=None)
    df.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values("Datetime").set_index("Datetime")

    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    return df


# ============================================================
# 2) FEATURE ENGINEERING
# ============================================================

def add_moving_averages(df, windows=(20, 50)):
    """
    Kept for backward compatibility with the UI. These columns are
    also recomputed inside create_features().
    """
    df = df.copy()
    for window in windows:
        df[f"{window}_MA"] = df["Close"].rolling(window=window).mean()
    return df


def add_RSI(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_MACD(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features using only information available up to time t.
    Target is the next-hour log return, which is generally much more
    stable than predicting raw price levels.
    """
    df = df.copy()

    # Base trend / momentum features
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["EMA_spread"] = df["EMA12"] - df["EMA26"]
    df["price_vs_ma20"] = df["Close"] / df["MA20"] - 1.0
    df["price_vs_ma50"] = df["Close"] / df["MA50"] - 1.0

    # Returns and lags
    df["log_close"] = np.log(df["Close"].replace(0, np.nan))
    df["log_ret_1"] = df["log_close"].diff(1)
    df["log_ret_3"] = df["log_close"].diff(3)
    df["log_ret_6"] = df["log_close"].diff(6)
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_6"] = df["Close"].pct_change(6)

    for lag in (1, 2, 3, 6, 12):
        df[f"close_lag{lag}"] = df["Close"].shift(lag)
        df[f"vol_lag{lag}"] = df["Volume"].shift(lag)
        df[f"ret_lag{lag}"] = df["ret_1"].shift(lag)
        df[f"range_lag{lag}"] = ((df["High"] - df["Low"]) / df["Close"]).shift(lag)

    # Rolling volatility / range
    df["volatility_12"] = df["ret_1"].rolling(12).std()
    df["volatility_24"] = df["ret_1"].rolling(24).std()
    df["range_12"] = ((df["High"] - df["Low"]) / df["Close"]).rolling(12).mean()
    df["range_24"] = ((df["High"] - df["Low"]) / df["Close"]).rolling(24).mean()

    # Volume features
    df["volume_chg_1"] = df["Volume"].pct_change(1)
    df["volume_chg_6"] = df["Volume"].pct_change(6)
    df["volume_ma12"] = df["Volume"].rolling(12).mean()
    df["volume_ma24"] = df["Volume"].rolling(24).mean()
    df["volume_ratio_12"] = df["Volume"] / df["volume_ma12"]
    df["volume_ratio_24"] = df["Volume"] / df["volume_ma24"]

    # Technical indicators
    df = add_RSI(df, period=14)
    df = add_MACD(df)

    # Time-of-day structure for intraday data
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    # Target: next-hour log return
    df["target_log_return"] = np.log(df["Close"].shift(-1) / df["Close"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


# Backward-compatible feature list for downstream code / UI.
feature_cols = [
    "Close", "High", "Low", "Open", "Volume",
    "MA20", "MA50", "EMA12", "EMA26", "EMA_spread",
    "price_vs_ma20", "price_vs_ma50",
    "log_ret_1", "log_ret_3", "log_ret_6",
    "ret_1", "ret_3", "ret_6",
    "close_lag1", "close_lag2", "close_lag3", "close_lag6", "close_lag12",
    "vol_lag1", "vol_lag2", "vol_lag3", "vol_lag6", "vol_lag12",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag6", "ret_lag12",
    "range_lag1", "range_lag2", "range_lag3", "range_lag6", "range_lag12",
    "volatility_12", "volatility_24",
    "range_12", "range_24",
    "volume_chg_1", "volume_chg_6",
    "volume_ma12", "volume_ma24",
    "volume_ratio_12", "volume_ratio_24",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "hour", "dayofweek"
]


# ============================================================
# 3) CHARTING
# ============================================================

def plot_and_save_chart(df, save_path="stock_analysis.html"):
    vol_df = df["Volume"].resample("5D").sum()

    daily_dir = (
        df["Close"].resample("5D").last() >= df["Open"].resample("5D").first()
    )

    vol_colors = [
        "rgba(0,255,0,0.7)" if up else "rgba(255,0,0,0.7)"
        for up in daily_dir
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Price & Trends", "Volume"),
        row_width=[0.2, 0.8]
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="NVDA"
        ),
        row=1, col=1
    )

    for ma in [c for c in df.columns if c.endswith("_MA") or c in ("MA20", "MA50")]:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[ma], line=dict(width=2), name=ma),
                row=1, col=1
            )

    fig.add_trace(
        go.Bar(
            x=vol_df.index,
            y=vol_df.values,
            marker=dict(color=vol_colors),
            name="Volume"
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="NVDA Hourly Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark"
    )

    fig.write_html(save_path, include_plotlyjs=True, full_html=True)
    fig.show()


# ============================================================
# 4) TRADING-HOUR GENERATOR
# ============================================================

def get_future_trading_hours(start_time, n_hours):
    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=start_time.date(),
        end_date=(start_time + timedelta(days=14)).date()
    )

    trading_hours = []
    for _, row in schedule.iterrows():
        hours = pd.date_range(
            start=row["market_open"],
            end=row["market_close"],
            freq="1h",
            tz="UTC"
        )
        trading_hours.extend(hours)

    trading_hours = [t for t in trading_hours if t > start_time]
    return trading_hours[:n_hours]


# ============================================================
# 5) MODEL TRAINING / FORECASTING
# ============================================================

def split_data_timewise(df, target_col="target_log_return", train_frac=0.8):
    split_i = int(len(df) * train_frac)
    train = df.iloc[:split_i]
    test = df.iloc[split_i:]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    A stronger default than RandomForest for this problem:
    - trained on lagged time-series features
    - predicts next-hour log return
    - handles non-linear interactions well
    """
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=600,
        max_depth=5,
        min_samples_leaf=25,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def forecast_future_close(df, model, future_datetimes, feature_columns):
    """
    Recursive forecast:
    - predict next-hour log return
    - convert to next-hour close
    - append the synthetic row
    - repeat for the remaining horizon
    """
    df_fc = df.copy()
    forecasts = []

    for dt in future_datetimes:
        df_feats = create_features(df_fc)

        last_row = df_feats.iloc[-1]
        cols = [c for c in feature_columns if c in last_row.index]
        X_last = pd.DataFrame([last_row[cols].values], columns=cols, index=[pd.to_datetime(dt)])

        pred_log_return = float(model.predict(X_last)[0])
        last_close = float(df_fc["Close"].iloc[-1])
        pred_close = last_close * np.exp(pred_log_return)

        new_idx = pd.to_datetime(dt)
        new_row = df_fc.iloc[-1].copy()
        new_row["Open"] = last_close
        new_row["High"] = max(last_close, pred_close)
        new_row["Low"] = min(last_close, pred_close)
        new_row["Close"] = pred_close
        new_row["Volume"] = float(df_fc["Volume"].iloc[-1])

        new_df = pd.DataFrame([new_row.values], columns=new_row.index, index=[new_idx])
        df_fc = pd.concat([df_fc, new_df]).sort_index()

        forecasts.append({"Datetime": new_idx, "PredictedClose": pred_close})

    return pd.DataFrame(forecasts).set_index("Datetime")


def run_ml_pipeline_and_forecast(df, hours_to_predict=24):
    print("\n================ ML PIPELINE ================\n")

    # 1) Feature engineering
    df_feat = create_features(df)

    # Use the most recent market regime
    df_feat = df_feat.iloc[-2000:]

    print(f"Total samples used: {len(df_feat)}")

    # 2) Time-wise split
    X_train, X_test, y_train, y_test = split_data_timewise(df_feat)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # 3) Align features
    cols = [c for c in feature_cols if c in X_train.columns]
    X_train = X_train[cols]
    X_test = X_test[cols]

    print(f"\nNumber of features: {len(cols)}")

    # 4) Train model
    print("\nTraining HistGradientBoostingRegressor model...")
    model = train_model(X_train, y_train)

    # 5) Evaluation on next-hour log return
    preds_test_logret = model.predict(X_test)
    mse = mean_squared_error(y_test, preds_test_logret)
    rmse = np.sqrt(mse)

    # Convert return forecasts back to prices for a fair comparison
    actual_next_close = X_test["Close"].values * np.exp(y_test.values)
    predicted_next_close = X_test["Close"].values * np.exp(preds_test_logret)

    naive_next_close = X_test["Close"].values

    price_rmse = np.sqrt(mean_squared_error(actual_next_close, predicted_next_close))
    naive_rmse = np.sqrt(mean_squared_error(actual_next_close, naive_next_close))

    print("\n================ EVALUATION ================\n")
    print(f"Return MSE       : {mse:.6f}")
    print(f"Return RMSE      : {rmse:.6f}")
    print(f"Price RMSE       : {price_rmse:.4f} $")
    print(f"Naive RMSE       : {naive_rmse:.4f} $")
    print(f"Improvement vs naive: {naive_rmse - price_rmse:.4f} $")

    # 6) Feature importance proxy
    print("\n=========== TOP FEATURE IMPORTANCE ==========\n")
    try:
        # HistGradientBoostingRegressor does not expose feature_importances_.
        # This simple permutation-style proxy uses absolute correlation with target on train.
        corr = X_train.copy()
        corr["target"] = y_train.values
        feat_imp = (
            corr.corr(numeric_only=True)["target"]
            .drop("target")
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )
        for name, val in feat_imp.items():
            print(f"{name:15s} -> {val:.4f}")
    except Exception:
        print("Feature importance not available.")

    # 7) Forecast future
    print("\n=============== FORECASTING ================\n")

    future_datetimes = get_future_trading_hours(df.index[-1], hours_to_predict)
    forecast_df = forecast_future_close(df, model, future_datetimes, cols)

    print("Forecast completed.\n")
    return model, forecast_df


# ============================================================
# 6) PLOT REAL VS PREDICTED
# ============================================================

def plot_forecast_close(forecast_df):
    """
    Accepts a DataFrame with either:
    - Datetime index + PredictedClose
    - Datetime column + Close or PredictedClose
    """
    forecast_df = forecast_df.copy()

    if "Datetime" in forecast_df.columns:
        forecast_df["Datetime"] = pd.to_datetime(forecast_df["Datetime"])
        forecast_df = forecast_df.set_index("Datetime")

    if "PredictedClose" in forecast_df.columns:
        pred_col = "PredictedClose"
    elif "Close" in forecast_df.columns:
        pred_col = "Close"
    else:
        raise ValueError("forecast_df must contain either 'PredictedClose' or 'Close'.")

    start = forecast_df.index.min()
    end = forecast_df.index.max() + timedelta(hours=1)

    real_df = yf.download(
        "NVDA",
        start=start,
        end=end,
        interval="1h",
        auto_adjust=False,
        progress=False
    ).dropna()

    if real_df.empty:
        print("No real market data available for the forecast window.")
        return

    real_close = real_df["Close"]
    common_idx = forecast_df.index.intersection(real_close.index)

    if len(common_idx) == 0:
        print("No overlapping timestamps found between forecast and real data.")
        return

    forecast_df = forecast_df.loc[common_idx]
    real_close = real_close.loc[common_idx]

    plt.figure(figsize=(13, 6))
    plt.plot(real_close.index, real_close.values, label="Real Close (yfinance)", marker="o")
    plt.plot(forecast_df.index, forecast_df[pred_col], label="Predicted Close", linestyle="--", marker="x")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Datetime (Trading Hours)")
    plt.ylabel("Close Price")
    plt.title("NVDA Trading-Hour Close: Prediction vs Reality")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7) MAIN
# ============================================================

def main():
    file_path = "NVDA_hourly_last_2_years.csv"

    # Load data
    df = load_nvda_data(file_path)

    # Chart is optional
    df = add_moving_averages(df)
    plot_and_save_chart(df)

    # Ask user how many future trading hours to predict
    try:
        hours_to_predict = int(input("Enter trading hours to forecast: "))
        if hours_to_predict <= 0:
            raise ValueError
    except Exception:
        print("Invalid input → defaulting to 24 trading hours.")
        hours_to_predict = 24

    # Train + forecast
    model, forecast_df = run_ml_pipeline_and_forecast(df, hours_to_predict)

    print("\nForecasted Close prices:")
    print(forecast_df)

    # Plot comparison
    forecast_plot_df = forecast_df.reset_index()
    plot_forecast_close(forecast_plot_df)


if __name__ == "__main__":
    main()
