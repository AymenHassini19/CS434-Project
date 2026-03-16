import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import pandas_market_calendars as mcal
import os

# ----------------------------
# 1. DATA COLLECTION
# ----------------------------
def load_nvda_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df = pd.read_csv(file_path, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True, errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df.set_index('Datetime', inplace=True)
    df = df.sort_index()

    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    return df

# ----------------------------
# 2. FEATURE ENGINEERING
# ----------------------------
def add_moving_averages(df, windows=[20, 50]):
    for window in windows:
        df[f'{window}_MA'] = df['Close'].rolling(window=window).mean()
    return df

# RSI and MACD from earlier + full feature creator
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

def create_features(df):
    df = df.copy()
    # basic MAs (you already have but ensure consistent names)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # lags of Close and Volume
    for lag in (1,2,3,6,12):
        df[f"close_lag{lag}"] = df["Close"].shift(lag)
        df[f"vol_lag{lag}"] = df["Volume"].shift(lag)

    # returns
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_6"] = df["Close"].pct_change(6)

    # rolling volatility (std of returns)
    df["volatility_12"] = df["ret_1"].rolling(12).std()

    # indicators
    df = add_RSI(df, period=14)
    df = add_MACD(df)

    # target: next-step Close
    df["target"] = df["Close"].shift(-1)

    df = df.dropna()
    return df

# ----------------------------
# 3. INTERACTIVE CANDLE CHART
# ----------------------------
def plot_and_save_chart(df, save_path="stock_analysis.html"):

    
    vol_df = df["Volume"].resample("5D").sum()

    
    daily_dir = (
        df["Close"].resample("5D").last()
        >= df["Open"].resample("5D").first() 
    )

    vol_colors = [
        "rgba(0,255,0,0.7)" if up else "rgba(255,0,0,0.7)"
        for up in daily_dir
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Price & Trends", "Volume"),
        row_width=[0.2, 0.8]
    )

    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="NVDA"
    ), row=1, col=1)

    
    for ma in [c for c in df.columns if "MA" in c]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[ma],
            line=dict(width=2),
            name=ma
        ), row=1, col=1)

    
    fig.add_trace(go.Bar(
        x=vol_df.index,
        y=vol_df.values,
        marker=dict(color=vol_colors),
        name="Daily Volume"
    ), row=2, col=1)

    fig.update_layout(
        title="NVDA Hourly Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        template="plotly_dark"
    )

    fig.write_html(save_path)
    fig.show()

# ----------------------------
# 4. TRADING-HOUR GENERATOR
# ----------------------------
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

# ----------------------------
# 5. PREDICTION FUNCTION
# ----------------------------

def split_data_timewise(df, target_col="target", train_frac=0.8):
    split_i = int(len(df) * train_frac)
    train = df.iloc[:split_i]
    test = df.iloc[split_i:]
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    return X_train, X_test, y_train, y_test

def train_rf(X_train, y_train):
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def forecast_future_close(df, model, future_datetimes, feature_columns):
    """
    Iterative/recursive forecasting that preserves column names and datetime index.
    """
    df_fc = df.copy()
    forecasts = []

    for dt in future_datetimes:
        # 1) recompute features on current history
        df_feats = create_features(df_fc)

        # 2) pick the last feature row (DataFrame) and build a DataFrame with the same column names
        last_row = df_feats.iloc[-1]
        # ensure feature_columns exist in last_row
        cols = [c for c in feature_columns if c in last_row.index]
        X_last_df = pd.DataFrame([last_row[cols].values], columns=cols, index=[pd.to_datetime(dt)])

        # 3) predict using a DataFrame (keeps feature names aligned)
        pred_close = float(model.predict(X_last_df)[0])

        # 4) create a new row (Series) with the predicted Close and sensible OHLC/Volume
        new_row = df_fc.iloc[-1].copy()
        new_idx = pd.to_datetime(dt)
        new_row["Open"] = df_fc["Close"].iloc[-1]
        new_row["High"] = max(df_fc["Close"].iloc[-1], pred_close)
        new_row["Low"] = min(df_fc["Close"].iloc[-1], pred_close)
        new_row["Close"] = pred_close
        new_row["Volume"] = df_fc["Volume"].iloc[-1]

        # 5) append preserving the datetime index
        new_df = pd.DataFrame([new_row.values], columns=new_row.index, index=[new_idx])
        df_fc = pd.concat([df_fc, new_df])
        df_fc = df_fc.sort_index()

        forecasts.append({"Datetime": new_idx, "PredictedClose": pred_close})

    return pd.DataFrame(forecasts).set_index("Datetime")



def get_predictions_all_columns(df, hours):
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    predictions = pd.DataFrame()

    df_p = df[features].copy()
    df_p['Index'] = np.arange(len(df_p))
    last_idx = df_p['Index'].iloc[-1]

    future_dates = get_future_trading_hours(df_p.index[-1], hours)
    predictions['Datetime'] = future_dates

    for col in features:
        model = LinearRegression()
        model.fit(df_p[['Index']], df_p[col])

        future_idx = np.arange(
            last_idx + 1,
            last_idx + 1 + len(future_dates)
        ).reshape(-1, 1)

        predictions[col] = model.predict(future_idx)

    return predictions

def run_ml_pipeline_and_forecast(df, hours_to_predict=24):

    print("\n================ ML PIPELINE ================\n")

    # 1️⃣ Feature engineering
    df_feat = create_features(df)

    # use recent market regime
    df_feat = df_feat.iloc[-2000:]

    print(f"Total samples used: {len(df_feat)}")

    # 2️⃣ Time-series split
    X_train, X_test, y_train, y_test = split_data_timewise(df_feat)

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # 3️⃣ Align features
    global feature_cols
    cols = [c for c in feature_cols if c in X_train.columns]

    X_train = X_train[cols]
    X_test = X_test[cols]

    print(f"\nNumber of features: {len(cols)}")

    # 4️⃣ Train model
    print("\nTraining RandomForest model...")
    model = train_rf(X_train, y_train)

    # 5️⃣ Evaluation
    from sklearn.metrics import mean_squared_error
    import numpy as np

    preds_test = model.predict(X_test)

    mse = mean_squared_error(y_test, preds_test)
    rmse = np.sqrt(mse)

    baseline_preds = X_test["close_lag1"].values
    baseline_mse = mean_squared_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(baseline_mse)

    print("\n================ EVALUATION ================\n")
    print(f"Model MSE        : {mse:.4f}")
    print(f"Model RMSE       : {rmse:.4f} $")
    print(f"Naive MSE        : {baseline_mse:.4f}")
    print(f"Naive RMSE       : {baseline_rmse:.4f} $")

    improvement = baseline_rmse - rmse
    print(f"\nModel improvement vs naive: {improvement:.4f} $")

    # 6️⃣ Feature importance
    print("\n=========== TOP FEATURE IMPORTANCE ==========\n")

    try:
        importances = model.feature_importances_
        feat_imp = sorted(
            zip(cols, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        for name, val in feat_imp:
            print(f"{name:15s} -> {val:.4f}")

    except Exception:
        print("Feature importance not available.")

    # 7️⃣ Forecast future
    print("\n=============== FORECASTING ================\n")

    future_datetimes = get_future_trading_hours(
        df.index[-1],
        hours_to_predict
    )

    forecast_df = forecast_future_close(
        df,
        model,
        future_datetimes,
        cols
    )

    print("Forecast completed.\n")

    return model, forecast_df

# ----------------------------
# 6. MATPLOTLIB (NON-OVERLAPPING TIME AXIS)
# ----------------------------
def plot_forecast_close(forecast_df):
    start = forecast_df['Datetime'].min()
    end = forecast_df['Datetime'].max() + timedelta(hours=1)

    real_df = yf.download(
        "NVDA",
        start=start,
        end=end,
        interval="1h",
        auto_adjust=False,
        progress=False
    ).dropna()

    real_close = real_df['Close']
    common = forecast_df['Datetime'].isin(real_close.index)

    forecast_df = forecast_df[common]
    real_close = real_close.loc[forecast_df['Datetime']]

    plt.figure(figsize=(13, 6))

    plt.plot(
        real_close.index,
        real_close.values,
        label="Real Close (yfinance)",
        marker="o"
    )

    plt.plot(
        forecast_df['Datetime'],
        forecast_df['Close'],
        label="Predicted Close",
        linestyle="--",
        marker="x"
    )

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Datetime (Trading Hours)")
    plt.ylabel("Close Price")
    plt.title("NVDA Trading-Hour Close: Prediction vs Reality")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# after create_features(df) run on historical data, derive the model feature columns:
# Example:
feature_cols = [
    "close_lag1","close_lag2","close_lag3","close_lag6","close_lag12",
    "vol_lag1","vol_lag2","vol_lag3","vol_lag6","vol_lag12",
    "ret_1","ret_6","volatility_12",
    "MA20","MA50",
    "RSI","MACD","MACD_signal","MACD_hist"
]

# ----------------------------
# 7. MAIN
# ----------------------------
def main():

    file_path = "NVDA_hourly_last_2_years.csv"

    # ---------------- LOAD DATA ----------------
    df = load_nvda_data(file_path)

    df = add_moving_averages(df)

    # (optional) keep your interactive candle chart
    plot_and_save_chart(df)

    # ---------------- USER INPUT ----------------
    try:
        hours_to_predict = int(input("Enter trading hours to forecast: "))
        if hours_to_predict <= 0:
            raise ValueError
    except Exception:
        print("Invalid input → defaulting to 24 trading hours.")
        hours_to_predict = 24

    # ---------------- MACHINE LEARNING PIPELINE ----------------
    model, forecast_df = run_ml_pipeline_and_forecast(
        df,
        hours_to_predict
    )

    print("\nForecasted Close prices:")
    print(forecast_df)

    # ---------------- OPTIONAL PLOT ----------------
    # If you want to compare prediction vs real future prices
    # convert to format expected by your existing plot function

    forecast_plot_df = forecast_df.reset_index()
    forecast_plot_df.rename(
        columns={
            "PredictedClose": "Close"
        },
        inplace=True
    )

    plot_forecast_close(forecast_plot_df)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()
