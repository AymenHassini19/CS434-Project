import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import os

# ----------------------------
# 1. DATA COLLECTION
# ----------------------------
def load_nvda_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    df = pd.read_csv(file_path, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
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

# ----------------------------
# 3. INTERACTIVE VISUALIZATION
# ----------------------------
def plot_and_save_chart(df, save_path="stock_analysis.html"):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Trends', 'Volume'),
        row_width=[0.2, 0.7]
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='NVDA'
    ), row=1, col=1)

    for ma in [col for col in df.columns if 'MA' in col]:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[ma],
            line=dict(width=2),
            name=ma
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color='grey'
    ), row=2, col=1)

    fig.update_layout(
        title='NVDA Hourly Analysis',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )

    fig.write_html(save_path)
    fig.show()

# ----------------------------
# 4. PREDICTION FUNCTION
# ----------------------------
def get_predictions_all_columns(df, hours):
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    predictions = pd.DataFrame()

    df_p = df[features].copy()
    df_p['Index'] = np.arange(len(df_p))
    last_idx = df_p['Index'].iloc[-1]

    future_dates = [df_p.index[-1] + timedelta(hours=i + 1) for i in range(hours)]
    predictions['Datetime'] = future_dates

    for col in features:
        model = LinearRegression()
        model.fit(df_p[['Index']], df_p[col])

        future_idx = np.arange(last_idx + 1, last_idx + 1 + hours).reshape(-1, 1)
        predictions[col] = model.predict(future_idx)

    return predictions

# ----------------------------
# 5. MATPLOTLIB: REAL vs PREDICTED CLOSE
# ----------------------------

def plot_forecast_close(forecast_df):
    start = forecast_df['Datetime'].min()
    end = forecast_df['Datetime'].max() + timedelta(hours=1)

    real_df = yf.download(
        "NVDA",
        start=start,
        end=end,
        interval="1h",
        progress=False
    )

    if real_df.empty:
        print("No real data available yet from yfinance.")
        return

    # ---- ALIGN DATA ----
    common_times = forecast_df['Datetime'].isin(real_df.index)
    aligned_forecast = forecast_df[common_times]

    if aligned_forecast.empty:
        print("No overlapping hours between prediction and real data yet.")
        return

    real_close = real_df.loc[aligned_forecast['Datetime'], 'Close']

    # ---- PLOT ----
    plt.figure(figsize=(12, 6))

    plt.plot(
        real_close.index,
        real_close.values,
        label="Real Close (yfinance)",
        marker="o"
    )

    plt.plot(
        aligned_forecast['Datetime'],
        aligned_forecast['Close'],
        label="Predicted Close",
        linestyle="--",
        marker="x"
    )

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    plt.xticks(rotation=45)
    plt.xlabel("Datetime (Hourly)")
    plt.ylabel("Close Price")
    plt.title("NVDA Hourly Close Price: Prediction vs Reality (Aligned)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------
# 6. MAIN
# ----------------------------
def main():
    file_path = 'NVDA_hourly_last_2_years.csv'

    df = load_nvda_data(file_path)
    df = add_moving_averages(df)

    plot_and_save_chart(df)

    try:
        hours_to_predict = int(input("Enter hours to forecast: "))
        if hours_to_predict <= 0:
            raise ValueError
    except ValueError:
        print("Invalid input! Defaulting to 24 hours.")
        hours_to_predict = 24

    forecast_df = get_predictions_all_columns(df, hours_to_predict)
    print(forecast_df)

    plot_forecast_close(forecast_df)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    main()
