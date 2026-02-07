import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ----------------------------
# 1. DATA COLLECTION
# ----------------------------
def load_nvda_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    # Skip metadata rows (if present) and handle missing headers
    df = pd.read_csv(file_path, skiprows=3, header=None)
    df.columns = ['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Convert datetime
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

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='NVDA'
    ), row=1, col=1)
    
    # Moving averages
    for ma in [col for col in df.columns if 'MA' in col]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[ma], line=dict(width=2), name=ma
        ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume', marker_color='grey'
    ), row=2, col=1)

    fig.update_layout(
        title='NVDA Hourly Analysis',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )
    
    fig.write_html(save_path)
    print(f"Interactive chart saved as '{save_path}'. Open this file in your browser.")
    fig.show()

# ----------------------------
# 4. PREDICTION FUNCTION (All Columns)
# ----------------------------
def get_predictions_all_columns(df, hours):
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    predictions = pd.DataFrame()
    
    df_p = df[features].dropna().copy()
    df_p['Index'] = np.arange(len(df_p))
    last_idx = df_p['Index'].iloc[-1]
    
    future_dates = [df_p.index[-1] + timedelta(hours=i+1) for i in range(hours)]
    predictions['Datetime'] = future_dates
    
    for col in features:
        model = LinearRegression()
        model.fit(df_p[['Index']], df_p[col])
        future_idx = np.arange(last_idx + 1, last_idx + 1 + hours).reshape(-1, 1)
        predictions[col] = model.predict(future_idx)
    
    return predictions

# ----------------------------
# 5. MATPLOTLIB PLOTTING (Close Only)
# ----------------------------
def plot_forecast_close(df, forecast_df, last_n=100):
    plt.figure(figsize=(12,6))
    
    # Last actual Close values
    actual = df['Close'].tail(last_n)
    
    # Plot actual
    plt.plot(actual.index, actual.values, label='Actual Close', marker='o', markersize=4)
    # Plot forecast
    plt.plot(forecast_df['Datetime'], forecast_df['Close'], label='Forecast Close', linestyle='--', color='orange', marker='x')
    
    # Format x-axis for hours
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    plt.xlabel('Datetime (Hourly)')
    plt.ylabel('Close Price')
    plt.title('NVDA Close Price Forecast')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 6. MAIN EXECUTION
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
        print("Invalid input! Forecasting 24 hours by default.")
        hours_to_predict = 24

    forecast_df = get_predictions_all_columns(df, hours_to_predict)
    print(forecast_df)

    # Plot only Close price with proper hourly scaling
    plot_forecast_close(df, forecast_df)

# Run the main function
if __name__ == "__main__":
    main()
