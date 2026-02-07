import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import matplotlib.pyplot as plt
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
    df = df.dropna(subset=['Datetime'])  # Drop rows with invalid dates
    
    df.set_index('Datetime', inplace=True)
    df = df.sort_index()
    
    # Ensure numeric columns are numeric
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

    # Layout
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
# 4. PREDICTION FUNCTION
# ----------------------------
def get_predictions(df, hours):
    df_p = df[['Close']].dropna().copy()
    if df_p.empty:
        raise ValueError("No valid 'Close' data available for prediction.")
    
    df_p['Index'] = np.arange(len(df_p))
    
    model = LinearRegression()
    model.fit(df_p[['Index']], df_p['Close'])
    
    last_idx = df_p['Index'].iloc[-1]
    future_idx = np.arange(last_idx + 1, last_idx + 1 + hours).reshape(-1, 1)
    preds = model.predict(future_idx)
    
    future_dates = [df_p.index[-1] + timedelta(hours=i+1) for i in range(hours)]
    return pd.DataFrame({'Datetime': future_dates, 'Predicted': preds})

# ----------------------------
# 5. MATPLOTLIB PLOTTING
# ----------------------------
def plot_forecast(df, forecast_df, last_n=100):
    plt.figure(figsize=(12,6))
    plt.plot(df.index[-last_n:], df['Close'].tail(last_n), label='Actual', marker='o', markersize=2)
    plt.plot(forecast_df['Datetime'], forecast_df['Predicted'], label='Forecast', linestyle='--', color='orange', marker='x')
    plt.xlabel('Datetime')
    plt.ylabel('Price')
    plt.title('NVDA Price Forecast')
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

    forecast_df = get_predictions(df, hours_to_predict)
    print(forecast_df)
    plot_forecast(df, forecast_df)

# Run the main function
if __name__ == "__main__":
    main()
