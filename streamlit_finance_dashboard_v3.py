import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from io import BytesIO
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_market_calendars as mcal
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Finance Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #08111f 0%, #101826 100%);
            color: #f8fafc;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            color: #f8fafc;
        }
        h1, h2, h3, h4, h5, h6, p, span, label, div, a, li {
            color: #f8fafc !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(15,23,42,0.92);
            border: 1px solid rgba(255,255,255,0.12);
            padding: 14px 16px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        }
        div[data-testid="stMetric"] * {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] {
            background: #0b1320;
        }

        /* High-contrast sidebar inputs */
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #0f172a !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] [role="listbox"] {
            background-color: #0f172a !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
        }
        section[data-testid="stSidebar"] [role="option"] {
            background-color: #0f172a !important;
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] [role="option"]:hover {
            background-color: #1e293b !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
            background: #0f172a !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            border-radius: 16px !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] * {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] button {
            background: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
        }

        .card {
            background: rgba(15,23,42,0.96);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 20px;
            padding: 18px 18px 8px 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }
        .output-card {
            background: rgba(15,23,42,0.96);
            border: 1px solid rgba(96,165,250,0.35);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        }
        .small-note {
            color: #cbd5e1 !important;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Helpers -----------------------------
CANONICAL_COLS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


def _guess_column(options, preferred_keywords):
    lower = {c.lower(): c for c in options}
    for key in preferred_keywords:
        for lc, orig in lower.items():
            if key in lc:
                return orig
    return options[0] if options else None


@st.cache_data(show_spinner=False)
def fetch_yfinance_data(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    return df


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes):
    return pd.read_csv(BytesIO(file_bytes))


def normalize_dataframe(df: pd.DataFrame, dt_col: str, open_col: str, high_col: str, low_col: str, close_col: str, volume_col: str | None):
    out = pd.DataFrame()
    out["Datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    out["Open"] = pd.to_numeric(df[open_col], errors="coerce")
    out["High"] = pd.to_numeric(df[high_col], errors="coerce")
    out["Low"] = pd.to_numeric(df[low_col], errors="coerce")
    out["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    if volume_col and volume_col in df.columns:
        out["Volume"] = pd.to_numeric(df[volume_col], errors="coerce")
    else:
        out["Volume"] = 0

    out = out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    out = out.sort_values("Datetime").drop_duplicates("Datetime")
    out = out.set_index("Datetime")
    return out


def add_indicators(df: pd.DataFrame):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["Returns"] = df["Close"].pct_change()
    return df


FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA20", "MA50", "RSI", "MACD", "MACD_signal", "MACD_hist",
    "ret_1", "ret_3", "ret_6", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "volatility_12", "volatility_24",
]


def build_features(df: pd.DataFrame):
    df = add_indicators(df)
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_6"] = df["Close"].pct_change(6)
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)
    df["volatility_12"] = df["Returns"].rolling(12).std()
    df["volatility_24"] = df["Returns"].rolling(24).std()
    df["target_next_return"] = np.log(df["Close"].shift(-1) / df["Close"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def infer_frequency(index: pd.DatetimeIndex):
    inferred = pd.infer_freq(index)
    if inferred:
        return inferred
    if len(index) < 2:
        return "1D"
    deltas = index.to_series().diff().dropna()
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return "1D"
    return median_delta


def future_index(last_dt, horizon, freq):
    if isinstance(freq, str):
        return pd.date_range(start=last_dt, periods=horizon + 1, freq=freq, inclusive="right")
    if isinstance(freq, pd.Timedelta):
        return pd.date_range(start=last_dt, periods=horizon + 1, freq=freq, inclusive="right")
    return pd.date_range(start=last_dt, periods=horizon + 1, freq="1D", inclusive="right")


def recursive_forecast(df: pd.DataFrame, horizon: int):
    feat = build_features(df)
    if len(feat) < 60:
        raise ValueError("Not enough cleaned rows after feature engineering.")

    split = int(len(feat) * 0.8)
    train = feat.iloc[:split]
    test = feat.iloc[split:]

    cols = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[cols]
    y_train = train["target_next_return"]
    X_test = test[cols]
    y_test = test["target_next_return"]

    model = train_model(X_train, y_train)
    pred_test = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

    actual_next_close = X_test["Close"].values * np.exp(y_test.values)
    predicted_next_close = X_test["Close"].values * np.exp(pred_test)
    naive_next_close = X_test["Close"].values
    price_rmse = float(np.sqrt(mean_squared_error(actual_next_close, predicted_next_close)))
    naive_rmse = float(np.sqrt(mean_squared_error(actual_next_close, naive_next_close)))

    corr = X_train.copy()
    corr["target"] = y_train.values
    feat_imp = (
        corr.corr(numeric_only=True)["target"]
        .drop("target")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )
    feat_df = feat_imp.reset_index()
    feat_df.columns = ["Feature", "Importance"]

    pipeline_lines = [
        "================ ML PIPELINE ================",
        "",
        f"Input rows after cleaning: {len(df):,}",
        f"Rows after feature engineering: {len(feat):,}",
        f"Training rows: {len(train):,}",
        f"Testing rows: {len(test):,}",
        f"Number of features used: {len(cols)}",
        "",
        "Feature groups:",
        "- trend and momentum indicators",
        "- lagged prices and returns",
        "- rolling volatility",
        "- RSI and MACD technical indicators",
        "- volume-based features",
        "",
        "Model:",
        "HistGradientBoostingRegressor(",
        "    loss='squared_error', learning_rate=0.05, max_iter=400,",
        "    max_depth=5, min_samples_leaf=20, l2_regularization=0.1,",
        "    random_state=42",
        ")",
        "",
        "Prediction target:",
        "Next-hour log return",
    ]
    pipeline_text = "\n".join(pipeline_lines)

    eval_lines = [
        "================ EVALUATION ================",
        "",
        f"Return RMSE: {rmse:.6f}",
        f"Price RMSE: {price_rmse:.4f} $",
        f"Naive RMSE: {naive_rmse:.4f} $",
        f"Improvement vs naive: {naive_rmse - price_rmse:.4f} $",
    ]
    eval_text = "\n".join(eval_lines)

    df_fc = df.copy()
    freq = infer_frequency(df_fc.index)
    future_dates = future_index(df_fc.index[-1], horizon, freq)
    forecast_rows = []

    for dt in future_dates:
        feat_fc = build_features(df_fc)
        last = feat_fc.iloc[-1]
        X_last = pd.DataFrame([last[cols].values], columns=cols)
        pred_log_ret = float(model.predict(X_last)[0])
        last_close = float(df_fc["Close"].iloc[-1])
        pred_close = last_close * np.exp(pred_log_ret)

        new_row = df_fc.iloc[-1].copy()
        new_row["Open"] = last_close
        new_row["High"] = max(last_close, pred_close)
        new_row["Low"] = min(last_close, pred_close)
        new_row["Close"] = pred_close
        new_row["Volume"] = float(df_fc["Volume"].iloc[-1]) if "Volume" in df_fc.columns else 0
        df_fc.loc[pd.to_datetime(dt)] = new_row

        forecast_rows.append((pd.to_datetime(dt), pred_close))

    forecast_df = pd.DataFrame(forecast_rows, columns=["Datetime", "PredictedClose"]).set_index("Datetime")
    return model, rmse, forecast_df, feat, cols, pipeline_text, eval_text, feat_df


def candlestick_figure(df: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_width=[0.25, 0.75],
        subplot_titles=(
            "Price Chart: Candlesticks and Moving Averages",
            "Trading Volume",
        ),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1, col=1,
    )
    for col in [c for c in ["MA20", "MA50"] if c in df.columns]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode="lines"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=55, b=10),
        title="Stock Price Overview: Open, High, Low, Close, Moving Averages, and Volume",
        xaxis_title="Date / Time",
        yaxis_title="Price",
        yaxis2_title="Volume",
    )
    return fig
# ----------------------------- Sidebar -----------------------------
st.sidebar.title("Dashboard Inputs")
source = st.sidebar.radio("Data source", ["Upload CSV", "Yahoo Finance ticker"], index=0)

raw_df = None
source_name = None

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded is not None:
        raw_df = read_uploaded_csv(uploaded.getvalue())
        source_name = uploaded.name
else:
    ticker = st.sidebar.text_input("Ticker", value="NVDA")
    interval = st.sidebar.selectbox("Interval", ["1h", "1d", "30m", "15m"], index=0)

    if interval == "1h":
        period = "730d"
        st.sidebar.caption("Hourly Yahoo Finance data uses the maximum 730-day lookback.")
    else:
        period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    if st.sidebar.button("Load market data"):
        raw_df = fetch_yfinance_data(ticker, period, interval)
        source_name = f"{ticker} ({period}, {interval})"

horizon = st.sidebar.slider("Forecast horizon", 5, 100, 24)

st.title("📈 Finance Forecast Dashboard")
st.caption("Modern Streamlit dashboard with uploads, charts, indicators, and forecasting.")

if raw_df is None or raw_df.empty:
    st.info("Upload a CSV or load market data from Yahoo Finance to start.")
    st.stop()

st.subheader(f"Loaded data: {source_name}")

# Column mapping for uploads
if source == "Upload CSV":
    cols = list(raw_df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        dt_col = st.selectbox("Datetime column", cols, index=cols.index(_guess_column(cols, ["date", "time", "datetime"])) if cols else 0)
    with c2:
        open_col = st.selectbox("Open column", cols, index=cols.index(_guess_column(cols, ["open"])) if cols else 0)
        high_col = st.selectbox("High column", cols, index=cols.index(_guess_column(cols, ["high"])) if cols else 0)
    with c3:
        low_col = st.selectbox("Low column", cols, index=cols.index(_guess_column(cols, ["low"])) if cols else 0)
        close_col = st.selectbox("Close column", cols, index=cols.index(_guess_column(cols, ["close", "adj close"])) if cols else 0)
    volume_candidates = ["(none)"] + cols
    vol_guess = _guess_column(cols, ["volume", "vol"])
    volume_col = st.selectbox("Volume column", volume_candidates, index=volume_candidates.index(vol_guess) if vol_guess in volume_candidates else 0)
    volume_col = None if volume_col == "(none)" else volume_col
    df = normalize_dataframe(raw_df, dt_col, open_col, high_col, low_col, close_col, volume_col)
else:
    # yfinance data already comes in standard columns
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "Date" in df.columns and "Datetime" not in df.columns:
        df = df.rename(columns={"Date": "Datetime"})
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0

# Clean / enrich
if not isinstance(df.index, pd.DatetimeIndex):
    st.error("The selected datetime column could not be parsed correctly.")
    st.stop()

df = df.sort_index().dropna(subset=["Open", "High", "Low", "Close"])
df = add_indicators(df)

# KPIs
latest_close = float(df["Close"].iloc[-1])
start_dt = df.index.min()
end_dt = df.index.max()
rows = len(df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{rows:,}")
k2.metric("Start", str(start_dt.date()))
k3.metric("End", str(end_dt.date()))
k4.metric("Latest Close", f"{latest_close:,.2f}")

# Main tabs
overview_tab, indicators_tab, forecast_tab, data_tab, pipeline_tab, evaluation_tab, features_tab = st.tabs(
    ["Overview", "Indicators", "Forecast", "Data", "ML Pipeline", "Evaluation", "Top Features"]
)

with overview_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(candlestick_figure(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with indicators_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    ind1, ind2 = st.columns(2)
    with ind1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
        fig.add_hline(y=70, line_dash="dash")
        fig.add_hline(y=30, line_dash="dash")
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            title="Relative Strength Index (RSI): Momentum and Overbought/Oversold Levels",
            xaxis_title="Date / Time",
            yaxis_title="RSI Value",
        )
        st.plotly_chart(fig, use_container_width=True)
    with ind2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
        if "MACD_signal" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal"))
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=50, b=10),
            title="MACD Indicator: Trend Strength and Momentum",
            xaxis_title="Date / Time",
            yaxis_title="Indicator Value",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df[[c for c in ["MA20", "MA50", "RSI", "MACD", "MACD_signal", "MACD_hist"] if c in df.columns]].tail(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with forecast_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    try:
        model, rmse, forecast_df, feat_df, used_cols, pipeline_text, eval_text, top_features_df = recursive_forecast(
            df[["Open", "High", "Low", "Close", "Volume"]].copy(), horizon
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Validation RMSE", f"{rmse:.6f}")
        c2.metric("Features used", f"{len(used_cols)}")
        c3.metric("Forecast steps", f"{horizon}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-200:], y=df["Close"].tail(200), mode="lines", name="Historical Close"))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["PredictedClose"], mode="lines+markers", name="Forecasted Close"))
        fig.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=10, r=10, t=50, b=10),
            title="Forecasted Close Price Compared with Recent Historical Prices",
            xaxis_title="Date / Time",
            yaxis_title="Close Price",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df, use_container_width=True)
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
with data_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.tail(200), use_container_width=True)
    csv = df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned data as CSV", csv, file_name="cleaned_data.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with pipeline_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    try:
        st.text(pipeline_text)
    except NameError:
        st.write("Run the forecast tab first to generate the ML pipeline output.")
    st.markdown('</div>', unsafe_allow_html=True)

with evaluation_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    try:
        st.text(eval_text)
    except NameError:
        st.write("Run the forecast tab first to generate the evaluation output.")
    st.markdown('</div>', unsafe_allow_html=True)

with features_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    try:
        st.dataframe(top_features_df, use_container_width=True)
        feat_fig = go.Figure()
        feat_fig.add_trace(go.Bar(x=top_features_df["Feature"], y=top_features_df["Importance"], name="Importance"))
        feat_fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            title="Top 10 Feature Importance Scores",
            xaxis_title="Feature",
            yaxis_title="Importance Score",
        )
        st.plotly_chart(feat_fig, use_container_width=True)
    except NameError:
        st.write("Run the forecast tab first to generate top feature importance.")
    st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.subheader("Summary statistics for the loaded dataset")
st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
