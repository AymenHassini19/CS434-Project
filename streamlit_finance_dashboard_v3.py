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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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

        /* Dropdowns: force readable dark text in the control and menu */
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid rgba(15,23,42,0.35) !important;
            box-shadow: none !important;
        }
        [data-baseweb="select"],
        [data-baseweb="select"] * {
            color: #000000 !important;
        }
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {
            color: #000000 !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="select"] input::placeholder {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
        }
        [data-baseweb="select"] svg {
            fill: #000000 !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [role="listbox"] {
            background-color: #ffffff !important;
            border: 1px solid rgba(15,23,42,0.18) !important;
        }
        [data-baseweb="popover"] [role="option"],
        [role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        [data-baseweb="popover"] [role="option"] *,
        [role="listbox"] *,
        [role="option"] * {
            color: #000000 !important;
        }
        [data-baseweb="popover"] [role="option"]:hover,
        [role="option"]:hover {
            background-color: #e5e7eb !important;
            color: #000000 !important;
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
    "close_vs_ma20", "close_vs_ma50", "ma_gap_20_50",
    "macd_gap", "rsi_centered", "range_pct", "candle_body_pct",
    "volume_change", "volume_zscore_20", "trend_strength_12",
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
    df["close_vs_ma20"] = (df["Close"] / df["MA20"]) - 1
    df["close_vs_ma50"] = (df["Close"] / df["MA50"]) - 1
    df["ma_gap_20_50"] = (df["MA20"] / df["MA50"]) - 1
    df["macd_gap"] = df["MACD"] - df["MACD_signal"]
    df["rsi_centered"] = df["RSI"] - 50
    df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["candle_body_pct"] = (df["Close"] - df["Open"]) / df["Open"]
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_zscore_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["trend_strength_12"] = (
        df["Returns"].rolling(12).mean().abs() / df["Returns"].rolling(12).std()
    )
    df["target_next_return"] = np.log(df["Close"].shift(-1) / df["Close"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


@st.cache_resource(show_spinner=False)
def train_models(X_train, y_train):
    reg_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_depth=5,
        min_samples_leaf=16,
        l2_regularization=0.15,
        random_state=42,
    )
    direction_target = (y_train > 0).astype(int)
    cls_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.04,
        max_iter=350,
        max_depth=4,
        min_samples_leaf=16,
        l2_regularization=0.15,
        random_state=42,
    )
    reg_model.fit(X_train, y_train)
    if direction_target.nunique() < 2:
        cls_model = DummyClassifier(strategy="constant", constant=int(direction_target.iloc[0]))
    cls_model.fit(X_train, direction_target)
    return reg_model, cls_model


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


def build_trade_backtest(
    X_frame: pd.DataFrame,
    y_frame: pd.Series,
    pred_returns: np.ndarray,
    prob_up: np.ndarray,
    return_threshold: float,
    prob_threshold: float,
):
    trade_df = pd.DataFrame(index=X_frame.index.copy())
    trade_df.index.name = "Datetime"
    trade_df["Close"] = X_frame["Close"].values
    trade_df["PredictedLogReturn"] = pred_returns
    trade_df["ActualLogReturn"] = y_frame.values
    trade_df["ProbUp"] = prob_up
    trade_df["PredictedSimpleReturn"] = np.exp(trade_df["PredictedLogReturn"]) - 1
    trade_df["ActualSimpleReturn"] = np.exp(trade_df["ActualLogReturn"]) - 1

    long_trend = (
        (X_frame["Close"] >= X_frame["MA20"])
        & (X_frame["MA20"] >= X_frame["MA50"])
        & (X_frame["MACD_hist"] >= 0)
    )
    short_trend = (
        (X_frame["Close"] <= X_frame["MA20"])
        & (X_frame["MA20"] <= X_frame["MA50"])
        & (X_frame["MACD_hist"] <= 0)
    )
    long_signal = (
        (trade_df["PredictedSimpleReturn"] >= return_threshold)
        & (trade_df["ProbUp"] >= prob_threshold)
        & long_trend.values
    )
    short_signal = (
        (trade_df["PredictedSimpleReturn"] <= -return_threshold)
        & (trade_df["ProbUp"] <= (1 - prob_threshold))
        & short_trend.values
    )

    trade_df["Signal"] = np.where(long_signal, "Long", np.where(short_signal, "Short", "Flat"))
    trade_df["Confidence"] = np.where(
        trade_df["Signal"] == "Long",
        trade_df["ProbUp"],
        np.where(trade_df["Signal"] == "Short", 1 - trade_df["ProbUp"], abs(trade_df["ProbUp"] - 0.5) * 2),
    )
    trade_df["DirectionalReturn"] = np.where(
        trade_df["Signal"] == "Long",
        trade_df["ActualSimpleReturn"],
        np.where(trade_df["Signal"] == "Short", -trade_df["ActualSimpleReturn"], 0.0),
    )
    trade_df["PredictedNextClose"] = trade_df["Close"] * np.exp(trade_df["PredictedLogReturn"])
    trade_df["ActualNextClose"] = trade_df["Close"] * np.exp(trade_df["ActualLogReturn"])
    return trade_df


def simulate_trades(trade_df: pd.DataFrame, starting_cash: float, trade_count: int, position_size: float = 1.0):
    selected = trade_df.loc[trade_df["Signal"] != "Flat"].head(min(trade_count, len(trade_df))).copy()
    cash = float(starting_cash)
    history = []

    for trade_number, (dt, row) in enumerate(selected.iterrows(), start=1):
        cash_before = cash
        portfolio_return = position_size * float(row["DirectionalReturn"])
        trade_multiplier = max(0.0, 1 + portfolio_return)
        cash = cash_before * trade_multiplier
        history.append(
            {
                "Trade": trade_number,
                "Datetime": pd.to_datetime(dt),
                "Signal": row["Signal"],
                "Confidence": float(row["Confidence"]) * 100,
                "Predicted Return %": float(row["PredictedSimpleReturn"]) * 100,
                "Actual Return %": float(row["ActualSimpleReturn"]) * 100,
                "Trade Return %": float(row["DirectionalReturn"]) * 100,
                "Position Size %": position_size * 100,
                "Cash Before": cash_before,
                "Cash After": cash,
                "Outcome": "Win" if cash > cash_before else "Loss" if cash < cash_before else "Flat",
            }
        )
        if cash <= 0:
            break

    sim_df = pd.DataFrame(history)
    if sim_df.empty:
        summary = {
            "final_cash": float(starting_cash),
            "net_profit": 0.0,
            "total_return_pct": 0.0,
            "trades_executed": 0,
            "wins": 0,
            "losses": 0,
            "skipped": int((trade_df["Signal"] == "Flat").sum()),
            "bankrupt": False,
        }
        return sim_df, summary

    wins = int((sim_df["Outcome"] == "Win").sum())
    losses = int((sim_df["Outcome"] == "Loss").sum())
    final_cash = float(sim_df["Cash After"].iloc[-1])
    summary = {
        "final_cash": final_cash,
        "net_profit": final_cash - float(starting_cash),
        "total_return_pct": ((final_cash / float(starting_cash)) - 1) * 100 if starting_cash else 0.0,
        "trades_executed": int(len(sim_df)),
        "wins": wins,
        "losses": losses,
        "skipped": int((trade_df["Signal"] == "Flat").sum()),
        "bankrupt": final_cash <= 0,
    }
    return sim_df, summary


def optimize_trading_policy(X_calib: pd.DataFrame, y_calib: pd.Series, pred_calib: np.ndarray, prob_up_calib: np.ndarray):
    pred_simple = np.exp(pred_calib) - 1
    pred_abs = pd.Series(np.abs(pred_simple))
    threshold_candidates = sorted(
        {
            0.0,
            round(float(pred_abs.quantile(0.40)), 6),
            round(float(pred_abs.quantile(0.55)), 6),
            round(float(pred_abs.quantile(0.70)), 6),
            round(float(pred_abs.quantile(0.82)), 6),
        }
    )
    prob_threshold_candidates = [0.50, 0.55, 0.60, 0.65]
    position_size_candidates = [0.25, 0.40, 0.50, 0.65]
    min_trades = max(8, min(40, len(X_calib) // 8))
    best_policy = {
        "return_threshold": 0.0,
        "prob_threshold": 0.55,
        "position_size": 0.40,
    }
    best_score = float("-inf")

    for return_threshold in threshold_candidates:
        for prob_threshold in prob_threshold_candidates:
            trade_df = build_trade_backtest(
                X_calib,
                y_calib,
                pred_calib,
                prob_up_calib,
                return_threshold,
                prob_threshold,
            )
            executed_trades = int((trade_df["Signal"] != "Flat").sum())
            if executed_trades < min_trades:
                continue

            for position_size in position_size_candidates:
                _, summary = simulate_trades(
                    trade_df,
                    starting_cash=10000.0,
                    trade_count=executed_trades,
                    position_size=position_size,
                )
                score = summary["final_cash"] + (summary["wins"] - summary["losses"]) * 10
                if score > best_score:
                    best_score = score
                    best_policy = {
                        "return_threshold": return_threshold,
                        "prob_threshold": prob_threshold,
                        "position_size": position_size,
                    }

    return best_policy


def recursive_forecast(df: pd.DataFrame, horizon: int):
    feat = build_features(df)
    if len(feat) < 60:
        raise ValueError("Not enough cleaned rows after feature engineering.")

    split = int(len(feat) * 0.8)
    train = feat.iloc[:split]
    test = feat.iloc[split:]
    calibration_rows = max(20, int(len(train) * 0.2))
    if len(train) - calibration_rows < 40:
        calibration_rows = max(10, len(train) // 4)
    model_train = train.iloc[:-calibration_rows]
    calibration = train.iloc[-calibration_rows:]
    if len(model_train) < 40 or len(calibration) < 10:
        model_train = train
        calibration = train.tail(max(10, len(train) // 5)).copy()

    cols = [c for c in FEATURE_COLS if c in train.columns]
    X_model_train = model_train[cols]
    y_model_train = model_train["target_next_return"]
    X_calib = calibration[cols]
    y_calib = calibration["target_next_return"]
    X_train = train[cols]
    y_train = train["target_next_return"]
    X_test = test[cols]
    y_test = test["target_next_return"]

    reg_model, cls_model = train_models(X_model_train, y_model_train)
    pred_calib = reg_model.predict(X_calib)
    prob_up_calib = cls_model.predict_proba(X_calib)[:, 1]
    trade_policy = optimize_trading_policy(X_calib, y_calib, pred_calib, prob_up_calib)

    reg_model, cls_model = train_models(X_train, y_train)
    pred_test = reg_model.predict(X_test)
    prob_up_test = cls_model.predict_proba(X_test)[:, 1]
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))

    actual_next_close = X_test["Close"].values * np.exp(y_test.values)
    predicted_next_close = X_test["Close"].values * np.exp(pred_test)
    naive_next_close = X_test["Close"].values
    price_rmse = float(np.sqrt(mean_squared_error(actual_next_close, predicted_next_close)))
    naive_rmse = float(np.sqrt(mean_squared_error(actual_next_close, naive_next_close)))
    trade_backtest_df = build_trade_backtest(
        X_test,
        y_test,
        pred_test,
        prob_up_test,
        trade_policy["return_threshold"],
        trade_policy["prob_threshold"],
    )
    _, test_trade_summary = simulate_trades(
        trade_backtest_df,
        starting_cash=10000.0,
        trade_count=int((trade_backtest_df["Signal"] != "Flat").sum()),
        position_size=trade_policy["position_size"],
    )

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
        f"Policy calibration rows: {len(calibration):,}",
        f"Testing rows: {len(test):,}",
        f"Number of features used: {len(cols)}",
        "",
        "Feature groups:",
        "- trend and momentum indicators",
        "- lagged prices and returns",
        "- rolling volatility",
        "- RSI and MACD technical indicators",
        "- volume-based and trend-strength features",
        "",
        "Models:",
        "- HistGradientBoostingRegressor for next-period return size",
        "- HistGradientBoostingClassifier for up/down probability",
        "",
        "Trading policy:",
        f"- minimum predicted move: {trade_policy['return_threshold'] * 100:.3f}%",
        f"- minimum directional probability: {trade_policy['prob_threshold']:.2f}",
        f"- position size per trade: {trade_policy['position_size'] * 100:.0f}%",
        "- trend filter: only trade when MA/MACD agree with the direction",
    ]
    pipeline_text = "\n".join(pipeline_lines)

    eval_lines = [
        "================ EVALUATION ================",
        "",
        f"Return RMSE: {rmse:.6f}",
        f"Price RMSE: {price_rmse:.4f} $",
        f"Naive RMSE: {naive_rmse:.4f} $",
        f"Improvement vs naive: {naive_rmse - price_rmse:.4f} $",
        "",
        "Trading backtest on unseen validation period:",
        f"Trades taken: {test_trade_summary['trades_executed']}",
        f"Trades skipped: {test_trade_summary['skipped']}",
        f"Win / loss count: {test_trade_summary['wins']} / {test_trade_summary['losses']}",
        f"Ending cash from 10,000 $: {test_trade_summary['final_cash']:.2f} $",
        f"Backtest return: {test_trade_summary['total_return_pct']:.2f} %",
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
        pred_log_ret = float(reg_model.predict(X_last)[0])
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
    return reg_model, rmse, forecast_df, feat, cols, pipeline_text, eval_text, feat_df, trade_backtest_df, trade_policy


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
overview_tab, indicators_tab, forecast_tab, data_tab, pipeline_tab, evaluation_tab, features_tab, trading_tab = st.tabs(
    ["Overview", "Indicators", "Forecast", "Data", "ML Pipeline", "Evaluation", "Top Features", "Trade Simulation"]
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
        model, rmse, forecast_df, feat_df, used_cols, pipeline_text, eval_text, top_features_df, trade_backtest_df, trade_policy = recursive_forecast(
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

with trading_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    try:
        eligible_trades = int((trade_backtest_df["Signal"] != "Flat").sum())
        if eligible_trades == 0:
            st.warning("The optimized strategy skipped every validation-period setup, so there are no trades to simulate.")
        else:
            max_trades = eligible_trades
            default_trades = min(25, max_trades)
            c1, c2 = st.columns(2)
            with c1:
                starting_cash = st.number_input("Starting cash ($)", min_value=100.0, value=10000.0, step=500.0)
            with c2:
                trade_count = st.slider("Number of trades to simulate", 1, max_trades, default_trades)

            st.caption(
                f"Simulation uses the optimized policy from the training data: minimum predicted move "
                f"{trade_policy['return_threshold'] * 100:.3f}%, minimum direction probability "
                f"{trade_policy['prob_threshold']:.2f}, and {trade_policy['position_size'] * 100:.0f}% of capital per trade."
            )

            sim_df, sim_summary = simulate_trades(
                trade_backtest_df,
                starting_cash,
                trade_count,
                position_size=trade_policy["position_size"],
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final cash", f"${sim_summary['final_cash']:,.2f}", delta=f"${sim_summary['net_profit']:,.2f}")
            m2.metric("Total return", f"{sim_summary['total_return_pct']:.2f}%")
            m3.metric("Trades / Wins", f"{sim_summary['trades_executed']} / {sim_summary['wins']}")
            m4.metric("Fell to zero", "Yes" if sim_summary["bankrupt"] else "No")

            sim_fig = go.Figure()
            sim_fig.add_trace(go.Scatter(x=sim_df["Datetime"], y=sim_df["Cash After"], mode="lines+markers", name="Portfolio Value"))
            sim_fig.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                title="Simulated Portfolio Value Across Validation Trades",
                xaxis_title="Trade Date / Time",
                yaxis_title="Portfolio Value ($)",
            )
            st.plotly_chart(sim_fig, use_container_width=True)
            st.dataframe(sim_df, use_container_width=True)
    except NameError:
        st.write("Run the forecast tab first to generate the trade simulation.")
    st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.subheader("Summary statistics for the loaded dataset")
st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
