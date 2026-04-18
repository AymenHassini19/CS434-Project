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
        .hero-panel {
            position: relative;
            overflow: hidden;
            padding: 24px 24px 20px 24px;
            border-radius: 26px;
            border: 1px solid rgba(148,163,184,0.22);
            background:
                radial-gradient(circle at top right, rgba(56,189,248,0.22), transparent 34%),
                radial-gradient(circle at bottom left, rgba(249,115,22,0.18), transparent 30%),
                linear-gradient(135deg, rgba(8,17,31,0.98), rgba(15,23,42,0.94));
            box-shadow: 0 18px 45px rgba(2,6,23,0.35);
            margin-bottom: 1rem;
        }
        .hero-panel::after {
            content: "";
            position: absolute;
            inset: auto -120px -120px auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(59,130,246,0.22), transparent 65%);
            pointer-events: none;
        }
        .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.74rem;
            color: #7dd3fc !important;
            margin-bottom: 0.55rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
            color: #f8fafc !important;
        }
        .hero-copy {
            margin: 0.6rem 0 0 0;
            max-width: 820px;
            color: #cbd5e1 !important;
            line-height: 1.6;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.85rem;
            border-radius: 999px;
            border: 1px solid rgba(125,211,252,0.22);
            background: rgba(15,23,42,0.78);
            color: #e2e8f0 !important;
            font-size: 0.9rem;
        }
        .mini-panel {
            height: 100%;
            padding: 18px 18px 14px 18px;
            border-radius: 22px;
            border: 1px solid rgba(148,163,184,0.16);
            background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 10px 24px rgba(2,6,23,0.22);
        }
        .panel-title {
            margin: 0 0 0.45rem 0;
            font-size: 1.02rem;
            font-weight: 700;
            color: #f8fafc !important;
        }
        .panel-copy {
            margin: 0;
            color: #cbd5e1 !important;
            line-height: 1.6;
        }
        .section-label {
            margin: 1rem 0 0.6rem 0;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            color: #7dd3fc !important;
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


def render_hero_panel(kicker: str, title: str, body: str, pills: list[str] | None = None):
    pills_html = "".join(f'<span class="pill">{pill}</span>' for pill in (pills or []))
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">{kicker}</div>
            <h2 class="hero-title">{title}</h2>
            <p class="hero-copy">{body}</p>
            <div class="pill-row">{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_panel(title: str, body: str):
    st.markdown(
        f"""
        <div class="mini-panel">
            <div class="panel-title">{title}</div>
            <p class="panel-copy">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_figure(fig: go.Figure, title: str, height: int = 360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=85, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        title=dict(
            text=title,
            y=0.97,
            x=0.02,
            xanchor="left",
            font=dict(color="#ffffff", size=24),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.14,
            x=0,
            font=dict(color="#ffffff", size=12),
        ),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#f8fafc"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        color="#ffffff",
        tickfont=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.16)",
        zeroline=False,
        color="#ffffff",
        tickfont=dict(color="#ffffff"),
        title_font=dict(color="#ffffff"),
    )
    return fig


def build_feature_group_breakdown(columns: list[str]):
    group_map = {
        "Core OHLCV": {"Open", "High", "Low", "Close", "Volume"},
        "Trend & momentum": {"MA20", "MA50", "close_vs_ma20", "close_vs_ma50", "ma_gap_20_50", "trend_strength_12"},
        "Return & lag memory": {"ret_1", "ret_3", "ret_6", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12"},
        "Volatility & candle structure": {"volatility_12", "volatility_24", "range_pct", "candle_body_pct"},
        "RSI / MACD": {"RSI", "MACD", "MACD_signal", "MACD_hist", "macd_gap", "rsi_centered"},
        "Volume behavior": {"volume_change", "volume_zscore_20"},
    }
    counts = {group: 0 for group in group_map}
    counts["Other engineered"] = 0
    for column in columns:
        matched = False
        for group, names in group_map.items():
            if column in names:
                counts[group] += 1
                matched = True
                break
        if not matched:
            counts["Other engineered"] += 1

    group_df = pd.DataFrame(
        [{"Group": group, "Count": count} for group, count in counts.items() if count > 0]
    ).sort_values("Count", ascending=True)
    return group_df


def build_pipeline_flow_figure(details: dict):
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=18,
                line=dict(color="rgba(226,232,240,0.12)", width=1),
                label=[
                    f"Cleaned rows<br>{details['input_rows']:,}",
                    f"Engineered rows<br>{details['feature_rows']:,}",
                    f"Final train split<br>{details['train_rows']:,}",
                    f"Policy calibration<br>{details['calibration_rows']:,}",
                    f"Model-fit core<br>{details['model_train_rows']:,}",
                    f"Held-out test<br>{details['test_rows']:,}",
                ],
                color=["#0f172a", "#111f35", "#1d4ed8", "#0ea5e9", "#22c55e", "#f97316"],
            ),
            link=dict(
                source=[0, 1, 1, 2, 2],
                target=[1, 2, 5, 3, 4],
                value=[
                    details["feature_rows"],
                    details["train_rows"],
                    details["test_rows"],
                    details["calibration_rows"],
                    details["model_train_rows"],
                ],
                color=[
                    "rgba(56,189,248,0.30)",
                    "rgba(59,130,246,0.30)",
                    "rgba(249,115,22,0.35)",
                    "rgba(14,165,233,0.35)",
                    "rgba(34,197,94,0.30)",
                ],
            ),
        )
    )
    return style_plotly_figure(fig, "Data Journey Through the Modeling Pipeline", height=380)


def build_feature_group_figure(group_df: pd.DataFrame):
    fig = go.Figure(
        go.Bar(
            x=group_df["Count"],
            y=group_df["Group"],
            orientation="h",
            marker=dict(
                color=group_df["Count"],
                colorscale=[[0.0, "#38bdf8"], [0.55, "#3b82f6"], [1.0, "#f97316"]],
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=group_df["Count"],
            textposition="outside",
            hovertemplate="%{y}: %{x} features<extra></extra>",
        )
    )
    fig.update_xaxes(title="Features used")
    return style_plotly_figure(fig, "Feature Family Mix", height=330)


def build_rmse_comparison_figure(details: dict):
    labels = ["Model", "Naive baseline"]
    values = [details["price_rmse"], details["naive_rmse"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#38bdf8", "#f97316"],
            text=[f"${value:,.3f}" for value in values],
            textposition="outside",
            textfont=dict(color="#ffffff", size=14),
            hovertemplate="%{x}: %{y:.4f} $<extra></extra>",
        )
    )
    improvement = details["improvement_vs_naive"]
    fig.add_annotation(
        x=0.5,
        y=max(values) * 1.08 if max(values) > 0 else 0.1,
        text=f"Edge vs baseline: {improvement:+.3f} $",
        showarrow=False,
        font=dict(size=14, color="#ffffff"),
    )
    fig.update_yaxes(title="RMSE ($)")
    return style_plotly_figure(fig, "Error Comparison on the Validation Slice", height=340)


def build_outcome_donut_figure(details: dict):
    fig = style_plotly_figure(
        go.Figure(
        go.Pie(
            labels=["Wins", "Losses", "Flat filtered", "Not simulated"],
            values=[
                details["wins"],
                details["losses"],
                details["flat_filtered"],
                details["not_simulated"],
            ],
            hole=0.62,
            marker=dict(colors=["#22c55e", "#ef4444", "#64748b", "#38bdf8"]),
            textinfo="label+percent",
            textfont=dict(color="#ffffff", size=14),
            insidetextfont=dict(color="#ffffff", size=14),
            outsidetextfont=dict(color="#ffffff", size=14),
            textposition="outside",
            automargin=True,
            sort=False,
            direction="clockwise",
            domain=dict(x=[0.08, 0.92], y=[0.0, 0.78]),
            hovertemplate="%{label}: %{value}<extra></extra>",
        )
        ),
        "Trade Outcomes on Unseen Data",
        height=380,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=120, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            x=0.5,
            xanchor="center",
            font=dict(color="#ffffff", size=12),
        ),
        title=dict(
            text="Trade Outcomes on Unseen Data",
            y=0.98,
            x=0.02,
            xanchor="left",
            font=dict(color="#ffffff", size=24),
        ),
    )
    return fig


def build_equity_curve_figure(sim_df: pd.DataFrame, starting_cash: float):
    equity_df = pd.DataFrame({"Trade": [0], "Cash After": [starting_cash]})
    if not sim_df.empty:
        equity_df = pd.concat([equity_df, sim_df[["Trade", "Cash After"]]], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_df["Trade"],
            y=equity_df["Cash After"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=7, color="#f97316"),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.14)",
            name="Portfolio value",
            hovertemplate="Trade %{x}: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=starting_cash, line_dash="dash", line_color="rgba(226,232,240,0.35)")
    fig.update_xaxes(title="Executed trade number")
    fig.update_yaxes(title="Portfolio value ($)")
    return style_plotly_figure(fig, "Validation Equity Curve", height=340)


def build_prediction_quality_figure(prediction_df: pd.DataFrame):
    preview = prediction_df.tail(min(80, len(prediction_df)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preview.index,
            y=preview["ActualNextClose"],
            mode="lines",
            name="Actual next close",
            line=dict(color="#38bdf8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=preview.index,
            y=preview["PredictedNextClose"],
            mode="lines",
            name="Predicted next close",
            line=dict(color="#f97316", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=preview.index,
            y=preview["NaiveNextClose"],
            mode="lines",
            name="Naive baseline",
            line=dict(color="#94a3b8", width=2, dash="dash"),
        )
    )
    fig.update_yaxes(title="Next close ($)")
    return style_plotly_figure(fig, "Actual vs Predicted Next Close", height=360)


def initialize_dashboard_state():
    defaults = {
        "data_source": "Upload CSV",
        "ticker_symbol": "NVDA",
        "ticker_interval": "1h",
        "ticker_period": "1y",
        "forecast_horizon": 24,
        "simulation_starting_cash": 10000.0,
        "simulation_trade_count": 100,
        "raw_df": None,
        "source_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_evaluation_text(details: dict):
    return "\n".join(
        [
            "================ EVALUATION ================",
            "",
            f"Return RMSE: {details['return_rmse']:.6f}",
            f"Price RMSE: {details['price_rmse']:.4f} $",
            f"Naive RMSE: {details['naive_rmse']:.4f} $",
            f"Improvement vs naive: {details['improvement_vs_naive']:.4f} $",
            "",
            "Trading backtest on unseen validation period:",
            f"Trades taken: {details['trades_taken']}",
            f"Flat signals filtered out: {details['flat_filtered']}",
            f"Eligible trades not simulated: {details['not_simulated']}",
            f"Win / loss count: {details['wins']} / {details['losses']}",
            f"Ending cash from {details['starting_cash']:,.0f} $: {details['ending_cash']:.2f} $",
            f"Backtest return: {details['total_return_pct']:.2f} %",
        ]
    )


def build_live_evaluation_details(
    base_details: dict,
    trade_backtest_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    sim_summary: dict,
    starting_cash: float,
    trade_count: int,
):
    details = dict(base_details)
    eligible_trades = int((trade_backtest_df["Signal"] != "Flat").sum())
    flat_filtered = int((trade_backtest_df["Signal"] == "Flat").sum())
    not_simulated = max(eligible_trades - int(sim_summary["trades_executed"]), 0)
    details.update(
        {
            "trades_taken": int(sim_summary["trades_executed"]),
            "flat_filtered": flat_filtered,
            "trades_skipped": flat_filtered,
            "not_simulated": not_simulated,
            "wins": int(sim_summary["wins"]),
            "losses": int(sim_summary["losses"]),
            "ending_cash": float(sim_summary["final_cash"]),
            "net_profit": float(sim_summary["net_profit"]),
            "total_return_pct": float(sim_summary["total_return_pct"]),
            "win_rate_pct": (float(sim_summary["wins"]) / float(sim_summary["trades_executed"]) * 100)
            if sim_summary["trades_executed"]
            else 0.0,
            "avg_trade_return_pct": float(sim_df["Trade Return %"].mean()) if not sim_df.empty else 0.0,
            "avg_confidence_pct": float(sim_df["Confidence"].mean()) if not sim_df.empty else 0.0,
            "starting_cash": float(starting_cash),
            "selected_trade_count": int(trade_count),
            "eligible_trades": eligible_trades,
        }
    )
    details["eval_text"] = build_evaluation_text(details)
    return details


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
    test_sim_df, test_trade_summary = simulate_trades(
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

    feature_groups = [
        "Trend and momentum indicators",
        "Lagged prices and returns",
        "Rolling volatility",
        "RSI and MACD technical indicators",
        "Volume-based and trend-strength features",
    ]
    models_used = [
        "HistGradientBoostingRegressor for next-period return size",
        "HistGradientBoostingClassifier for up/down probability",
    ]
    signal_counts = (
        trade_backtest_df["Signal"]
        .value_counts()
        .reindex(["Long", "Short", "Flat"], fill_value=0)
        .to_dict()
    )
    prediction_df = trade_backtest_df[
        ["PredictedNextClose", "ActualNextClose", "Close", "PredictedSimpleReturn", "ActualSimpleReturn", "ProbUp", "Signal"]
    ].copy()
    prediction_df = prediction_df.rename(columns={"Close": "NaiveNextClose"})
    group_df = build_feature_group_breakdown(cols)

    pipeline_details = {
        "input_rows": int(len(df)),
        "feature_rows": int(len(feat)),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "model_train_rows": int(len(model_train)),
        "test_rows": int(len(test)),
        "features_used": int(len(cols)),
        "feature_groups": feature_groups,
        "models_used": models_used,
        "policy": trade_policy,
        "feature_group_df": group_df,
    }
    evaluation_details = {
        "return_rmse": rmse,
        "price_rmse": price_rmse,
        "naive_rmse": naive_rmse,
        "improvement_vs_naive": naive_rmse - price_rmse,
        "relative_edge_pct": ((naive_rmse - price_rmse) / naive_rmse * 100) if naive_rmse else 0.0,
        "trades_taken": int(test_trade_summary["trades_executed"]),
        "flat_filtered": int(test_trade_summary["skipped"]),
        "trades_skipped": int(test_trade_summary["skipped"]),
        "not_simulated": 0,
        "wins": int(test_trade_summary["wins"]),
        "losses": int(test_trade_summary["losses"]),
        "ending_cash": float(test_trade_summary["final_cash"]),
        "net_profit": float(test_trade_summary["net_profit"]),
        "total_return_pct": float(test_trade_summary["total_return_pct"]),
        "win_rate_pct": (float(test_trade_summary["wins"]) / float(test_trade_summary["trades_executed"]) * 100)
        if test_trade_summary["trades_executed"]
        else 0.0,
        "avg_trade_return_pct": float(test_sim_df["Trade Return %"].mean()) if not test_sim_df.empty else 0.0,
        "avg_confidence_pct": float(test_sim_df["Confidence"].mean()) if not test_sim_df.empty else 0.0,
        "signal_counts": signal_counts,
        "starting_cash": 10000.0,
        "selected_trade_count": int((trade_backtest_df["Signal"] != "Flat").sum()),
        "eligible_trades": int((trade_backtest_df["Signal"] != "Flat").sum()),
    }
    eval_text = build_evaluation_text(evaluation_details)
    evaluation_details["eval_text"] = eval_text

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
    return {
        "model": reg_model,
        "rmse": rmse,
        "forecast_df": forecast_df,
        "feature_frame": feat,
        "used_cols": cols,
        "pipeline_text": pipeline_text,
        "eval_text": eval_text,
        "top_features_df": feat_df,
        "trade_backtest_df": trade_backtest_df,
        "trade_policy": trade_policy,
        "pipeline_details": pipeline_details,
        "evaluation_details": evaluation_details,
        "test_sim_df": test_sim_df,
        "prediction_df": prediction_df,
    }


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


initialize_dashboard_state()

st.title("Finance Forecast Dashboard")
st.caption("Modern Streamlit dashboard with uploads, charts, indicators, and forecasting.")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Data Input Deck")
st.caption("Upload a CSV or load Yahoo Finance data from this panel. It remains visible without relying on the sidebar.")

top_left, top_right = st.columns([1.7, 1])
with top_left:
    active_source = st.radio(
        "Data source",
        ["Upload CSV", "Yahoo Finance ticker"],
        key="data_source",
        horizontal=True,
    )
with top_right:
    active_horizon = st.slider(
        "Forecast horizon",
        5,
        100,
        key="forecast_horizon",
    )

if st.session_state.get("last_data_source") != active_source:
    st.session_state["raw_df"] = None
    st.session_state["source_name"] = None
    st.session_state["last_data_source"] = active_source

if active_source == "Upload CSV":
    active_upload = st.file_uploader("Upload your CSV", type=["csv"], key="main_csv_uploader")
    if active_upload is not None:
        st.session_state["raw_df"] = read_uploaded_csv(active_upload.getvalue())
        st.session_state["source_name"] = active_upload.name
else:
    ticker_col, interval_col, period_col = st.columns([1.15, 1, 1])
    with ticker_col:
        active_ticker = st.text_input("Ticker", key="ticker_symbol")
    with interval_col:
        active_interval = st.selectbox("Interval", ["1h", "1d", "30m", "15m"], key="ticker_interval")
    with period_col:
        if active_interval == "1h":
            active_period = "730d"
            st.text_input("Period", value=active_period, disabled=True, key="ticker_period_locked")
            st.caption("Hourly Yahoo Finance data uses the maximum 730-day lookback.")
        else:
            active_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], key="ticker_period")

    if st.button("Load market data", type="primary", use_container_width=True):
        st.session_state["raw_df"] = fetch_yfinance_data(active_ticker, active_period, active_interval)
        st.session_state["source_name"] = f"{active_ticker} ({active_period}, {active_interval})"

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------- Sidebar -----------------------------
st.sidebar.title("Quick Controls")
st.sidebar.caption("The full input deck is shown at the top of the page.")
source = st.session_state.get("data_source", "Upload CSV")

raw_df = st.session_state.get("raw_df")
source_name = st.session_state.get("source_name")

if source == "Upload CSV":
    uploaded = None
    if uploaded is not None:
        raw_df = read_uploaded_csv(uploaded.getvalue())
        source_name = uploaded.name
else:
    ticker = st.session_state.get("ticker_symbol", "NVDA")
    interval = st.session_state.get("ticker_interval", "1h")

    if interval == "1h":
        period = "730d"
        st.sidebar.caption("Hourly Yahoo Finance data uses the maximum 730-day lookback.")
    else:
        period = st.session_state.get("ticker_period", "1y")

    if False:
        raw_df = fetch_yfinance_data(ticker, period, interval)
        source_name = f"{ticker} ({period}, {interval})"

horizon = st.session_state.get("forecast_horizon", 24)

st.sidebar.caption(f"Current mode: {source}")

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

forecast_bundle = None
forecast_error = None
shared_sim_df = pd.DataFrame()
shared_sim_summary = None
shared_evaluation_details = None
eligible_trades = 0
try:
    forecast_bundle = recursive_forecast(
        df[["Open", "High", "Low", "Close", "Volume"]].copy(), horizon
    )
except Exception as e:
    forecast_error = e

if forecast_bundle is not None:
    trade_backtest_df = forecast_bundle["trade_backtest_df"]
    trade_policy = forecast_bundle["trade_policy"]
    eligible_trades = int((trade_backtest_df["Signal"] != "Flat").sum())

    if eligible_trades > 0:
        default_trade_count = min(100, eligible_trades)
        if (
            "simulation_trade_count" not in st.session_state
            or st.session_state["simulation_trade_count"] < 1
            or st.session_state["simulation_trade_count"] > eligible_trades
        ):
            st.session_state["simulation_trade_count"] = default_trade_count

        shared_starting_cash = float(st.session_state.get("simulation_starting_cash", 10000.0))
        shared_trade_count = int(st.session_state.get("simulation_trade_count", default_trade_count))
        shared_sim_df, shared_sim_summary = simulate_trades(
            trade_backtest_df,
            shared_starting_cash,
            shared_trade_count,
            position_size=trade_policy["position_size"],
        )
        shared_evaluation_details = build_live_evaluation_details(
            forecast_bundle["evaluation_details"],
            trade_backtest_df,
            shared_sim_df,
            shared_sim_summary,
            shared_starting_cash,
            shared_trade_count,
        )
    else:
        shared_evaluation_details = dict(forecast_bundle["evaluation_details"])

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
    if forecast_bundle is not None:
        forecast_df = forecast_bundle["forecast_df"]
        rmse = forecast_bundle["rmse"]
        used_cols = forecast_bundle["used_cols"]
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
    else:
        st.error(f"Forecasting failed: {forecast_error}")
    st.markdown('</div>', unsafe_allow_html=True)
with data_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.tail(200), use_container_width=True)
    csv = df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download cleaned data as CSV", csv, file_name="cleaned_data.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with pipeline_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    if forecast_bundle is not None:
        pipeline_details = forecast_bundle["pipeline_details"]
        policy = pipeline_details["policy"]
        render_hero_panel(
            "Pipeline Intelligence",
            "A full ML workflow, presented like a product demo",
            (
                f"The pipeline starts with {pipeline_details['input_rows']:,} cleaned market rows, engineers "
                f"{pipeline_details['features_used']} predictive signals, calibrates a trading policy, and then "
                f"tests the model on {pipeline_details['test_rows']:,} unseen rows."
            ),
            [
                f"{pipeline_details['features_used']} features",
                f"{pipeline_details['train_rows']:,} train rows",
                f"{pipeline_details['test_rows']:,} test rows",
                f"{pipeline_details['calibration_rows']:,} calibration rows",
            ],
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cleaned rows", f"{pipeline_details['input_rows']:,}")
        m2.metric("Feature rows", f"{pipeline_details['feature_rows']:,}")
        m3.metric("Model-fit core", f"{pipeline_details['model_train_rows']:,}")
        m4.metric("Signals engineered", f"{pipeline_details['features_used']}")

        left, right = st.columns([1.35, 1])
        with left:
            st.plotly_chart(build_pipeline_flow_figure(pipeline_details), use_container_width=True)
        with right:
            st.markdown('<div class="section-label">Model stack</div>', unsafe_allow_html=True)
            render_mini_panel(
                "Learners in the ensemble",
                "<br>".join(f"&bull; {item}" for item in pipeline_details["models_used"]),
            )
            st.markdown('<div class="section-label">Trading policy</div>', unsafe_allow_html=True)
            render_mini_panel(
                "Optimized execution filters",
                (
                    f"Minimum predicted move: {policy['return_threshold'] * 100:.3f}%<br>"
                    f"Directional probability: {policy['prob_threshold']:.2f}+<br>"
                    f"Capital deployed per trade: {policy['position_size'] * 100:.0f}%<br>"
                    "Trend gate: MA and MACD must agree before a trade is allowed."
                ),
            )

        lower_left, lower_right = st.columns([1.1, 1])
        with lower_left:
            st.plotly_chart(
                build_feature_group_figure(pipeline_details["feature_group_df"]),
                use_container_width=True,
            )
        with lower_right:
            st.markdown('<div class="section-label">Feature families</div>', unsafe_allow_html=True)
            render_mini_panel(
                "What the model actually learns from",
                "<br>".join(f"&bull; {item}" for item in pipeline_details["feature_groups"]),
            )
    else:
        st.error(f"Forecasting failed: {forecast_error}")
    st.markdown('</div>', unsafe_allow_html=True)

with evaluation_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    if forecast_bundle is not None:
        evaluation_details = shared_evaluation_details or forecast_bundle["evaluation_details"]
        summary_title = (
            "Model beats the naive baseline"
            if evaluation_details["improvement_vs_naive"] >= 0
            else "Baseline still edges out the model"
        )
        summary_body = (
            f"Validation price RMSE landed at ${evaluation_details['price_rmse']:.3f} versus "
            f"${evaluation_details['naive_rmse']:.3f} for the naive carry-forward baseline. "
            f"The unseen-period backtest finished with ${evaluation_details['ending_cash']:,.2f} "
            f"from a ${evaluation_details['starting_cash']:,.0f} starting bankroll."
        )
        render_hero_panel(
            "Evaluation Snapshot",
            summary_title,
            summary_body,
            [
                f"{evaluation_details['win_rate_pct']:.1f}% win rate",
                f"{evaluation_details['trades_taken']} trades",
                f"{evaluation_details['total_return_pct']:+.2f}% backtest return",
                f"{evaluation_details['avg_confidence_pct']:.1f}% avg confidence",
            ],
        )

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Return RMSE", f"{evaluation_details['return_rmse']:.6f}")
        e2.metric("Price RMSE", f"${evaluation_details['price_rmse']:.4f}")
        e3.metric(
            "Edge vs naive",
            f"{evaluation_details['improvement_vs_naive']:+.4f} $",
            delta=f"{evaluation_details['relative_edge_pct']:+.2f}%",
        )
        e4.metric(
            "Ending cash",
            f"${evaluation_details['ending_cash']:,.2f}",
            delta=f"${evaluation_details['net_profit']:,.2f}",
        )

        eval_left, eval_right = st.columns(2)
        with eval_left:
            st.caption(
                f"Live view uses the current simulation setup: ${evaluation_details['starting_cash']:,.0f} starting cash "
                f"and up to {evaluation_details['selected_trade_count']} executed trades."
            )
            st.plotly_chart(build_rmse_comparison_figure(evaluation_details), use_container_width=True)
            st.plotly_chart(
                build_prediction_quality_figure(forecast_bundle["prediction_df"]),
                use_container_width=True,
            )
        with eval_right:
            st.plotly_chart(build_outcome_donut_figure(evaluation_details), use_container_width=True)
            st.plotly_chart(
                build_equity_curve_figure(
                    shared_sim_df,
                    evaluation_details["starting_cash"],
                ),
                use_container_width=True,
            )

        signal_counts = evaluation_details["signal_counts"]
        insight_left, insight_right = st.columns(2)
        with insight_left:
            render_mini_panel(
                "Backtest reading",
                (
                    f"Executed trades: {evaluation_details['trades_taken']}<br>"
                    f"Flat signals filtered out: {evaluation_details['flat_filtered']}<br>"
                    f"Eligible trades not simulated: {evaluation_details['not_simulated']}<br>"
                    f"Wins / losses: {evaluation_details['wins']} / {evaluation_details['losses']}<br>"
                    f"Average trade return: {evaluation_details['avg_trade_return_pct']:+.2f}%"
                ),
            )
        with insight_right:
            render_mini_panel(
                "Signal mix",
                (
                    f"Long setups: {signal_counts['Long']}<br>"
                    f"Short setups: {signal_counts['Short']}<br>"
                    f"Filtered out: {signal_counts['Flat']}<br>"
                    "This makes the strategy feel explainable instead of purely black-box."
                ),
            )
    else:
        st.error(f"Forecasting failed: {forecast_error}")
    st.markdown('</div>', unsafe_allow_html=True)

with features_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    if forecast_bundle is not None:
        top_features_df = forecast_bundle["top_features_df"]
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
    else:
        st.error(f"Forecasting failed: {forecast_error}")
    st.markdown('</div>', unsafe_allow_html=True)

with trading_tab:
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    if forecast_bundle is not None:
        trade_backtest_df = forecast_bundle["trade_backtest_df"]
        trade_policy = forecast_bundle["trade_policy"]
        if eligible_trades == 0:
            st.warning("The optimized strategy skipped every validation-period setup, so there are no trades to simulate.")
        else:
            max_trades = eligible_trades
            default_trades = min(100, max_trades)
            c1, c2 = st.columns(2)
            with c1:
                starting_cash = st.number_input(
                    "Starting cash ($)",
                    min_value=100.0,
                    step=500.0,
                    key="simulation_starting_cash",
                )
            with c2:
                trade_count = st.slider(
                    "Number of trades to simulate",
                    1,
                    max_trades,
                    min(int(st.session_state.get("simulation_trade_count", default_trades)), max_trades),
                    key="simulation_trade_count",
                )

            st.caption(
                f"Simulation uses the optimized policy from the training data: minimum predicted move "
                f"{trade_policy['return_threshold'] * 100:.3f}%, minimum direction probability "
                f"{trade_policy['prob_threshold']:.2f}, and {trade_policy['position_size'] * 100:.0f}% of capital per trade. "
                "The Evaluation tab mirrors these exact settings."
            )
            sim_df = shared_sim_df
            sim_summary = shared_sim_summary
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
    else:
        st.error(f"Forecasting failed: {forecast_error}")
    st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.subheader("Summary statistics for the loaded dataset")
st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
