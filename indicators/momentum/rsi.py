import pandas as pd
import numpy as np

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index
    Args:
        series: pd.Series of prices
        period: RSI period
    Returns:
        pd.Series of RSI values
    """
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
