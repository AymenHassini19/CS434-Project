import pandas as pd

def MACD(series: pd.Series, fast=12, slow=26, signal=9):
    """
    Compute MACD and signal line
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line
