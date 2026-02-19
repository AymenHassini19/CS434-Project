import pandas as pd

def SMA(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average
    Args:
        series: pd.Series of prices
        window: rolling window size
    Returns:
        pd.Series of SMA values
    """
    return series.rolling(window=window).mean()
