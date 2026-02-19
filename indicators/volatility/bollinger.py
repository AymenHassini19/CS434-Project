import pandas as pd

def Bollinger_Bands(series: pd.Series, window: int = 20, num_std: float = 2):
    """
    Bollinger Bands
    Args:
        series: pd.Series of prices
        window: rolling window
        num_std: number of standard deviations for bands
    Returns:
        upper_band, lower_band: pd.Series
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, lower
