import pandas as pd

def EMA(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average
    Args:
        series: pd.Series of prices
        span: EMA span (number of periods)
    Returns:
        pd.Series of EMA values
    """
    return series.ewm(span=span, adjust=False).mean()