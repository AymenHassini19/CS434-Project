import pandas as pd

def VWAP(close: pd.Series, volume: pd.Series):
    return (close * volume).cumsum() / volume.cumsum()
