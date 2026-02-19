import pandas as pd

def ATR(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr
