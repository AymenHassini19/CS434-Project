import pandas as pd
import numpy as np

def ADX(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    """
    Average Directional Index
    """
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx
