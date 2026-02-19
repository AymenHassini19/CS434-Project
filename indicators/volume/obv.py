import pandas as pd
import numpy as np

def OBV(close: pd.Series, volume: pd.Series):
    direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
    obv = (volume * direction).cumsum()
    return obv
