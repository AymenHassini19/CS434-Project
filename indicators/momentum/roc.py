import pandas as pd

def ROC(series: pd.Series, period=12):
    return (series - series.shift(period)) / series.shift(period)
