import numpy as np


def mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    if not mask.any():
        # every actual in this batch is zero - MAPE is undefined here, not 0 or inf.
        # Returning NaN (rather than letting np.mean hit an empty slice and warn) lets
        # downstream pandas .mean()/.groupby().mean() calls skip it cleanly via skipna.
        return np.nan
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100


def wape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        # all-zero actuals: division would silently produce inf, which (unlike NaN) is
        # NOT skipped by pandas .mean(), so it would poison any aggregate it touches.
        return np.nan
    return np.sum(np.abs(actual - forecast)) / denom * 100


def mase(actual, forecast, train_series, seasonal_period=7):
    actual, forecast = np.array(actual), np.array(forecast)
    train_series = np.array(train_series)
    naive_errors = np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return np.mean(np.abs(actual - forecast)) / scale if scale != 0 else np.nan