import numpy as np


# BEGIN SYNCED METRICS: mape
# Canonical source of this function. kaggle/notebooks/03_statistical_models.ipynb and
# kaggle/notebooks/04_ml_models.ipynb run in an isolated Kaggle environment and can't
# `import src.utils.metrics`, so each embeds a byte-identical copy between matching
# BEGIN/END markers. Don't hand-edit the notebook copies - after changing this function,
# run `python scripts/sync_notebook_metrics.py` to propagate it, or
# `python scripts/sync_notebook_metrics.py --check` to see if they've drifted (this is
# also what tests/test_notebook_metrics_sync.py runs, so drift fails the test suite).
def mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    if not mask.any():
        # every actual in this batch is zero - MAPE is undefined here, not 0 or inf.
        # Returning NaN (rather than letting np.mean hit an empty slice and warn) lets
        # downstream pandas .mean()/.groupby().mean() calls skip it cleanly via skipna.
        return np.nan
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
# END SYNCED METRICS: mape


# BEGIN SYNCED METRICS: wape
def wape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        # all-zero actuals: division would silently produce inf, which (unlike NaN) is
        # NOT skipped by pandas .mean(), so it would poison any aggregate it touches.
        return np.nan
    return np.sum(np.abs(actual - forecast)) / denom * 100
# END SYNCED METRICS: wape


# BEGIN SYNCED METRICS: mase
def mase(actual, forecast, train_series, seasonal_period=7):
    actual, forecast = np.array(actual), np.array(forecast)
    train_series = np.array(train_series)
    naive_errors = np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period])
    scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return np.mean(np.abs(actual - forecast)) / scale if scale != 0 else np.nan
# END SYNCED METRICS: mase
