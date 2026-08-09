import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import mape, mase, wape


def test_mape_perfect_forecast():
    assert mape([10, 20, 30], [10, 20, 30]) == 0


def test_mape_ignores_zero_actuals():
    assert mape([0, 10], [5, 12]) == mape([10], [12])


def test_wape_basic():
    assert wape([10, 10], [12, 8]) == 20.0


def test_mase_naive_baseline_equals_one():
    train = list(range(1, 100))
    actual = [50, 51, 52]
    forecast = [43, 44, 45]
    assert abs(mase(actual, forecast, train) - 1.0) < 1e-9


def test_mase_perfect_forecast_is_zero():
    train = list(range(1, 30))
    assert mase([5, 5, 5], [5, 5, 5], train) == 0


def test_mape_all_zero_actuals_returns_nan():
    # Previously: masked array was empty -> np.mean hit an empty slice and returned NaN
    # via a runtime warning instead of a deliberate, documented return value.
    assert np.isnan(mape([0, 0, 0], [1, 2, 3]))


def test_wape_all_zero_actuals_returns_nan():
    # Previously: division by zero silently produced inf, which pandas .mean() does NOT
    # skip the way it skips NaN - one degenerate all-zero-actual fold could poison an
    # entire averaged metric downstream without raising anything.
    assert np.isnan(wape([0, 0], [1, 2]))