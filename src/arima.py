from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from .config import CFG

def _make_stationary(series: pd.Series) -> tuple[pd.Series, int]:
    """Make series stationary by differencing if needed"""
    d = 0
    current_series = series.copy()
    
    # Test for stationarity
    result = adfuller(current_series)
    p_value = result[1]
    
    # If not stationary, difference until it becomes stationary
    while p_value > 0.05 and d < 2:
        d += 1
        current_series = current_series.diff().dropna()
        if len(current_series) < 10:  # Too few observations
            break
        result = adfuller(current_series)
        p_value = result[1]
    
    return current_series, d

def _find_best_arima_params(series: pd.Series, max_p: int = 3, max_q: int = 3) -> tuple[int, int]:
    """Find best ARIMA parameters using AIC"""
    best_aic = float('inf')
    best_p, best_q = 1, 1
    
    # Grid search for best p, q parameters
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(series, order=(p, 0, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_p, best_q = p, q
            except:
                continue
    
    return best_p, best_q

def fit_arima(train_series: pd.Series, test_series: pd.Series):
    """Fit ARIMA model using statsmodels"""
    # Make series stationary
    stationary_series, d = _make_stationary(train_series)
    
    # Find best ARIMA parameters
    p, q = _find_best_arima_params(stationary_series)
    
    # Fit ARIMA model
    model = ARIMA(train_series, order=(p, d, q))
    fitted_model = model.fit()
    
    # Generate predictions
    train_pred = fitted_model.predict(start=0, end=len(train_series)-1)
    
    # For test predictions, we need to update the model with test data
    # or use the fitted model to forecast
    test_pred = fitted_model.forecast(steps=len(test_series))
    
    # For future predictions, we need to update the model with all data
    full_series = pd.concat([train_series, test_series], axis=0)
    full_model = ARIMA(full_series, order=(p, d, q))
    full_fitted_model = full_model.fit()
    future = full_fitted_model.forecast(steps=CFG.FORECAST_DAYS)
    
    return {
        "model": fitted_model,
        "train_pred": np.asarray(train_pred),
        "test_pred": np.asarray(test_pred),
        "future": np.asarray(future),
    }
