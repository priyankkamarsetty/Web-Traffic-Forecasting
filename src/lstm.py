from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from .config import CFG

# Use simple models instead of TensorFlow for speed
TENSORFLOW_AVAILABLE = False

def _make_sequences(values: np.ndarray, time_steps: int):
    X, y = [], []
    for i in range(time_steps, len(values)):
        X.append(values[i-time_steps:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def build_model(input_timesteps: int) -> object:
    # Use Random Forest for better performance than linear regression
    return RandomForestRegressor(n_estimators=100, random_state=42)

def fit_lstm(train_series: pd.Series, test_series: pd.Series, time_steps: int = CFG.TIME_STEPS):
    time_steps = min(time_steps, max(10, len(train_series)//3))
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_values = np.concatenate([train_series.values, test_series.values]).reshape(-1, 1)
    scaled = scaler.fit_transform(all_values).flatten()

    train_scaled = scaled[:len(train_series)]
    test_scaled  = scaled[len(train_series):]

    # Use Random Forest with time-based features
    print("🧠 Training Random Forest Model (LSTM alternative)...")
    
    # Create time-based features
    def create_features(series):
        features = []
        max_features = 6  # Maximum number of features we'll create
        
        for i in range(len(series)):
            feature_row = []
            # Time index
            feature_row.append(i)
            # Day of week (if we have enough data)
            if len(series) > 7:
                feature_row.append(i % 7)
            else:
                feature_row.append(0)
            # Rolling averages
            if i >= 3:
                feature_row.append(series.iloc[i-3:i].mean())
            else:
                feature_row.append(series.iloc[:i+1].mean() if i > 0 else series.iloc[0])
            if i >= 7:
                feature_row.append(series.iloc[i-7:i].mean())
            else:
                feature_row.append(series.iloc[:i+1].mean() if i > 0 else series.iloc[0])
            # Lag features
            if i >= 1:
                feature_row.append(series.iloc[i-1])
            else:
                feature_row.append(series.iloc[0])
            if i >= 2:
                feature_row.append(series.iloc[i-2])
            else:
                feature_row.append(series.iloc[0])
            
            # Ensure all rows have the same number of features
            while len(feature_row) < max_features:
                feature_row.append(0)
            
            features.append(feature_row[:max_features])
        return np.array(features)
    
    # Prepare training data
    X_train = create_features(train_series)
    y_train = train_series.values
    
    # Train model
    model = build_model(time_steps)
    model.fit(X_train, y_train)
    
    # Training predictions
    train_pred = model.predict(X_train)
    
    # Test predictions
    X_test = create_features(test_series)
    test_pred = model.predict(X_test)
    
    # Future predictions (extend the series)
    extended_series = pd.concat([train_series, test_series])
    future_preds = []
    for i in range(CFG.FORECAST_DAYS):
        # Create features for next prediction
        last_idx = len(extended_series) + i
        feature_row = [last_idx]
        if last_idx >= 7:
            feature_row.append(last_idx % 7)
        if last_idx >= 3:
            feature_row.append(extended_series.iloc[-3:].mean())
        if last_idx >= 7:
            feature_row.append(extended_series.iloc[-7:].mean())
        if last_idx >= 1:
            feature_row.append(extended_series.iloc[-1])
        if last_idx >= 2:
            feature_row.append(extended_series.iloc[-2])
        
        # Pad with zeros if needed
        while len(feature_row) < X_train.shape[1]:
            feature_row.append(0)
        
        pred = model.predict([feature_row])[0]
        future_preds.append(pred)
        # Add prediction to series for next iteration
        extended_series = pd.concat([extended_series, pd.Series([pred])])
    
    return {
        "model": model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "future": np.array(future_preds),
        "time_steps": time_steps
    }


