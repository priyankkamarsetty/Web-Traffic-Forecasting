# 🌐 Web Traffic Forecasting Project

A web traffic forecasting system that compares **Random Forest (LSTM alternative)** and **ARIMA** models to predict website user traffic patterns. This project demonstrates both traditional statistical methods and modern machine learning approaches for time series forecasting.

## 🎯 Project Overview

This project implements a time series forecasting pipeline that:

1. **Analyzes web traffic data** with time series visualization
2. **Trains two different models**: Random Forest (machine learning) and ARIMA (statistical)
3. **Compares model performance** using RMSE metrics
4. **Generates predictions** and visualizations
5. **Provides interactive dashboard** through Streamlit

## 🏆 Key Findings

**ARIMA typically performs better than Random Forest** for this web traffic forecasting because:
- ARIMA is specifically designed for time series data
- It handles trends and seasonality more effectively
- Traditional statistical methods can be more reliable for certain patterns

## 📊 Features

### Data Analysis
- ✅ **Time Series Visualization**: Interactive plots showing traffic patterns
- ✅ **Data Overview**: Statistics and metrics display
- ✅ **Model Comparison**: Side-by-side performance evaluation

### Model Implementation
- ✅ **Random Forest Model**: Machine learning approach with feature engineering
- ✅ **ARIMA Model**: Traditional statistical method with automatic parameter selection
- ✅ **Feature Engineering**: Time-based features, lags, and rolling averages
- ✅ **Future Forecasting**: Predictions for test data

### Visualization & Dashboard
- ✅ **Interactive Streamlit Dashboard**: Modern web interface
- ✅ **Real-time Metrics**: RMSE comparison and improvement percentages
- ✅ **Interactive Charts**: Plotly-based visualizations
- ✅ **Performance Comparison**: Visual model comparison

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```

### 3. Use the Application
1. Upload your CSV file with `Date` and `Users` columns
2. Click "Run Training & Analysis" to see model comparisons
3. View the results and performance metrics

## 📁 Project Structure

```
web-traffic-forecasting/
├── app.py                         # Full-featured Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore file
└── src/
    ├── __init__.py
    ├── config.py                 # Configuration settings
    ├── data.py                   # Data loading and preprocessing
    ├── analysis.py               # Time series analysis functions
    ├── lstm.py                   # Random Forest model implementation
    ├── arima.py                  # ARIMA model implementation
    ├── train_and_forecast.py     # Main training pipeline
    └── utils.py                  # Utility functions
```

## 📈 Data Format

Your CSV file should have the following structure:

```csv
Date,Users
2020-01-01,650
2020-01-02,680
2020-01-03,720
...
```

- **Date**: Date in YYYY-MM-DD format
- **Users**: Number of users (integer)

## 🔧 Configuration

Key settings in `src/config.py`:

```python
TRAIN_RATIO: float = 0.65      # 65% for training, 35% for testing
TIME_STEPS: int = 100          # Feature window size
FORECAST_DAYS: int = 30        # Future prediction days
RANDOM_SEED: int = 42          # For reproducibility
```

## 📊 Model Details

### Random Forest Model
- **Algorithm**: Random Forest Regressor with 100 trees
- **Features**: Time index, lag features, rolling averages, day of week
- **Strengths**: Captures non-linear patterns, handles multiple features

### ARIMA Model
- **Method**: Automatic parameter selection (p, d, q)
- **Stationarity**: Automatic differencing if needed
- **Features**: AIC-based model selection
- **Strengths**: Interpretable, handles trends and seasonality

## 🎯 Analysis Pipeline

1. **Data Generation**: Create sample web traffic data with trend and seasonality
2. **Data Visualization**: Plot time series and show statistics
3. **Feature Engineering**: Create time-based features and lags
4. **Model Training**: Train Random Forest and ARIMA models
5. **Performance Comparison**: Calculate RMSE and show results
6. **Visualization**: Display predictions vs actual values

## 📊 Expected Results

The project typically shows that:
- **ARIMA achieves lower RMSE** than Random Forest
- **Statistical methods can be more reliable** for certain time series patterns
- **Feature engineering** significantly improves model performance

## 🛠️ Technical Requirements

- Python 3.8+
- Scikit-learn 1.5+
- Statsmodels 0.14+
- Streamlit 1.37+
- Pandas 2.2+
- NumPy 1.26+
- Plotly 5.22+
- Matplotlib 3.9+

## 🚀 Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new models
- Improving feature engineering
- Enhancing visualizations
- Adding more evaluation metrics

## 📝 License

This project is open source and available under the MIT License.

---
