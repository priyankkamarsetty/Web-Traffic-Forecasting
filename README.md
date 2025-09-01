# 🌐 Web Traffic Forecasting Project

A web traffic forecasting system that compares **Random Forest** and **Linear Regression** models to predict website user traffic patterns. This project demonstrates time series forecasting concepts with a simple, fast implementation.

## 🎯 Project Overview

This project implements a time series forecasting pipeline that:

1. **Analyzes web traffic data** with time series visualization
2. **Trains two different models**: Random Forest and Linear Regression
3. **Compares model performance** using RMSE metrics
4. **Generates predictions** and visualizations
5. **Provides interactive dashboard** through Streamlit

## 🏆 Key Findings

**Random Forest typically performs better than Linear Regression** for web traffic forecasting because:
- Random Forest can capture non-linear patterns in the data
- It handles multiple features and interactions effectively
- Machine learning models excel at learning from historical patterns

## 📊 Features

### Data Analysis
- ✅ **Time Series Visualization**: Interactive plots showing traffic patterns
- ✅ **Data Overview**: Statistics and metrics display
- ✅ **Model Comparison**: Side-by-side performance evaluation

### Model Implementation
- ✅ **Random Forest Model**: Machine learning approach with feature engineering
- ✅ **Linear Regression Model**: Traditional statistical method
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
- **Algorithm**: Random Forest Regressor with 50-100 trees
- **Features**: Time index, lag features, rolling averages, day of week
- **Strengths**: Captures non-linear patterns, handles multiple features

### Linear Regression Model
- **Algorithm**: Linear Regression with feature engineering
- **Features**: Same feature set as Random Forest
- **Strengths**: Fast, interpretable, good baseline

## 🎯 Analysis Pipeline

1. **Data Generation**: Create sample web traffic data with trend and seasonality
2. **Data Visualization**: Plot time series and show statistics
3. **Feature Engineering**: Create time-based features and lags
4. **Model Training**: Train Random Forest and Linear Regression
5. **Performance Comparison**: Calculate RMSE and show results
6. **Visualization**: Display predictions vs actual values

## 📊 Expected Results

The project typically shows that:
- **Random Forest achieves lower RMSE** than Linear Regression
- **Machine learning captures complex patterns** better than linear methods
- **Feature engineering** significantly improves model performance

## 🛠️ Technical Requirements

- Python 3.8+
- Streamlit 1.37+
- Pandas 2.2+
- NumPy 1.26+
- Scikit-learn 1.5+
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

**Built with ❤️ for time series forecasting and machine learning education**
