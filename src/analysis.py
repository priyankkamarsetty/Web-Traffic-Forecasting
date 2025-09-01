import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from .config import CFG

def visualize_time_series(df: pd.DataFrame, save_path: str = None):
    """Visualize the time series data"""
    plt.figure(figsize=(15, 10))
    
    # Main time series plot
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Users'], linewidth=2)
    plt.title('Web Traffic Time Series', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Users')
    plt.grid(True, alpha=0.3)
    
    # Rolling statistics
    plt.subplot(3, 1, 2)
    rolling_mean = df['Users'].rolling(window=7).mean()
    rolling_std = df['Users'].rolling(window=7).std()
    plt.plot(df['Date'], df['Users'], label='Original', alpha=0.7)
    plt.plot(df['Date'], rolling_mean, label='7-day Rolling Mean', linewidth=2)
    plt.plot(df['Date'], rolling_std, label='7-day Rolling Std', linewidth=2)
    plt.title('Rolling Statistics', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution
    plt.subplot(3, 1, 3)
    plt.hist(df['Users'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Users', fontsize=12)
    plt.xlabel('Number of Users')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to prevent display issues

def test_stationarity(series: pd.Series, title: str = "Stationarity Test"):
    """Test if the time series is stationary using Augmented Dickey-Fuller test"""
    try:
        result = adfuller(series.dropna())
        
        print(f"\n{title}")
        print('=' * 50)
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print('Critical values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        if result[1] <= 0.05:
            print("✅ Series is stationary (p-value <= 0.05)")
            return True
        else:
            print("❌ Series is not stationary (p-value > 0.05)")
            return False
    except Exception as e:
        print(f"⚠️ Error in stationarity test: {e}")
        return False

def plot_correlation_analysis(df: pd.DataFrame, save_path: str = None):
    """Plot correlation and autocorrelation charts"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Autocorrelation Function (ACF)
        plt.subplot(2, 2, 1)
        plot_acf(df['Users'].dropna(), lags=40, ax=plt.gca())
        plt.title('Autocorrelation Function (ACF)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Partial Autocorrelation Function (PACF)
        plt.subplot(2, 2, 2)
        plot_pacf(df['Users'].dropna(), lags=40, ax=plt.gca())
        plt.title('Partial Autocorrelation Function (PACF)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Seasonal decomposition
        plt.subplot(2, 2, 3)
        decomposition = seasonal_decompose(df['Users'], period=7, extrapolate_trend='freq')
        plt.plot(df['Date'], decomposition.trend, linewidth=2)
        plt.title('Trend Component', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Seasonal component
        plt.subplot(2, 2, 4)
        plt.plot(df['Date'], decomposition.seasonal, linewidth=2)
        plt.title('Seasonal Component (7-day)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️ Error in correlation analysis: {e}")

def analyze_data(df: pd.DataFrame, output_dir: str):
    """Complete data analysis pipeline"""
    print("🔍 Starting Time Series Analysis...")
    
    # 1. Visualize the Time Series Data
    print("\n1. Visualizing Time Series Data...")
    visualize_time_series(df, f"{output_dir}/time_series_analysis.png")
    
    # 2. Test for stationarity
    print("\n2. Testing for Stationarity...")
    is_stationary = test_stationarity(df['Users'], "Original Series")
    
    if not is_stationary:
        print("\nMaking series stationary by differencing...")
        diff_series = df['Users'].diff().dropna()
        test_stationarity(diff_series, "Differenced Series")
    
    # 3. Plot correlation and autocorrelation charts
    print("\n3. Analyzing Correlations...")
    plot_correlation_analysis(df, f"{output_dir}/correlation_analysis.png")
    
    # 4. Summary statistics
    print("\n4. Summary Statistics:")
    print('=' * 50)
    print(df['Users'].describe())
    
    return df
