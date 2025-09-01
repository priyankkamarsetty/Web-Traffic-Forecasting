from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CFG
from .utils import ensure_dir, rmse, save_json
from .data import load_or_create_data, train_test_split_series
from .lstm import fit_lstm
from .arima import fit_arima
from .analysis import analyze_data
import numpy as np

def run(data_path: str = CFG.DATA_PATH):
    try:
        out_dir = ensure_dir(CFG.OUTPUT_DIR)
        df = load_or_create_data(data_path)
        
        # Run comprehensive data analysis
        print("🚀 Starting Web Traffic Forecasting Project")
        print("=" * 60)
        analyze_data(df, out_dir)
        
        # Split data for training
        train_df, test_df = train_test_split_series(df, CFG.TRAIN_RATIO)
        train_y = train_df["Users"].reset_index(drop=True)
        test_y  = test_df["Users"].reset_index(drop=True)

        print(f"\n📊 Data Split: {len(train_df)} training samples, {len(test_df)} test samples")
        
        # Train LSTM Model
        print("\n🧠 Training LSTM Model...")
        lstm_out = fit_lstm(train_y, test_y, CFG.TIME_STEPS)
        lstm_train_true = train_y.iloc[lstm_out["time_steps"]:]
        lstm_train_pred = lstm_out["train_pred"]
        lstm_rmse_train = rmse(lstm_train_true, lstm_train_pred)
        lstm_rmse_test  = rmse(test_y, lstm_out["test_pred"])

        # Train ARIMA Model
        print("\n📈 Training ARIMA Model...")
        arima_out = fit_arima(train_y, test_y)
        arima_rmse_train = rmse(train_y, arima_out["train_pred"])
        arima_rmse_test  = rmse(test_y, arima_out["test_pred"])

        # Model comparison
        print("\n📊 Model Performance Comparison:")
        print("=" * 50)
        print(f"LSTM  - Train RMSE: {lstm_rmse_train:.2f}, Test RMSE: {lstm_rmse_test:.2f}")
        print(f"ARIMA - Train RMSE: {arima_rmse_train:.2f}, Test RMSE: {arima_rmse_test:.2f}")
        
        if lstm_rmse_test < arima_rmse_test:
            print("🏆 LSTM performs better than ARIMA!")
        else:
            print("🏆 ARIMA performs better than LSTM!")

        metrics = {
            "lstm": {"rmse_train": lstm_rmse_train, "rmse_test": lstm_rmse_test},
            "arima": {"rmse_train": arima_rmse_train, "rmse_test": arima_rmse_test},
            "comparison": {
                "lstm_better": lstm_rmse_test < arima_rmse_test,
                "improvement_percent": abs(lstm_rmse_test - arima_rmse_test) / arima_rmse_test * 100
            }
        }
        save_json(metrics, Path(out_dir, "metrics.json").as_posix())

        # Prepare predictions for visualization
        df_all = df.copy()
        df_all["Model"] = "Actual"

        lstm_train_dates = train_df["Date"].iloc[len(train_df)-len(lstm_train_true):].reset_index(drop=True)
        lstm_train = pd.DataFrame({"Date": lstm_train_dates, "Users": lstm_train_pred, "Model": "LSTM (train)"})
        lstm_test  = pd.DataFrame({"Date": test_df["Date"].reset_index(drop=True),
                                   "Users": lstm_out["test_pred"], "Model": "LSTM (test)"})

        arima_train = pd.DataFrame({"Date": train_df["Date"].reset_index(drop=True),
                                    "Users": arima_out["train_pred"], "Model": "ARIMA (train)"})
        arima_test  = pd.DataFrame({"Date": test_df["Date"].reset_index(drop=True),
                                    "Users": arima_out["test_pred"], "Model": "ARIMA (test)"})

        # Future predictions
        last_date = df["Date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=CFG.FORECAST_DAYS, freq="D")
        lstm_future = pd.DataFrame({"Date": future_dates, "Users": lstm_out["future"], "Model": "LSTM (future)"})
        arima_future = pd.DataFrame({"Date": future_dates, "Users": arima_out["future"], "Model": "ARIMA (future)"})

        preds = pd.concat([lstm_train, lstm_test, arima_train, arima_test, lstm_future, arima_future], ignore_index=True)
        preds.to_csv(Path(out_dir, "predictions.csv"), index=False)

        # Enhanced visualizations
        create_comparison_plots(df, lstm_train, lstm_test, arima_train, arima_test, 
                               lstm_future, arima_future, out_dir)

        print(f"\n✅ Training completed! Check {out_dir}/ for results.")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise

def create_comparison_plots(df, lstm_train, lstm_test, arima_train, arima_test, 
                           lstm_future, arima_future, out_dir):
    """Create enhanced comparison plots"""
    try:
        # Historical predictions comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(df["Date"], df["Users"], label="Actual", linewidth=2, color='black')
        plt.plot(lstm_train["Date"], lstm_train["Users"], label="LSTM (train)", linewidth=2, alpha=0.8)
        plt.plot(lstm_test["Date"], lstm_test["Users"], label="LSTM (test)", linewidth=2, alpha=0.8)
        plt.plot(arima_train["Date"], arima_train["Users"], label="ARIMA (train)", linewidth=2, alpha=0.8)
        plt.plot(arima_test["Date"], arima_test["Users"], label="ARIMA (test)", linewidth=2, alpha=0.8)
        plt.title("Historical Data & Model Predictions", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Number of Users")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Future forecast comparison
        plt.subplot(2, 1, 2)
        plt.plot(df["Date"], df["Users"], label="Historical", linewidth=2, color='black')
        plt.plot(lstm_future["Date"], lstm_future["Users"], label="LSTM 30-day forecast", 
                 linewidth=3, linestyle='--', color='red')
        plt.plot(arima_future["Date"], arima_future["Users"], label="ARIMA 30-day forecast", 
                 linewidth=3, linestyle='--', color='blue')
        plt.title("Next 30 Days Forecast", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Number of Users")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(out_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Model performance comparison bar chart
        plt.figure(figsize=(10, 6))
        models = ['LSTM', 'ARIMA']
        train_rmse = [rmse(df["Users"], lstm_train["Users"]), rmse(df["Users"], arima_train["Users"])]
        test_rmse = [rmse(df["Users"], lstm_test["Users"]), rmse(df["Users"], arima_test["Users"])]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8)
        plt.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('RMSE')
        plt.title('Model Performance Comparison', fontweight='bold')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(out_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")

if __name__ == "__main__":
    run()
