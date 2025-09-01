import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from src.config import CFG
from src.train_and_forecast import run as run_training
from src.analysis import analyze_data
import json

st.set_page_config(page_title="Web Traffic Forecasting", layout="wide", page_icon="📊")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌐 Web Traffic Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # File upload
    uploaded = st.file_uploader("Upload CSV (Date,Users)", type=["csv"], accept_multiple_files=False)
    
    # Training button
    if st.button("🚀 Run Training & Analysis", type="primary"):
        if uploaded:
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True, parents=True)
            tmp = pd.read_csv(uploaded)
            tmp.to_csv(Path(CFG.DATA_PATH), index=False)
            st.success("Data uploaded successfully!")
        
        with st.spinner("Training models and analyzing data..."):
            run_training(CFG.DATA_PATH)
        st.success("✅ Training completed! Check the results below.")

# Main content
preds_path = Path(CFG.OUTPUT_DIR, "predictions.csv")
metrics_path = Path(CFG.OUTPUT_DIR, "metrics.json")

if preds_path.exists() and metrics_path.exists():
    # Load data
    preds = pd.read_csv(preds_path, parse_dates=["Date"])
    metrics = json.loads(Path(metrics_path).read_text())
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("LSTM Test RMSE", f"{metrics['lstm']['rmse_test']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ARIMA Test RMSE", f"{metrics['arima']['rmse_test']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        improvement = metrics['comparison']['improvement_percent']
        st.metric("Improvement %", f"{improvement:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        winner = "LSTM" if metrics['comparison']['lstm_better'] else "ARIMA"
        st.metric("Better Model", winner)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model comparison message
    if metrics['comparison']['lstm_better']:
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.markdown("🏆 **LSTM performs better than ARIMA!** This shows that deep learning can capture complex patterns in web traffic data.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-message">', unsafe_allow_html=True)
        st.markdown("🏆 **ARIMA performs better than LSTM!** Traditional time series methods can be effective for certain patterns.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Interactive predictions chart
    st.subheader("📈 Interactive Predictions")
    
    # Model selection
    models_to_show = st.multiselect(
        "Select models to display",
        sorted(preds["Model"].unique()),
        default=["LSTM (test)", "ARIMA (test)", "LSTM (future)", "ARIMA (future)"]
    )
    
    if models_to_show:
        to_plot = preds[preds["Model"].isin(models_to_show)]
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add traces for each model
        colors = {'LSTM (train)': '#ff7f0e', 'LSTM (test)': '#ff7f0e', 'LSTM (future)': '#ff7f0e',
                  'ARIMA (train)': '#2ca02c', 'ARIMA (test)': '#2ca02c', 'ARIMA (future)': '#2ca02c'}
        
        for model in models_to_show:
            model_data = to_plot[to_plot["Model"] == model]
            fig.add_trace(go.Scatter(
                x=model_data["Date"],
                y=model_data["Users"],
                mode='lines+markers',
                name=model,
                line=dict(color=colors.get(model, '#1f77b4'), width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Web Traffic Predictions",
            xaxis_title="Date",
            yaxis_title="Number of Users",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance comparison chart
    st.subheader("📊 Model Performance Comparison")
    
    # Create performance comparison
    models = ['LSTM', 'ARIMA']
    train_rmse = [metrics['lstm']['rmse_train'], metrics['arima']['rmse_train']]
    test_rmse = [metrics['lstm']['rmse_test'], metrics['arima']['rmse_test']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Train RMSE',
        x=models,
        y=train_rmse,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Test RMSE',
        x=models,
        y=test_rmse,
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="RMSE Comparison",
        xaxis_title="Models",
        yaxis_title="RMSE",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed metrics
    with st.expander("📋 Detailed Metrics"):
        st.json(metrics)
    
    # Show predictions table
    with st.expander("📋 Predictions Data"):
        st.dataframe(preds.sort_values("Date"))

else:
    st.info("📊 No predictions yet. Click **Run Training & Analysis** in the sidebar to generate them.")
    
    # Show sample data structure
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your CSV file should have the following columns:
    - **Date**: Date in YYYY-MM-DD format
    - **Users**: Number of users (integer)
    
    Example:
    ```
    Date,Users
    2020-01-01,650
    2020-01-02,680
    2020-01-03,720
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Web Traffic Forecasting using LSTM and ARIMA Models</p>
    <p>This project demonstrates time series forecasting with deep learning and traditional statistical methods.</p>
</div>
""", unsafe_allow_html=True)
