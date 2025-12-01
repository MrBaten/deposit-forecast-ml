"""
Interactive Dashboard for Customer Deposit Forecasting.
Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import DepositFeatureEngineer

# Page config
st.set_page_config(
    page_title="Deposit Forecasting Dashboard",
    page_icon="💰",
    layout="wide"
)

# Constants
DATA_PATH = "../data/customer_deposits_featured.csv"
MODEL_PATH = "../models/random_forest.pkl"
METRICS_PATH = "../outputs/test_results.csv"

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}")
        return None
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    """Load model metrics."""
    if not os.path.exists(METRICS_PATH):
        return None
    return pd.read_csv(METRICS_PATH)

def main():
    st.title("💰 Customer Deposit Forecasting Dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Explorer", "Model Performance", "Prediction Simulator"])
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        model = load_model()
    
    if df is None:
        return

    # -------------------------------------------------------------------------
    # PAGE 1: DATA EXPLORER
    # -------------------------------------------------------------------------
    if page == "Data Explorer":
        st.header("📊 Data Explorer")
        
        # Top level stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{df['customer_id'].nunique():,}")
        col2.metric("Total Deposits", f"${df['deposit_amount'].sum():,.2f}")
        col3.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        col4.metric("Avg Daily Deposit", f"${df.groupby('date')['deposit_amount'].sum().mean():,.2f}")
        
        # Segment Filter
        segments = ['All'] + list(df['segment'].unique())
        selected_segment = st.selectbox("Filter by Customer Segment", segments)
        
        filtered_df = df if selected_segment == "All" else df[df['segment'] == selected_segment]
        
        # Time Series Plot
        st.subheader("Daily Deposit Volume")
        daily_vol = filtered_df.groupby('date')['deposit_amount'].sum().reset_index()
        fig_vol = px.line(daily_vol, x='date', y='deposit_amount', title="Daily Total Deposits")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Deposit Amount Distribution (Non-zero)")
            nonzero = filtered_df[filtered_df['deposit_amount'] > 0]
            fig_dist = px.histogram(nonzero, x='deposit_amount', nbins=50, title="Distribution of Deposit Amounts")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col2:
            st.subheader("Deposits by Day of Week")
            dow_stats = filtered_df.groupby('day_of_week')['deposit_amount'].mean().reset_index()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_stats['day_name'] = dow_stats['day_of_week'].apply(lambda x: days[int(x)])
            fig_dow = px.bar(dow_stats, x='day_name', y='deposit_amount', title="Average Deposit by Day")
            st.plotly_chart(fig_dow, use_container_width=True)

    # -------------------------------------------------------------------------
    # PAGE 2: MODEL PERFORMANCE
    # -------------------------------------------------------------------------
    elif page == "Model Performance":
        st.header("🤖 Model Performance")
        
        metrics_df = load_metrics()
        if metrics_df is not None:
            st.subheader("Test Set Metrics")
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R2']), use_container_width=True)
            
            # Highlight best model
            best_model = metrics_df.loc[metrics_df['R2'].idxmax()]
            st.success(f"🏆 Best Model: **{best_model['model'].upper()}** with R² = {best_model['R2']:.4f}")
        
        # Feature Importance (if Random Forest)
        st.subheader("Feature Importance")
        try:
            # Try to load feature importance from file first
            fi_path = "../outputs/random_forest_feature_importance.csv"
            if os.path.exists(fi_path):
                fi_df = pd.read_csv(fi_path)
                fig_fi = px.bar(fi_df.head(15), x='importance', y='feature', orientation='h', 
                              title="Top 15 Important Features", color='importance')
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance file not found.")
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")

    # -------------------------------------------------------------------------
    # PAGE 3: PREDICTION SIMULATOR
    # -------------------------------------------------------------------------
    elif page == "Prediction Simulator":
        st.header("🔮 Prediction Simulator")
        st.markdown("Simulate a customer's state to predict their next-day deposit.")
        
        if model is None:
            st.error("Model not loaded.")
            return

        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Recent Activity")
                lag_1 = st.number_input("Deposit Yesterday ($)", min_value=0.0, value=50.0)
                lag_7 = st.number_input("Deposit 7 Days Ago ($)", min_value=0.0, value=50.0)
                days_since = st.number_input("Days Since Last Deposit", min_value=0, value=1)
                
            with col2:
                st.subheader("Rolling Stats (30d)")
                roll_mean = st.number_input("Avg Deposit (30d)", min_value=0.0, value=45.0)
                roll_std = st.number_input("Std Dev (30d)", min_value=0.0, value=10.0)
                roll_sum = st.number_input("Total (30d)", min_value=0.0, value=1350.0)
                
            with col3:
                st.subheader("Growth & Timing")
                wow_change = st.number_input("Week-over-Week Change (%)", value=0.0)
                is_weekend = st.checkbox("Is Tomorrow Weekend?", value=False)
                segment = st.selectbox("Customer Segment", df['segment'].unique())
            
            submitted = st.form_submit_button("Predict Deposit")
            
            if submitted:
                # Construct feature vector (simplified for demo)
                # In a real app, we'd need to reconstruct all 51 features exactly.
                # For this demo, we'll use a simplified approach or mock the missing ones with averages
                
                # Get feature names from model
                try:
                    feature_names = model.feature_names_in_
                except:
                    # Fallback if attribute not present
                    feature_names = [c for c in df.columns if c not in ['customer_id', 'date', 'segment', 'deposit_amount']]
                
                # Create a dictionary with inputs
                input_data = {
                    'deposit_lag_1d': lag_1,
                    'deposit_lag_7d': lag_7,
                    'days_since_last_deposit': days_since,
                    'rolling_mean_30d': roll_mean,
                    'rolling_std_30d': roll_std,
                    'rolling_sum_30d': roll_sum,
                    'wow_change': wow_change,
                    'is_weekend': 1 if is_weekend else 0,
                    # Fill others with defaults/averages from dataset
                }
                
                # Fill missing features with mean values from dataset
                # This is a simplification for the UI simulator
                for feat in feature_names:
                    if feat not in input_data:
                        if feat in df.columns:
                            input_data[feat] = df[feat].mean()
                        else:
                            input_data[feat] = 0.0
                
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ensure column order matches
                input_df = input_df[feature_names]
                
                # Predict
                prediction = model.predict(input_df)[0]
                prediction = max(0, prediction)  # No negative deposits
                
                st.success(f"### Predicted Deposit: ${prediction:.2f}")
                
                # Context
                if prediction > 100:
                    st.balloons()
                    st.info("🌟 High Value Prediction!")
                elif prediction < 5:
                    st.warning("Low probability of deposit.")

if __name__ == "__main__":
    main()
