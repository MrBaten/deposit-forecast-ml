"""
Tests for Feature Engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from feature_engineering import DepositFeatureEngineer

@pytest.fixture
def engineer():
    return DepositFeatureEngineer()

@pytest.fixture
def sample_ts_data():
    """Create sample time series data for one customer."""
    dates = pd.date_range(start='2023-01-01', periods=10)
    df = pd.DataFrame({
        'customer_id': [1] * 10,
        'date': dates,
        'deposit_amount': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        'segment': ['A'] * 10
    })
    return df

def test_lag_features(engineer, sample_ts_data):
    """Test creation of lag features."""
    lags = [1, 2]
    df_lag = engineer.create_lag_features(sample_ts_data, lag_days=lags)
    
    # Check columns exist
    assert 'deposit_lag_1d' in df_lag.columns
    assert 'deposit_lag_2d' in df_lag.columns
    
    # Check values (lag 1 should be shifted by 1)
    # Original: 10, 20, 30...
    # Lag 1: 0, 10, 20...
    assert df_lag.iloc[0]['deposit_lag_1d'] == 0
    assert df_lag.iloc[1]['deposit_lag_1d'] == 10.0
    assert df_lag.iloc[2]['deposit_lag_1d'] == 20.0

def test_rolling_statistics(engineer, sample_ts_data):
    """Test creation of rolling statistics."""
    windows = [3]
    df_roll = engineer.create_rolling_statistics(sample_ts_data, windows=windows)
    
    # Check columns
    assert 'rolling_mean_3d' in df_roll.columns
    
    # Check calculation
    # Values: 10, 20, 30, 40
    # Rolling mean 3d at index 3 (value 40) uses previous 3 values: 10, 20, 30 -> mean 20
    # Wait, implementation uses shift(1) before rolling
    # So at index 3, it sees window of [10, 20, 30] -> mean 20
    
    # Let's check index 3
    # Previous values: 10, 20, 30
    expected_mean = (10 + 20 + 30) / 3
    assert abs(df_roll.iloc[3]['rolling_mean_3d'] - expected_mean) < 0.001

def test_expanding_statistics(engineer, sample_ts_data):
    """Test expanding window statistics."""
    df_exp = engineer.create_expanding_statistics(sample_ts_data)
    
    assert 'total_deposits_to_date' in df_exp.columns
    
    # Check cumulative sum
    # 10, 20, 30 -> cumsum: 10, 30, 60
    assert df_exp.iloc[0]['total_deposits_to_date'] == 10.0
    assert df_exp.iloc[1]['total_deposits_to_date'] == 30.0
    assert df_exp.iloc[2]['total_deposits_to_date'] == 60.0

def test_data_leakage(engineer, sample_ts_data):
    """Test that features do not use future data."""
    # Lag 1 should rely only on past data
    df_lag = engineer.create_lag_features(sample_ts_data, lag_days=[1])
    
    # If we change the last value, the second to last lag feature should NOT change
    original_lag = df_lag.iloc[-1]['deposit_lag_1d']
    
    # Modify future data (conceptually impossible in real-time but possible in batch)
    # Here we just verify that row N's features depend on N-1, N-2...
    
    # Row 1 (index 1) has value 20. Its lag_1 feature should be 10 (value of row 0).
    assert df_lag.iloc[1]['deposit_lag_1d'] == sample_ts_data.iloc[0]['deposit_amount']
