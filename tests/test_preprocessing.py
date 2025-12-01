"""
Tests for Preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import DataPreprocessor

@pytest.fixture
def preprocessor():
    return DataPreprocessor()

def test_missing_value_handling(preprocessor):
    """Test handling of missing values."""
    # Create sample data with NaNs
    df = pd.DataFrame({
        'customer_id': [1, 1, 1],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'deposit_amount': [100.0, np.nan, 200.0],
        'segment': ['A', 'A', 'A']
    })
    
    df_clean = preprocessor.handle_missing_values(df)
    
    # Check that NaNs are filled
    assert not df_clean['deposit_amount'].isnull().any()
    # Should be forward filled or filled with 0
    assert df_clean.iloc[1]['deposit_amount'] == 100.0  # Forward fill

def test_outlier_removal(preprocessor):
    """Test outlier detection and removal."""
    # Create data with an extreme outlier
    df = pd.DataFrame({
        'customer_id': [1] * 100,
        'date': pd.date_range(start='2023-01-01', periods=100),
        'deposit_amount': [10.0] * 99 + [10000.0],  # One extreme value
        'segment': ['A'] * 100
    })
    
    df_clean, outliers = preprocessor.remove_outliers(df)
    
    # Check that outlier is modified
    assert df_clean['deposit_amount'].max() < 10000.0
    assert len(outliers) > 0

def test_temporal_features(preprocessor):
    """Test creation of temporal features."""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),  # Sunday
        'customer_id': [1],
        'deposit_amount': [100.0],
        'segment': ['A']
    })
    
    df_feat = preprocessor.create_temporal_features(df)
    
    expected_features = [
        'day_of_week', 'day_of_month', 'day_of_year', 'week_of_year',
        'month', 'is_weekend', 'is_month_start', 'is_month_end',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
    ]
    
    for feat in expected_features:
        assert feat in df_feat.columns
    
    # Check specific values for 2023-01-01 (Sunday)
    assert df_feat.iloc[0]['day_of_week'] == 6  # Sunday is 6
    assert df_feat.iloc[0]['is_weekend'] == 1
    assert df_feat.iloc[0]['is_month_start'] == 1
