"""
Tests for Data Generator module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_generator import DepositDataGenerator

def test_generator_initialization():
    """Test generator initialization with custom parameters."""
    generator = DepositDataGenerator(n_customers=50, days=60)
    assert generator.n_customers == 50
    assert generator.days == 60

def test_data_shape(sample_data):
    """Test generated data shape."""
    # Should have 10 customers * 30 days = 300 rows
    assert len(sample_data) == 300
    
    expected_columns = ['date', 'customer_id', 'segment', 'deposit_amount']
    for col in expected_columns:
        assert col in sample_data.columns

def test_data_types(sample_data):
    """Test data types of generated columns."""
    assert pd.api.types.is_datetime64_any_dtype(sample_data['date'])
    assert pd.api.types.is_integer_dtype(sample_data['customer_id'])
    assert pd.api.types.is_string_dtype(sample_data['segment'])
    assert pd.api.types.is_float_dtype(sample_data['deposit_amount'])

def test_customer_segments(sample_data):
    """Test that all customers are assigned a valid segment."""
    valid_segments = [
        'high_frequency_regular', 'medium_frequency_stable', 'growing_users',
        'declining_users', 'sporadic_high_value', 'weekend_warriors',
        'inactive_declining'
    ]
    unique_segments = sample_data['segment'].unique()
    for segment in unique_segments:
        assert segment in valid_segments

def test_no_negative_deposits(sample_data):
    """Test that there are no negative deposit amounts."""
    assert (sample_data['deposit_amount'] >= 0).all()
