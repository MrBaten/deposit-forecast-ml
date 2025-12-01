"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_generator import DepositDataGenerator

@pytest.fixture(scope="session")
def sample_data():
    """Generate a small sample dataset for testing."""
    generator = DepositDataGenerator(n_customers=10, days=30)
    df = generator.generate_data()
    return df

@pytest.fixture(scope="session")
def sample_customer_stats(sample_data):
    """Generate customer stats from sample data."""
    stats = sample_data.groupby('customer_id').agg({
        'deposit_amount': ['count', 'sum', 'mean', 'max']
    })
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    return stats
