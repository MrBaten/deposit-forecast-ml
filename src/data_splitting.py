"""
Train/Validation/Test Split Module
Creates proper time-based splits for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesSplitter:
    """Time-based train/validation/test split for forecasting."""
    
    def __init__(self,
                 train_months: int = 10,
                 val_months: int = 1,
                 test_months: int = 1):
        """
        Initialize time series splitter.
        
        Args:
            train_months: Number of months for training
            val_months: Number of months for validation
            test_months: Number of months for testing
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        
        self.split_dates = {}
        self.split_info = {}
        
    def create_time_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based splits.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("="*80)
        print("CREATING TIME-BASED TRAIN/VAL/TEST SPLITS")
        print("="*80)
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Get date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        print(f"\nDataset date range: {min_date.date()} to {max_date.date()}")
        print(f"Total days: {(max_date - min_date).days + 1}")
        
        # Calculate split dates
        # For simplicity, split by month boundaries
        # Train: First 10 months
        # Val: Month 11
        # Test: Month 12
        
        # Calculate approximate split dates
        total_days = (max_date - min_date).days + 1
        train_days = int(total_days * (self.train_months / 12))
        val_days = int(total_days * (self.val_months / 12))
        
        train_end_date = min_date + timedelta(days=train_days)
        val_end_date = train_end_date + timedelta(days=val_days)
        
        # Store split dates
        self.split_dates = {
            'train_start': min_date,
            'train_end': train_end_date,
            'val_start': train_end_date + timedelta(days=1),
            'val_end': val_end_date,
            'test_start': val_end_date + timedelta(days=1),
            'test_end': max_date
        }
        
        # Create splits
        train_df = df[df['date'] <= train_end_date].copy()
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
        test_df = df[df['date'] > val_end_date].copy()
        
        # Calculate split statistics
        self.split_info = {
            'train': {
                'n_records': len(train_df),
                'n_customers': train_df['customer_id'].nunique(),
                'n_days': train_df['date'].nunique(),
                'date_range': f"{train_df['date'].min().date()} to {train_df['date'].max().date()}",
                'total_deposits': train_df['deposit_amount'].sum(),
                'non_zero_deposits': (train_df['deposit_amount'] > 0).sum()
            },
            'val': {
                'n_records': len(val_df),
                'n_customers': val_df['customer_id'].nunique(),
                'n_days': val_df['date'].nunique(),
                'date_range': f"{val_df['date'].min().date()} to {val_df['date'].max().date()}",
                'total_deposits': val_df['deposit_amount'].sum(),
                'non_zero_deposits': (val_df['deposit_amount'] > 0).sum()
            },
            'test': {
                'n_records': len(test_df),
                'n_customers': test_df['customer_id'].nunique(),
                'n_days': test_df['date'].nunique(),
                'date_range': f"{test_df['date'].min().date()} to {test_df['date'].max().date()}",
                'total_deposits': test_df['deposit_amount'].sum(),
                'non_zero_deposits': (test_df['deposit_amount'] > 0).sum()
            }
        }
        
        # Display split information
        print("\n" + "="*80)
        print("SPLIT SUMMARY")
        print("="*80)
        
        for split_name, info in self.split_info.items():
            print(f"\n{split_name.upper()} SET:")
            print(f"  Date range: {info['date_range']}")
            print(f"  Records: {info['n_records']:,}")
            print(f"  Customers: {info['n_customers']}")
            print(f"  Days: {info['n_days']}")
            print(f"  Total deposits: ${info['total_deposits']:,.2f}")
            print(f"  Non-zero deposits: {info['non_zero_deposits']:,} ({info['non_zero_deposits']/info['n_records']:.1%})")
        
        print("\n" + "="*80)
        print("✓ TIME-BASED SPLITS CREATED SUCCESSFULLY")
        print("="*80)
        
        return train_df, val_df, test_df
    
    def prepare_features_and_target(self,
                                    df: pd.DataFrame,
                                    target_col: str = 'deposit_amount',
                                    exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (X, y)
        """
        if exclude_cols is None:
            exclude_cols = ['customer_id', 'date', 'segment', target_col]
        else:
            exclude_cols = exclude_cols + [target_col]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        return X, y
    
    def get_split_info(self) -> Dict:
        """Get information about the splits."""
        return self.split_info
    
    def get_split_dates(self) -> Dict:
        """Get split date boundaries."""
        return self.split_dates


def save_splits(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                test_df: pd.DataFrame,
                output_dir: str = "../data/") -> None:
    """
    Save train/val/test splits to CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save files
    """
    print("\nSaving splits to CSV files...")
    
    train_df.to_csv(f"{output_dir}train_data.csv", index=False)
    print(f"  ✓ Training data saved: {output_dir}train_data.csv")
    
    val_df.to_csv(f"{output_dir}val_data.csv", index=False)
    print(f"  ✓ Validation data saved: {output_dir}val_data.csv")
    
    test_df.to_csv(f"{output_dir}test_data.csv", index=False)
    print(f"  ✓ Test data saved: {output_dir}test_data.csv")


if __name__ == "__main__":
    # Load featured data
    print("Loading featured dataset...")
    df = pd.read_csv("../data/customer_deposits_featured.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Initialize splitter
    splitter = TimeSeriesSplitter(train_months=10, val_months=1, test_months=1)
    
    # Create splits
    train_df, val_df, test_df = splitter.create_time_split(df)
    
    # Save splits
    save_splits(train_df, val_df, test_df)
    
    # Prepare features and target for each split
    print("\n" + "="*80)
    print("PREPARING FEATURES AND TARGET VARIABLES")
    print("="*80)
    
    exclude_cols = ['customer_id', 'date', 'segment']
    
    X_train, y_train = splitter.prepare_features_and_target(train_df, exclude_cols=exclude_cols)
    X_val, y_val = splitter.prepare_features_and_target(val_df, exclude_cols=exclude_cols)
    X_test, y_test = splitter.prepare_features_and_target(test_df, exclude_cols=exclude_cols)
    
    print(f"\nTraining set:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Features: {X_train.shape[1]}")
    
    print(f"\nValidation set:")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")
    
    print(f"\nTest set:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    
    print("\n" + "="*80)
    print("FEATURE NAMES")
    print("="*80)
    print(f"\nTotal features: {len(X_train.columns)}")
    print("\nFirst 20 features:")
    for i, col in enumerate(X_train.columns[:20], 1):
        print(f"  {i:2d}. {col}")
    
    if len(X_train.columns) > 20:
        print(f"  ... and {len(X_train.columns) - 20} more features")
    
    print("\n✓ Splits and feature preparation completed!")
