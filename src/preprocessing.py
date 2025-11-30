"""
Data Preprocessing Module for Customer Deposit Forecasting
Handles missing values, outlier removal, and data normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DepositPreprocessor:
    """Preprocessing pipeline for customer deposit data."""
    
    def __init__(self, 
                 outlier_threshold: float = 3.0,
                 scaling_method: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            outlier_threshold: Number of standard deviations for outlier detection
            scaling_method: 'standard' or 'minmax'
        """
        self.outlier_threshold = outlier_threshold
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.outlier_stats = {}
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward-fill method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("Handling missing values...")
        
        df_processed = df.copy()
        missing_before = df_processed['deposit_amount'].isna().sum()
        
        # Forward fill missing values within each customer
        df_processed['deposit_amount'] = df_processed.groupby('customer_id')['deposit_amount'].fillna(method='ffill')
        
        # If still missing (at the start), fill with 0
        df_processed['deposit_amount'] = df_processed['deposit_amount'].fillna(0)
        
        missing_after = df_processed['deposit_amount'].isna().sum()
        
        print(f"  Missing values before: {missing_before}")
        print(f"  Missing values after: {missing_after}")
        
        return df_processed
    
    def remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove extreme outliers based on z-score method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed_df, outliers_df)
        """
        print(f"\nRemoving outliers (threshold: {self.outlier_threshold} std)...")
        
        df_processed = df.copy()
        
        # Calculate statistics per customer for non-zero deposits
        customer_stats = df_processed[df_processed['deposit_amount'] > 0].groupby('customer_id').agg({
            'deposit_amount': ['mean', 'std']
        })
        customer_stats.columns = ['mean', 'std']
        
        # Store stats
        self.outlier_stats = customer_stats.to_dict('index')
        
        # Identify outliers
        outliers = []
        
        for customer_id in df_processed['customer_id'].unique():
            customer_data = df_processed[df_processed['customer_id'] == customer_id]
            
            if customer_id in self.outlier_stats:
                mean = self.outlier_stats[customer_id]['mean']
                std = self.outlier_stats[customer_id]['std']
                
                if pd.notna(std) and std > 0:
                    # Calculate z-scores
                    z_scores = np.abs((customer_data['deposit_amount'] - mean) / std)
                    
                    # Find outliers
                    outlier_mask = (z_scores > self.outlier_threshold) & (customer_data['deposit_amount'] > 0)
                    
                    if outlier_mask.any():
                        outliers.append(customer_data[outlier_mask])
        
        # Combine outliers
        outliers_df = pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame()
        
        if not outliers_df.empty:
            # Remove outliers from main dataset (set to customer mean instead of removing)
            for idx in outliers_df.index:
                customer_id = outliers_df.loc[idx, 'customer_id']
                if customer_id in self.outlier_stats:
                    df_processed.loc[idx, 'deposit_amount'] = self.outlier_stats[customer_id]['mean']
        
        print(f"  Outliers detected: {len(outliers_df)}")
        print(f"  Outliers replaced with customer mean")
        
        return df_processed, outliers_df
    
    def create_daily_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily aggregate features per customer.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with daily aggregates
        """
        print("\nCreating daily aggregates...")
        
        # Ensure we have one row per customer per day
        df_daily = df.groupby(['customer_id', 'date']).agg({
            'deposit_amount': 'sum',  # Sum if multiple deposits per day
            'segment': 'first'
        }).reset_index()
        
        # Sort by customer and date
        df_daily = df_daily.sort_values(['customer_id', 'date']).reset_index(drop=True)
        
        print(f"  Daily aggregates shape: {df_daily.shape}")
        
        return df_daily
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        print("\nAdding temporal features...")
        
        df_processed = df.copy()
        df_processed['date'] = pd.to_datetime(df_processed['date'])
        
        # Day features
        df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
        df_processed['day_of_month'] = df_processed['date'].dt.day
        df_processed['day_of_year'] = df_processed['date'].dt.dayofyear
        
        # Week features
        df_processed['week_of_year'] = df_processed['date'].dt.isocalendar().week
        
        # Month features
        df_processed['month'] = df_processed['date'].dt.month
        
        # Binary features
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        df_processed['is_month_start'] = (df_processed['day_of_month'] <= 5).astype(int)
        df_processed['is_month_end'] = (df_processed['day_of_month'] >= 25).astype(int)
        
        # Cyclical encoding for day of week
        df_processed['dow_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
        df_processed['dow_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        
        print(f"  Added {len([col for col in df_processed.columns if col not in df.columns])} temporal features")
        
        return df_processed
    
    def scale_features(self, 
                      df: pd.DataFrame, 
                      features_to_scale: list,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale specified features.
        
        Args:
            df: Input DataFrame
            features_to_scale: List of feature names to scale
            fit: Whether to fit the scaler (True for train, False for test)
            
        Returns:
            DataFrame with scaled features
        """
        df_processed = df.copy()
        
        if fit:
            print(f"\nFitting scaler and scaling features ({self.scaling_method})...")
            df_processed[features_to_scale] = self.scaler.fit_transform(df_processed[features_to_scale])
        else:
            print(f"\nScaling features using fitted scaler...")
            df_processed[features_to_scale] = self.scaler.transform(df_processed[features_to_scale])
        
        return df_processed
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Tuple of (processed_df, outliers_df)
        """
        print("="*80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Remove outliers
        df_processed, outliers_df = self.remove_outliers(df_processed)
        
        # Step 3: Create daily aggregates
        df_processed = self.create_daily_aggregates(df_processed)
        
        # Step 4: Add temporal features
        df_processed = self.add_temporal_features(df_processed)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED")
        print("="*80)
        print(f"Final shape: {df_processed.shape}")
        print(f"Features: {list(df_processed.columns)}")
        
        return df_processed, outliers_df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get preprocessing statistics."""
        stats = {
            'total_rows': len(df),
            'total_customers': df['customer_id'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'total_deposits': df['deposit_amount'].sum(),
            'zero_deposits': (df['deposit_amount'] == 0).sum(),
            'non_zero_deposits': (df['deposit_amount'] > 0).sum(),
        }
        return stats


if __name__ == "__main__":
    # Load raw data
    print("Loading raw data...")
    df_raw = pd.read_csv("../data/customer_deposits_raw.csv")
    
    # Initialize preprocessor
    preprocessor = DepositPreprocessor(outlier_threshold=3.0, scaling_method='standard')
    
    # Run preprocessing pipeline
    df_processed, outliers = preprocessor.preprocess_pipeline(df_raw)
    
    # Save processed data
    df_processed.to_csv("../data/customer_deposits_preprocessed.csv", index=False)
    print(f"\n✓ Processed data saved to: ../data/customer_deposits_preprocessed.csv")
    
    if not outliers.empty:
        outliers.to_csv("../data/outliers_detected.csv", index=False)
        print(f"✓ Outliers saved to: ../data/outliers_detected.csv")
    
    # Display statistics
    print("\n" + "="*80)
    print("PREPROCESSING STATISTICS")
    print("="*80)
    stats = preprocessor.get_statistics(df_processed)
    for key, value in stats.items():
        print(f"{key:.<40} {value}")
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE PROCESSED DATA")
    print("="*80)
    print(df_processed.head(20))
