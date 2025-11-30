"""
Feature Engineering Module for Customer Deposit Forecasting
Creates lag features, rolling statistics, and customer behavior metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DepositFeatureEngineer:
    """Feature engineering for time series deposit forecasting."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        self.customer_metrics = {}
        
    def create_lag_features(self, 
                           df: pd.DataFrame, 
                           lag_days: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lag features for deposit amounts.
        
        Args:
            df: DataFrame with customer deposits (sorted by customer_id and date)
            lag_days: List of lag periods to create
            
        Returns:
            DataFrame with lag features added
        """
        print(f"\nCreating lag features for {len(lag_days)} periods...")
        
        df_lagged = df.copy()
        
        for lag in lag_days:
            feature_name = f'deposit_lag_{lag}d'
            
            # Create lag within each customer
            df_lagged[feature_name] = df_lagged.groupby('customer_id')['deposit_amount'].shift(lag)
            
            # Fill initial NaN values with 0
            df_lagged[feature_name] = df_lagged[feature_name].fillna(0)
            
            self.feature_names.append(feature_name)
        
        print(f"  ✓ Created {len(lag_days)} lag features: {', '.join([f'lag_{d}d' for d in lag_days])}")
        
        return df_lagged
    
    def create_rolling_statistics(self,
                                  df: pd.DataFrame,
                                  windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: DataFrame with customer deposits
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling statistics added
        """
        print(f"\nCreating rolling statistics for {len(windows)} windows...")
        
        df_rolling = df.copy()
        
        for window in windows:
            # Rolling mean
            feature_name = f'rolling_mean_{window}d'
            df_rolling[feature_name] = df_rolling.groupby('customer_id')['deposit_amount'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df_rolling[feature_name] = df_rolling[feature_name].fillna(0)
            self.feature_names.append(feature_name)
            
            # Rolling standard deviation (volatility)
            feature_name = f'rolling_std_{window}d'
            df_rolling[feature_name] = df_rolling.groupby('customer_id')['deposit_amount'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            df_rolling[feature_name] = df_rolling[feature_name].fillna(0)
            self.feature_names.append(feature_name)
            
            # Rolling maximum
            feature_name = f'rolling_max_{window}d'
            df_rolling[feature_name] = df_rolling.groupby('customer_id')['deposit_amount'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
            df_rolling[feature_name] = df_rolling[feature_name].fillna(0)
            self.feature_names.append(feature_name)
            
            # Rolling minimum
            feature_name = f'rolling_min_{window}d'
            df_rolling[feature_name] = df_rolling.groupby('customer_id')['deposit_amount'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )
            df_rolling[feature_name] = df_rolling[feature_name].fillna(0)
            self.feature_names.append(feature_name)
            
            # Rolling sum (total deposits in window)
            feature_name = f'rolling_sum_{window}d'
            df_rolling[feature_name] = df_rolling.groupby('customer_id')['deposit_amount'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
            )
            df_rolling[feature_name] = df_rolling[feature_name].fillna(0)
            self.feature_names.append(feature_name)
        
        print(f"  ✓ Created {len(windows) * 5} rolling features (mean, std, max, min, sum)")
        
        return df_rolling
    
    def create_expanding_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create expanding window statistics (cumulative from start).
        
        Args:
            df: DataFrame with customer deposits
            
        Returns:
            DataFrame with expanding statistics added
        """
        print("\nCreating expanding window statistics...")
        
        df_expanding = df.copy()
        
        # Total deposits to date
        df_expanding['total_deposits_to_date'] = df_expanding.groupby('customer_id')['deposit_amount'].cumsum()
        self.feature_names.append('total_deposits_to_date')
        
        # Number of deposits to date
        df_expanding['num_deposits_to_date'] = df_expanding.groupby('customer_id').cumcount() + 1
        self.feature_names.append('num_deposits_to_date')
        
        # Average deposit to date
        df_expanding['avg_deposit_to_date'] = (
            df_expanding['total_deposits_to_date'] / df_expanding['num_deposits_to_date']
        )
        self.feature_names.append('avg_deposit_to_date')
        
        # Number of non-zero deposits to date
        df_expanding['non_zero_deposits_to_date'] = df_expanding.groupby('customer_id')['deposit_amount'].transform(
            lambda x: (x > 0).cumsum()
        )
        self.feature_names.append('non_zero_deposits_to_date')
        
        # Deposit frequency to date
        df_expanding['deposit_frequency_to_date'] = (
            df_expanding['non_zero_deposits_to_date'] / df_expanding['num_deposits_to_date']
        )
        self.feature_names.append('deposit_frequency_to_date')
        
        print(f"  ✓ Created 5 expanding window features")
        
        return df_expanding
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features like days since last deposit.
        
        Args:
            df: DataFrame with customer deposits
            
        Returns:
            DataFrame with time-based features added
        """
        print("\nCreating time-based features...")
        
        df_time = df.copy()
        
        # Days since last deposit
        def calculate_days_since_last_deposit(group):
            days_since = []
            last_deposit_idx = -1
            
            for idx, (i, row) in enumerate(group.iterrows()):
                if row['deposit_amount'] > 0:
                    last_deposit_idx = idx
                    days_since.append(0)
                else:
                    if last_deposit_idx == -1:
                        days_since.append(idx)  # Days from start
                    else:
                        days_since.append(idx - last_deposit_idx)
            
            return pd.Series(days_since, index=group.index)
        
        df_time['days_since_last_deposit'] = df_time.groupby('customer_id').apply(
            calculate_days_since_last_deposit
        ).reset_index(level=0, drop=True)
        self.feature_names.append('days_since_last_deposit')
        
        # Days from start
        df_time['days_from_start'] = df_time.groupby('customer_id').cumcount()
        self.feature_names.append('days_from_start')
        
        # Is previous day deposit (binary)
        df_time['prev_day_had_deposit'] = (df_time.groupby('customer_id')['deposit_amount'].shift(1) > 0).astype(int)
        df_time['prev_day_had_deposit'] = df_time['prev_day_had_deposit'].fillna(0)
        self.feature_names.append('prev_day_had_deposit')
        
        print(f"  ✓ Created 3 time-based features")
        
        return df_time
    
    def create_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create growth and trend features.
        
        Args:
            df: DataFrame with customer deposits
            
        Returns:
            DataFrame with growth features added
        """
        print("\nCreating growth and trend features...")
        
        df_growth = df.copy()
        
        # 7-day vs 30-day comparison
        if 'rolling_sum_7d' in df.columns and 'rolling_sum_30d' in df.columns:
            df_growth['growth_7d_vs_30d'] = (
                (df_growth['rolling_sum_7d'] - df_growth['rolling_sum_30d'] / 4) / 
                (df_growth['rolling_sum_30d'] / 4 + 1)  # Add 1 to avoid division by zero
            )
            df_growth['growth_7d_vs_30d'] = df_growth['growth_7d_vs_30d'].fillna(0)
            self.feature_names.append('growth_7d_vs_30d')
        
        # Week-over-week change
        df_growth['wow_change'] = df_growth.groupby('customer_id')['deposit_amount'].transform(
            lambda x: x - x.shift(7)
        )
        df_growth['wow_change'] = df_growth['wow_change'].fillna(0)
        self.feature_names.append('wow_change')
        
        # Deposit momentum (7-day avg vs 30-day avg)
        if 'rolling_mean_7d' in df.columns and 'rolling_mean_30d' in df.columns:
            df_growth['deposit_momentum'] = (
                df_growth['rolling_mean_7d'] - df_growth['rolling_mean_30d']
            )
            df_growth['deposit_momentum'] = df_growth['deposit_momentum'].fillna(0)
            self.feature_names.append('deposit_momentum')
        
        # Volatility ratio (recent vs historical)
        if 'rolling_std_7d' in df.columns and 'rolling_std_30d' in df.columns:
            df_growth['volatility_ratio'] = (
                df_growth['rolling_std_7d'] / (df_growth['rolling_std_30d'] + 1)
            )
            df_growth['volatility_ratio'] = df_growth['volatility_ratio'].fillna(0)
            self.feature_names.append('volatility_ratio')
        
        print(f"  ✓ Created 4 growth/trend features")
        
        return df_growth
    
    def create_same_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create same-day-of-week features (last week, 2 weeks ago, etc.).
        
        Args:
            df: DataFrame with customer deposits
            
        Returns:
            DataFrame with same-day features added
        """
        print("\nCreating same-day-of-week features...")
        
        df_same_day = df.copy()
        
        # Last week same day
        df_same_day['deposit_same_day_last_week'] = df_same_day.groupby('customer_id')['deposit_amount'].shift(7)
        df_same_day['deposit_same_day_last_week'] = df_same_day['deposit_same_day_last_week'].fillna(0)
        self.feature_names.append('deposit_same_day_last_week')
        
        # 2 weeks ago same day
        df_same_day['deposit_same_day_2weeks_ago'] = df_same_day.groupby('customer_id')['deposit_amount'].shift(14)
        df_same_day['deposit_same_day_2weeks_ago'] = df_same_day['deposit_same_day_2weeks_ago'].fillna(0)
        self.feature_names.append('deposit_same_day_2weeks_ago')
        
        # 4 weeks ago same day
        df_same_day['deposit_same_day_4weeks_ago'] = df_same_day.groupby('customer_id')['deposit_amount'].shift(28)
        df_same_day['deposit_same_day_4weeks_ago'] = df_same_day['deposit_same_day_4weeks_ago'].fillna(0)
        self.feature_names.append('deposit_same_day_4weeks_ago')
        
        # Average of last 4 same days
        df_same_day['avg_same_day_last_4weeks'] = (
            df_same_day['deposit_same_day_last_week'] +
            df_same_day['deposit_same_day_2weeks_ago'] +
            df_same_day.groupby('customer_id')['deposit_amount'].shift(21).fillna(0) +
            df_same_day['deposit_same_day_4weeks_ago']
        ) / 4
        self.feature_names.append('avg_same_day_last_4weeks')
        
        print(f"  ✓ Created 4 same-day-of-week features")
        
        return df_same_day
    
    def create_customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer-level behavior features.
        
        Args:
            df: DataFrame with customer deposits
            
        Returns:
            DataFrame with customer behavior features added
        """
        print("\nCreating customer behavior features...")
        
        df_behavior = df.copy()
        
        # Calculate customer-level statistics for past 30 days
        def calculate_recent_behavior(group):
            recent_30d = group.tail(30)
            
            # Recent activity level
            recent_activity = (recent_30d['deposit_amount'] > 0).sum() / 30
            
            # Recent average amount
            recent_avg = recent_30d[recent_30d['deposit_amount'] > 0]['deposit_amount'].mean()
            if pd.isna(recent_avg):
                recent_avg = 0
            
            # Recent trend (simple linear trend)
            if len(recent_30d) > 1:
                x = np.arange(len(recent_30d))
                y = recent_30d['deposit_amount'].values
                coef = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                recent_trend = 1 if coef > 0.1 else (-1 if coef < -0.1 else 0)
            else:
                recent_trend = 0
            
            return pd.Series({
                'customer_recent_activity': recent_activity,
                'customer_recent_avg': recent_avg,
                'customer_recent_trend': recent_trend
            })
        
        # Apply to each row with expanding window
        customer_behavior = []
        for customer_id in df_behavior['customer_id'].unique():
            customer_data = df_behavior[df_behavior['customer_id'] == customer_id].copy()
            
            for idx in range(len(customer_data)):
                if idx < 30:
                    behavior = {
                        'customer_recent_activity': 0,
                        'customer_recent_avg': 0,
                        'customer_recent_trend': 0
                    }
                else:
                    recent_data = customer_data.iloc[max(0, idx-30):idx]
                    behavior = calculate_recent_behavior(recent_data)
                
                customer_behavior.append(behavior)
        
        behavior_df = pd.DataFrame(customer_behavior)
        df_behavior[['customer_recent_activity', 'customer_recent_avg', 'customer_recent_trend']] = behavior_df
        
        for col in ['customer_recent_activity', 'customer_recent_avg', 'customer_recent_trend']:
            self.feature_names.append(col)
        
        print(f"  ✓ Created 3 customer behavior features")
        
        return df_behavior
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with all features engineered
        """
        print("="*80)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*80)
        
        # Ensure data is sorted
        df = df.sort_values(['customer_id', 'date']).reset_index(drop=True)
        
        # Apply all feature engineering steps
        df = self.create_lag_features(df)
        df = self.create_rolling_statistics(df)
        df = self.create_expanding_statistics(df)
        df = self.create_time_based_features(df)
        df = self.create_growth_features(df)
        df = self.create_same_day_features(df)
        df = self.create_customer_behavior_features(df)
        
        print("\n" + "="*80)
        print("FEATURE ENGINEERING COMPLETED")
        print("="*80)
        print(f"Total features created: {len(self.feature_names)}")
        print(f"Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all created feature names."""
        return self.feature_names


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv("../data/customer_deposits_preprocessed.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Input shape: {df.shape}")
    
    # Initialize feature engineer
    engineer = DepositFeatureEngineer()
    
    # Engineer all features
    df_featured = engineer.engineer_all_features(df)
    
    # Save featured dataset
    output_path = "../data/customer_deposits_featured.csv"
    df_featured.to_csv(output_path, index=False)
    print(f"\n✓ Featured dataset saved to: {output_path}")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE FEATURED DATA (Customer 1, last 10 days)")
    print("="*80)
    sample = df_featured[df_featured['customer_id'] == 1].tail(10)
    print(sample[['date', 'deposit_amount', 'deposit_lag_1d', 'rolling_mean_7d', 
                  'rolling_std_7d', 'days_since_last_deposit']].to_string(index=False))
    
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    print(f"Total features: {len(engineer.get_feature_names())}")
    print("\nFeature categories:")
    lag_features = [f for f in engineer.get_feature_names() if 'lag' in f]
    rolling_features = [f for f in engineer.get_feature_names() if 'rolling' in f]
    expanding_features = [f for f in engineer.get_feature_names() if 'to_date' in f]
    time_features = [f for f in engineer.get_feature_names() if 'days' in f or 'prev_day' in f]
    growth_features = [f for f in engineer.get_feature_names() if 'growth' in f or 'momentum' in f or 'volatility' in f or 'wow' in f]
    same_day_features = [f for f in engineer.get_feature_names() if 'same_day' in f]
    behavior_features = [f for f in engineer.get_feature_names() if 'customer' in f]
    
    print(f"  Lag features: {len(lag_features)}")
    print(f"  Rolling statistics: {len(rolling_features)}")
    print(f"  Expanding statistics: {len(expanding_features)}")
    print(f"  Time-based features: {len(time_features)}")
    print(f"  Growth features: {len(growth_features)}")
    print(f"  Same-day features: {len(same_day_features)}")
    print(f"  Customer behavior: {len(behavior_features)}")
