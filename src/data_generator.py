"""
Data Generator Module for Customer Deposit Forecasting
Generates synthetic deposit data for 1000 customers over 12 months
with realistic patterns including seasonality, trends, and noise.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class CustomerDepositGenerator:
    """Generate synthetic customer deposit data with realistic patterns."""
    
    def __init__(self, 
                 n_customers: int = 1000, 
                 n_days: int = 365,
                 random_seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            n_customers: Number of customers to generate
            n_days: Number of days of historical data
            random_seed: Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.n_days = n_days
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define customer behavior segments
        self.segment_distribution = {
            'high_frequency_regular': 0.15,  # 15% - Daily/regular depositors
            'medium_frequency_stable': 0.25,  # 25% - Weekly depositors
            'growing_users': 0.20,            # 20% - Increasing deposits
            'declining_users': 0.15,          # 15% - Decreasing deposits
            'sporadic_high_value': 0.10,      # 10% - Rare but large deposits
            'weekend_warriors': 0.10,         # 10% - Weekend depositors
            'inactive_declining': 0.05        # 5% - Very low activity
        }
        
    def generate_dates(self) -> pd.DatetimeIndex:
        """Generate date range for the dataset."""
        end_date = datetime(2024, 11, 30)
        start_date = end_date - timedelta(days=self.n_days - 1)
        return pd.date_range(start=start_date, end=end_date, freq='D')
    
    def assign_customer_segments(self) -> np.ndarray:
        """Assign each customer to a behavior segment."""
        segments = []
        for segment, proportion in self.segment_distribution.items():
            n_segment_customers = int(self.n_customers * proportion)
            segments.extend([segment] * n_segment_customers)
        
        # Fill remaining customers with most common segment
        while len(segments) < self.n_customers:
            segments.append('medium_frequency_stable')
        
        return np.array(segments)
    
    def generate_base_amount(self, segment: str) -> float:
        """Generate base deposit amount based on customer segment."""
        base_amounts = {
            'high_frequency_regular': np.random.gamma(shape=2, scale=50),
            'medium_frequency_stable': np.random.gamma(shape=2, scale=100),
            'growing_users': np.random.gamma(shape=2, scale=75),
            'declining_users': np.random.gamma(shape=2, scale=80),
            'sporadic_high_value': np.random.gamma(shape=1.5, scale=300),
            'weekend_warriors': np.random.gamma(shape=2, scale=120),
            'inactive_declining': np.random.gamma(shape=1, scale=30)
        }
        return max(10, base_amounts.get(segment, 50))
    
    def generate_seasonality(self, dates: pd.DatetimeIndex, segment: str) -> np.ndarray:
        """
        Generate seasonal patterns in deposits.
        
        Args:
            dates: Date range
            segment: Customer segment
            
        Returns:
            Array of seasonal multipliers
        """
        day_of_week = dates.dayofweek
        day_of_month = dates.day
        
        # Weekend effect
        weekend_multiplier = np.ones(len(dates))
        if segment in ['weekend_warriors', 'high_frequency_regular']:
            weekend_multiplier[day_of_week >= 5] = 1.5  # Boost on weekends
        
        # Month-end effect (salary days)
        month_end_multiplier = np.ones(len(dates))
        month_end_days = (day_of_month >= 25) | (day_of_month <= 5)
        month_end_multiplier[month_end_days] = 1.3
        
        # Weekly cycle
        weekly_pattern = 1 + 0.2 * np.sin(2 * np.pi * day_of_week / 7)
        
        return weekend_multiplier * month_end_multiplier * weekly_pattern
    
    def generate_trend(self, dates: pd.DatetimeIndex, segment: str) -> np.ndarray:
        """
        Generate trend component.
        
        Args:
            dates: Date range
            segment: Customer segment
            
        Returns:
            Array of trend multipliers
        """
        t = np.arange(len(dates))
        
        if segment == 'growing_users':
            # Exponential growth
            return 1 + 0.5 * (1 - np.exp(-t / 100))
        elif segment == 'declining_users':
            # Declining trend
            return 1.5 * np.exp(-t / 150)
        elif segment == 'inactive_declining':
            # Sharp decline
            return 1.2 * np.exp(-t / 80)
        else:
            # Stable with minor fluctuation
            return 1 + 0.1 * np.sin(2 * np.pi * t / 365)
    
    def generate_deposit_probability(self, dates: pd.DatetimeIndex, segment: str) -> np.ndarray:
        """
        Generate probability of making a deposit on each day.
        
        Args:
            dates: Date range
            segment: Customer segment
            
        Returns:
            Array of probabilities
        """
        day_of_week = dates.dayofweek
        
        base_probs = {
            'high_frequency_regular': 0.70,
            'medium_frequency_stable': 0.30,
            'growing_users': 0.35,
            'declining_users': 0.25,
            'sporadic_high_value': 0.08,
            'weekend_warriors': 0.20,
            'inactive_declining': 0.10
        }
        
        base_prob = base_probs.get(segment, 0.20)
        probs = np.full(len(dates), base_prob)
        
        # Weekend boost for relevant segments
        if segment == 'weekend_warriors':
            probs[day_of_week >= 5] = 0.60
        
        return probs
    
    def generate_customer_deposits(self, 
                                   customer_id: int, 
                                   segment: str, 
                                   dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate deposit time series for a single customer.
        
        Args:
            customer_id: Unique customer identifier
            segment: Customer behavior segment
            dates: Date range
            
        Returns:
            DataFrame with customer deposits
        """
        # Base amount
        base_amount = self.generate_base_amount(segment)
        
        # Seasonal and trend components
        seasonality = self.generate_seasonality(dates, segment)
        trend = self.generate_trend(dates, segment)
        
        # Deposit probability
        deposit_probs = self.generate_deposit_probability(dates, segment)
        
        # Generate deposits
        deposits = []
        for i, date in enumerate(dates):
            # Determine if deposit occurs
            if np.random.random() < deposit_probs[i]:
                # Calculate amount with noise
                amount = base_amount * seasonality[i] * trend[i]
                amount *= np.random.lognormal(0, 0.3)  # Add multiplicative noise
                amount = max(5, amount)  # Minimum deposit
                
                # Add occasional large deposits (outliers)
                if np.random.random() < 0.02:
                    amount *= np.random.uniform(2, 5)
                
                deposits.append(amount)
            else:
                deposits.append(0)
        
        return pd.DataFrame({
            'customer_id': customer_id,
            'date': dates,
            'deposit_amount': deposits,
            'segment': segment
        })
    
    def generate_full_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset for all customers.
        
        Returns:
            DataFrame with all customer deposits
        """
        print(f"Generating synthetic deposit data for {self.n_customers} customers...")
        
        dates = self.generate_dates()
        segments = self.assign_customer_segments()
        
        all_data = []
        for customer_id in range(1, self.n_customers + 1):
            if customer_id % 100 == 0:
                print(f"  Generated data for {customer_id}/{self.n_customers} customers")
            
            segment = segments[customer_id - 1]
            customer_data = self.generate_customer_deposits(customer_id, segment, dates)
            all_data.append(customer_data)
        
        df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✓ Dataset generated successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Total deposits: ${df['deposit_amount'].sum():,.2f}")
        
        return df


def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """Save dataset to CSV file."""
    df.to_csv(filepath, index=False)
    print(f"\n✓ Dataset saved to: {filepath}")


if __name__ == "__main__":
    # Generate dataset
    generator = CustomerDepositGenerator(n_customers=1000, n_days=365)
    df = generator.generate_full_dataset()
    
    # Save to CSV
    save_dataset(df, "../data/customer_deposits_raw.csv")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE DATA:")
    print("="*80)
    print(df.head(20))
    print("\n" + "="*80)
    print("DATASET STATISTICS:")
    print("="*80)
    print(df.describe())
