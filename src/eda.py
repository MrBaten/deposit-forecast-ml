"""
Exploratory Data Analysis Module for Customer Deposit Forecasting
Provides comprehensive EDA functions and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class DepositEDA:
    """Exploratory Data Analysis for customer deposit data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA class.
        
        Args:
            df: DataFrame with customer deposit data
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def get_summary_statistics(self) -> pd.DataFrame:
        """Calculate comprehensive summary statistics."""
        print("="*80)
        print("OVERALL DATASET STATISTICS")
        print("="*80)
        
        stats = {
            'Total Customers': self.df['customer_id'].nunique(),
            'Date Range': f"{self.df['date'].min().date()} to {self.df['date'].max().date()}",
            'Total Days': self.df['date'].nunique(),
            'Total Transactions': (self.df['deposit_amount'] > 0).sum(),
            'Total Deposit Volume': f"${self.df['deposit_amount'].sum():,.2f}",
            'Avg Daily Deposits (All)': f"${self.df.groupby('date')['deposit_amount'].sum().mean():,.2f}",
            'Avg Deposit per Transaction': f"${self.df[self.df['deposit_amount'] > 0]['deposit_amount'].mean():.2f}",
            'Median Deposit': f"${self.df[self.df['deposit_amount'] > 0]['deposit_amount'].median():.2f}",
            'Max Single Deposit': f"${self.df['deposit_amount'].max():,.2f}",
        }
        
        for key, value in stats.items():
            print(f"{key:.<40} {value}")
        
        print("\n" + "="*80)
        print("DEPOSIT AMOUNT DISTRIBUTION (NON-ZERO)")
        print("="*80)
        print(self.df[self.df['deposit_amount'] > 0]['deposit_amount'].describe())
        
        return pd.DataFrame([stats])
    
    def get_customer_statistics(self) -> pd.DataFrame:
        """Calculate per-customer statistics."""
        customer_stats = self.df.groupby('customer_id').agg({
            'deposit_amount': [
                ('total_deposits', 'sum'),
                ('num_deposits', lambda x: (x > 0).sum()),
                ('avg_deposit', lambda x: x[x > 0].mean()),
                ('max_deposit', 'max'),
                ('std_deposit', 'std')
            ],
            'segment': 'first'
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'total_deposits', 'num_deposits', 
                                 'avg_deposit', 'max_deposit', 'std_deposit', 'segment']
        
        # Calculate deposit frequency
        total_days = self.df['date'].nunique()
        customer_stats['deposit_frequency'] = customer_stats['num_deposits'] / total_days
        
        print("\n" + "="*80)
        print("PER-CUSTOMER STATISTICS")
        print("="*80)
        print(customer_stats.describe())
        
        return customer_stats
    
    def analyze_segments(self) -> pd.DataFrame:
        """Analyze behavior by customer segment."""
        segment_stats = self.df.groupby('segment').agg({
            'customer_id': 'nunique',
            'deposit_amount': [
                ('total_volume', 'sum'),
                ('avg_deposit', lambda x: x[x > 0].mean()),
                ('num_transactions', lambda x: (x > 0).sum()),
                ('avg_per_customer', lambda x: x.sum() / x.count())
            ]
        }).reset_index()
        
        segment_stats.columns = ['segment', 'num_customers', 'total_volume', 
                                'avg_deposit', 'num_transactions', 'avg_per_customer']
        
        print("\n" + "="*80)
        print("SEGMENT ANALYSIS")
        print("="*80)
        print(segment_stats.to_string(index=False))
        
        return segment_stats
    
    def plot_deposit_distributions(self, save_path: str = None):
        """Plot deposit amount distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Non-zero deposits
        non_zero = self.df[self.df['deposit_amount'] > 0]['deposit_amount']
        
        # 1. Histogram
        axes[0, 0].hist(non_zero, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Deposit Amount ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Deposit Amounts (Non-Zero)', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(non_zero.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${non_zero.mean():.2f}')
        axes[0, 0].axvline(non_zero.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${non_zero.median():.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Log-scale histogram
        axes[0, 1].hist(non_zero, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].set_xlabel('Deposit Amount ($)')
        axes[0, 1].set_ylabel('Frequency (log scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Distribution (Log Scale)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Box plot by segment
        segment_data = self.df[self.df['deposit_amount'] > 0]
        segment_data.boxplot(column='deposit_amount', by='segment', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Customer Segment')
        axes[1, 0].set_ylabel('Deposit Amount ($)')
        axes[1, 0].set_title('Deposit Distribution by Segment', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45, ha='right')
        
        # 4. Violin plot
        sns.violinplot(data=segment_data, y='deposit_amount', x='segment', ax=axes[1, 1], palette='Set2')
        axes[1, 1].set_xlabel('Customer Segment')
        axes[1, 1].set_ylabel('Deposit Amount ($)')
        axes[1, 1].set_title('Deposit Amount Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_time_series_patterns(self, save_path: str = None):
        """Plot time series patterns and trends."""
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        
        # Aggregate daily deposits
        daily_deposits = self.df.groupby('date')['deposit_amount'].agg(['sum', 'mean', 'count'])
        daily_deposits['num_transactions'] = self.df[self.df['deposit_amount'] > 0].groupby('date').size()
        
        # 1. Total daily deposits
        axes[0, 0].plot(daily_deposits.index, daily_deposits['sum'], linewidth=1.5, color='navy')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Deposits ($)')
        axes[0, 0].set_title('Daily Total Deposit Volume', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Number of transactions per day
        axes[0, 1].plot(daily_deposits.index, daily_deposits['num_transactions'], linewidth=1.5, color='darkgreen')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Transactions')
        axes[0, 1].set_title('Daily Transaction Count', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. 7-day moving average
        daily_deposits['ma_7'] = daily_deposits['sum'].rolling(window=7).mean()
        daily_deposits['ma_30'] = daily_deposits['sum'].rolling(window=30).mean()
        axes[1, 0].plot(daily_deposits.index, daily_deposits['sum'], alpha=0.3, label='Daily', color='gray')
        axes[1, 0].plot(daily_deposits.index, daily_deposits['ma_7'], linewidth=2, label='7-Day MA', color='blue')
        axes[1, 0].plot(daily_deposits.index, daily_deposits['ma_30'], linewidth=2, label='30-Day MA', color='red')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Total Deposits ($)')
        axes[1, 0].set_title('Deposit Trends with Moving Averages', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Day of week pattern
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_deposits = self.df.groupby('day_of_week')['deposit_amount'].sum().reindex(day_order)
        axes[1, 1].bar(range(7), dow_deposits.values, color='teal', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(day_order, rotation=45, ha='right')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Total Deposits ($)')
        axes[1, 1].set_title('Deposits by Day of Week', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        # 5. Monthly pattern
        self.df['month'] = self.df['date'].dt.month
        monthly_deposits = self.df.groupby('month')['deposit_amount'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[2, 0].bar(monthly_deposits.index, monthly_deposits.values, color='purple', alpha=0.7, edgecolor='black')
        axes[2, 0].set_xticks(range(1, 13))
        axes[2, 0].set_xticklabels(month_names)
        axes[2, 0].set_xlabel('Month')
        axes[2, 0].set_ylabel('Total Deposits ($)')
        axes[2, 0].set_title('Deposits by Month', fontsize=14, fontweight='bold')
        axes[2, 0].grid(alpha=0.3, axis='y')
        
        # 6. Distribution of deposits by day of month
        self.df['day_of_month'] = self.df['date'].dt.day
        dom_deposits = self.df.groupby('day_of_month')['deposit_amount'].sum()
        axes[2, 1].bar(dom_deposits.index, dom_deposits.values, color='orange', alpha=0.7, edgecolor='black')
        axes[2, 1].set_xlabel('Day of Month')
        axes[2, 1].set_ylabel('Total Deposits ($)')
        axes[2, 1].set_title('Deposits by Day of Month', fontsize=14, fontweight='bold')
        axes[2, 1].axvspan(25, 31, alpha=0.2, color='red', label='Month End')
        axes[2, 1].axvspan(1, 5, alpha=0.2, color='green', label='Month Start')
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_customer_behavior(self, save_path: str = None):
        """Plot customer behavior analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        customer_stats = self.get_customer_statistics()
        
        # 1. Customer segments distribution
        segment_counts = self.df.groupby('segment')['customer_id'].nunique()
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=sns.color_palette('Set3'))
        axes[0, 0].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Deposit frequency histogram
        axes[0, 1].hist(customer_stats['deposit_frequency'], bins=30, edgecolor='black', 
                       alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Deposit Frequency (deposits/day)')
        axes[0, 1].set_ylabel('Number of Customers')
        axes[0, 1].set_title('Distribution of Customer Deposit Frequency', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Total deposits per customer
        axes[1, 0].hist(customer_stats['total_deposits'], bins=40, edgecolor='black', 
                       alpha=0.7, color='coral')
        axes[1, 0].set_xlabel('Total Deposits ($)')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Distribution of Total Deposits per Customer', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Scatter: frequency vs average amount
        scatter = axes[1, 1].scatter(customer_stats['deposit_frequency'], 
                                     customer_stats['avg_deposit'],
                                     c=customer_stats['total_deposits'], 
                                     cmap='viridis', alpha=0.6, s=50)
        axes[1, 1].set_xlabel('Deposit Frequency')
        axes[1, 1].set_ylabel('Average Deposit Amount ($)')
        axes[1, 1].set_title('Frequency vs Average Amount (colored by total)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='Total Deposits ($)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_customers(self, n_samples: int = 9, save_path: str = None):
        """Plot time series for sample customers from different segments."""
        segments = self.df['segment'].unique()
        
        # Sample customers from each segment
        sample_customers = []
        for segment in segments[:n_samples]:
            customers = self.df[self.df['segment'] == segment]['customer_id'].unique()
            if len(customers) > 0:
                sample_customers.append(customers[0])
        
        rows = int(np.ceil(len(sample_customers) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(18, rows*4))
        axes = axes.flatten() if rows > 1 else [axes] if len(sample_customers) == 1 else axes
        
        for idx, customer_id in enumerate(sample_customers):
            customer_data = self.df[self.df['customer_id'] == customer_id].sort_values('date')
            segment = customer_data['segment'].iloc[0]
            
            ax = axes[idx]
            ax.plot(customer_data['date'], customer_data['deposit_amount'], 
                   linewidth=1.5, marker='o', markersize=2, color='darkblue')
            ax.set_xlabel('Date')
            ax.set_ylabel('Deposit Amount ($)')
            ax.set_title(f'Customer {customer_id} - {segment}', fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide extra subplots
        for idx in range(len(sample_customers), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_correlations(self):
        """Analyze correlations between temporal features and deposits."""
        # Create temporal features
        analysis_df = self.df.copy()
        analysis_df['day_of_week'] = analysis_df['date'].dt.dayofweek
        analysis_df['day_of_month'] = analysis_df['date'].dt.day
        analysis_df['month'] = analysis_df['date'].dt.month
        analysis_df['is_weekend'] = (analysis_df['day_of_week'] >= 5).astype(int)
        analysis_df['is_month_end'] = (analysis_df['day_of_month'] >= 25).astype(int)
        
        # Calculate correlations
        features = ['day_of_week', 'day_of_month', 'month', 'is_weekend', 'is_month_end']
        corr_data = analysis_df[features + ['deposit_amount']].corr()
        
        print("\n" + "="*80)
        print("CORRELATION WITH DEPOSIT AMOUNT")
        print("="*80)
        print(corr_data['deposit_amount'].sort_values(ascending=False))
        
        # Visualize
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/customer_deposits_raw.csv")
    
    # Perform EDA
    eda = DepositEDA(df)
    
    print("Starting Exploratory Data Analysis...\n")
    
    # Summary statistics
    eda.get_summary_statistics()
    customer_stats = eda.get_customer_statistics()
    segment_stats = eda.analyze_segments()
    
    # Visualizations
    print("\nGenerating visualizations...")
    eda.plot_deposit_distributions("../visualizations/01_deposit_distributions.png")
    eda.plot_time_series_patterns("../visualizations/02_time_series_patterns.png")
    eda.plot_customer_behavior("../visualizations/03_customer_behavior.png")
    eda.plot_sample_customers(n_samples=9, save_path="../visualizations/04_sample_customers.png")
    eda.analyze_correlations()
    
    print("\n✓ EDA completed successfully!")
