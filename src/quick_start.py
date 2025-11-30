"""
Quick Start Guide - How to Use Phase 1 Output
This script demonstrates how to load and explore the Phase 1 results.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("  CUSTOMER DEPOSIT FORECASTING - PHASE 1 QUICK START")
print("=" * 80)

# ============================================================================
# 1. Load the Preprocessed Data
# ============================================================================
print("\n1. Loading preprocessed data...")
df = pd.read_csv('../data/customer_deposits_preprocessed.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"   ✓ Loaded {len(df):,} records")
print(f"   ✓ Columns: {len(df.columns)}")
print(f"   ✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================================
# 2. Load Customer Statistics
# ============================================================================
print("\n2. Loading customer statistics...")
customer_stats = pd.read_csv('../data/customer_statistics.csv')

print(f"   ✓ Loaded statistics for {len(customer_stats)} customers")
print("\n   Top 5 customers by total deposits:")
top_customers = customer_stats.nlargest(5, 'total_deposits')[
    ['customer_id', 'segment', 'total_deposits', 'num_deposits', 'avg_deposit']
]
print(top_customers.to_string(index=False))

# ============================================================================
# 3. Explore a Single Customer
# ============================================================================
print("\n3. Example: Analyzing Customer #1...")
customer_1 = df[df['customer_id'] == 1].sort_values('date')

print(f"   Customer ID: 1")
print(f"   Segment: {customer_1['segment'].iloc[0]}")
print(f"   Total deposits: ${customer_1['deposit_amount'].sum():,.2f}")
print(f"   Number of deposits: {(customer_1['deposit_amount'] > 0).sum()}")
print(f"   Average deposit: ${customer_1[customer_1['deposit_amount'] > 0]['deposit_amount'].mean():.2f}")
print(f"   Deposit frequency: {(customer_1['deposit_amount'] > 0).sum() / len(customer_1):.2%}")

print("\n   Last 10 transactions:")
print(customer_1[['date', 'deposit_amount', 'day_of_week', 'is_weekend']].tail(10).to_string(index=False))

# ============================================================================
# 4. Segment Analysis
# ============================================================================
print("\n4. Segment comparison...")
segment_analysis = df.groupby('segment').agg({
    'deposit_amount': ['sum', 'mean', lambda x: (x > 0).sum()],
    'customer_id': 'nunique'
}).round(2)
segment_analysis.columns = ['Total Volume', 'Avg Amount', 'Num Transactions', 'Num Customers']

print("\n   Segment performance:")
print(segment_analysis.to_string())

# ============================================================================
# 5. Temporal Analysis
# ============================================================================
print("\n5. Temporal patterns...")

# Weekend vs Weekday
weekend_avg = df[df['is_weekend'] == 1]['deposit_amount'].mean()
weekday_avg = df[df['is_weekend'] == 0]['deposit_amount'].mean()

print(f"   Weekend avg deposit: ${weekend_avg:.2f}")
print(f"   Weekday avg deposit: ${weekday_avg:.2f}")
print(f"   Weekend boost: {((weekend_avg / weekday_avg - 1) * 100):+.1f}%")

# Month-end vs Other
month_end_avg = df[df['is_month_end'] == 1]['deposit_amount'].mean()
other_avg = df[df['is_month_end'] == 0]['deposit_amount'].mean()

print(f"\n   Month-end avg deposit: ${month_end_avg:.2f}")
print(f"   Other days avg deposit: ${other_avg:.2f}")
print(f"   Month-end boost: {((month_end_avg / other_avg - 1) * 100):+.1f}%")

# ============================================================================
# 6. Data Quality Check
# ============================================================================
print("\n6. Data quality check...")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Duplicate records: {df.duplicated().sum()}")
print(f"   Zero deposits: {(df['deposit_amount'] == 0).sum():,} ({(df['deposit_amount'] == 0).sum() / len(df):.1%})")
print(f"   Non-zero deposits: {(df['deposit_amount'] > 0).sum():,} ({(df['deposit_amount'] > 0).sum() / len(df):.1%})")

# Statistics for non-zero deposits
non_zero = df[df['deposit_amount'] > 0]['deposit_amount']
print(f"\n   Non-zero deposit statistics:")
print(f"   - Mean: ${non_zero.mean():.2f}")
print(f"   - Median: ${non_zero.median():.2f}")
print(f"   - Std Dev: ${non_zero.std():.2f}")
print(f"   - Min: ${non_zero.min():.2f}")
print(f"   - Max: ${non_zero.max():.2f}")

# ============================================================================
# 7. Feature Overview
# ============================================================================
print("\n7. Available features for modeling...")
print("\n   Core features:")
print("   - customer_id, date, deposit_amount, segment")

print("\n   Temporal features:")
temporal_features = ['day_of_week', 'day_of_month', 'day_of_year', 'week_of_year', 
                    'month', 'is_weekend', 'is_month_start', 'is_month_end',
                    'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
for feat in temporal_features:
    print(f"   - {feat}")

# ============================================================================
# 8. Sample Data for Modeling
# ============================================================================
print("\n8. Sample data preview for modeling...")
print("\n   First 5 records:")
print(df[['customer_id', 'date', 'deposit_amount', 'segment', 'day_of_week', 
          'is_weekend', 'is_month_end']].head().to_string(index=False))

# ============================================================================
# 9. Ready for Next Steps
# ============================================================================
print("\n" + "=" * 80)
print("  PHASE 1 DATA IS READY FOR PHASE 2: FEATURE ENGINEERING")
print("=" * 80)

print("\nNext steps:")
print("  1. Create lag features (previous 1, 3, 7, 14, 30 days)")
print("  2. Build rolling statistics (moving averages, std dev)")
print("  3. Engineer customer behavior metrics (frequency, trends)")
print("  4. Create train/validation/test splits")
print("  5. Begin model development in Phase 3")

print("\nFiles available:")
print("  • customer_deposits_preprocessed.csv - Main dataset (365,000 records)")
print("  • customer_statistics.csv - Per-customer aggregates (1,000 customers)")
print("  • outliers_detected.csv - Detected outliers for audit")
print("  • Visualizations in ../visualizations/")

print("\n✅ All systems ready for Phase 2!")
print("=" * 80)
