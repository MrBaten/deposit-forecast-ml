"""
Complete Model Execution Script
Re-runs all models with comprehensive output and error handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*80)
print("  COMPLETE MODEL TRAINING - ALL PHASES")
print("="*80)

# Check for advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM not available")

# ============================================================================
# Load Data
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_df = pd.read_csv("../data/train_data.csv")
val_df = pd.read_csv("../data/val_data.csv")
test_df = pd.read_csv("../data/test_data.csv")

print(f"\nDataset shapes:")
print(f"  Train: {train_df.shape}")
print(f"  Val:   {val_df.shape}")
print(f"  Test:  {test_df.shape}")

# Prepare features
exclude_cols = ['customer_id', 'date', 'segment']

def prepare_data(df):
    feature_cols = [col for col in df.columns if col not in exclude_cols + ['deposit_amount']]
    X = df[feature_cols]
    y = df['deposit_amount']
    return X, y

X_train, y_train = prepare_data(train_df)
X_val, y_val = prepare_data(val_df)
X_test, y_test = prepare_data(test_df)

print(f"\nFeatures: {X_train.shape[1]}")
print(f"Training samples: {len(X_train):,}")

# ============================================================================
# Define Metrics Function
# ============================================================================

def calculate_metrics(y_true, y_pred, model_name, split='Val'):
    """Calculate all metrics."""
    y_pred = np.maximum(y_pred, 0)  # No negative predictions
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE for non-zero
    non_zero_mask = y_true > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                             y_true[non_zero_mask])) * 100
    else:
        mape = 0.0
    
    return {
        'model': model_name,
        'split': split,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

# ============================================================================
# MODEL 1: Linear Regression
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LINEAR REGRESSION")
print("="*80)

lr_model = LinearRegression()
print("Training...")
lr_model.fit(X_train, y_train)

lr_val_pred = lr_model.predict(X_val)
lr_test_pred = lr_model.predict(X_test)

lr_val_metrics = calculate_metrics(y_val, lr_val_pred, 'Linear Regression', 'Val')
lr_test_metrics = calculate_metrics(y_test, lr_test_pred, 'Linear Regression', 'Test')

print(f"\nValidation Results:")
print(f"  MAE:  ${lr_val_metrics['MAE']:.2f}")
print(f"  RMSE: ${lr_val_metrics['RMSE']:.2f}")
print(f"  R²:   {lr_val_metrics['R2']:.4f}")

print(f"\nTest Results:")
print(f"  MAE:  ${lr_test_metrics['MAE']:.2f}")
print(f"  RMSE: ${lr_test_metrics['RMSE']:.2f}")
print(f"  R²:   {lr_test_metrics['R2']:.4f}")
print(f"  MAPE: {lr_test_metrics['MAPE']:.1f}%")

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(lr_model, "../models/linear_regression.pkl")
print("✓ Model saved")

# ============================================================================
# MODEL 2: Ridge Regression
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: RIDGE REGRESSION")
print("="*80)

ridge_model = Ridge(alpha=1.0)
print("Training...")
ridge_model.fit(X_train, y_train)

ridge_val_pred = ridge_model.predict(X_val)
ridge_test_pred = ridge_model.predict(X_test)

ridge_val_metrics = calculate_metrics(y_val, ridge_val_pred, 'Ridge', 'Val')
ridge_test_metrics = calculate_metrics(y_test, ridge_test_pred, 'Ridge', 'Test')

print(f"\nTest Results:")
print(f"  MAE:  ${ridge_test_metrics['MAE']:.2f}")
print(f"  RMSE: ${ridge_test_metrics['RMSE']:.2f}")
print(f"  R²:   {ridge_test_metrics['R2']:.4f}")

joblib.dump(ridge_model, "../models/ridge_regression.pkl")
print("✓ Model saved")

# ============================================================================
# MODEL 3: Random Forest
# ============================================================================
print("\n" + "="*80)
print("MODEL 3: RANDOM FOREST")
print("="*80)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training (this may take a few minutes)...")
rf_model.fit(X_train, y_train)

rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

rf_val_metrics = calculate_metrics(y_val, rf_val_pred, 'Random Forest', 'Val')
rf_test_metrics = calculate_metrics(y_test, rf_test_pred, 'Random Forest', 'Test')

print(f"\nTest Results:")
print(f"  MAE:  ${rf_test_metrics['MAE']:.2f}")
print(f"  RMSE: ${rf_test_metrics['RMSE']:.2f}")
print(f"  R²:   {rf_test_metrics['R2']:.4f}")
print(f"  MAPE: {rf_test_metrics['MAPE']:.1f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:.<50} {row['importance']:.4f}")

joblib.dump(rf_model, "../models/random_forest.pkl")
feature_importance.to_csv("../outputs/random_forest_feature_importance.csv", index=False)
print("✓ Model and feature importance saved")

# ============================================================================
# MODEL 4: XGBoost (if available)
# ============================================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*80)
    print("MODEL 4: XGBOOST")
    print("="*80)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print("Training...")
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_test_pred = xgb_model.predict(X_test)
    
    xgb_val_metrics = calculate_metrics(y_val, xgb_val_pred, 'XGBoost', 'Val')
    xgb_test_metrics = calculate_metrics(y_test, xgb_test_pred, 'XGBoost', 'Test')
    
    print(f"\nTest Results:")
    print(f"  MAE:  ${xgb_test_metrics['MAE']:.2f}")
    print(f"  RMSE: ${xgb_test_metrics['RMSE']:.2f}")
    print(f"  R²:   {xgb_test_metrics['R2']:.4f}")
    
    xgb_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    for idx, row in xgb_importance.head(10).iterrows():
        print(f"  {row['feature']:.<50} {row['importance']:.4f}")
    
    joblib.dump(xgb_model, "../models/xgboost.pkl")
    xgb_importance.to_csv("../outputs/xgboost_feature_importance.csv", index=False)
    print("✓ Model and feature importance saved")

# ============================================================================
# MODEL 5: LightGBM (if available)
# ============================================================================
if LIGHTGBM_AVAILABLE:
    print("\n" + "="*80)
    print("MODEL 5: LIGHTGBM")
    print("="*80)
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    print("Training...")
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
    
    lgb_val_pred = lgb_model.predict(X_val)
    lgb_test_pred = lgb_model.predict(X_test)
    
    lgb_val_metrics = calculate_metrics(y_val, lgb_val_pred, 'LightGBM', 'Val')
    lgb_test_metrics = calculate_metrics(y_test, lgb_test_pred, 'LightGBM', 'Test')
    
    print(f"\nTest Results:")
    print(f"  MAE:  ${lgb_test_metrics['MAE']:.2f}")
    print(f"  RMSE: ${lgb_test_metrics['RMSE']:.2f}")
    print(f"  R²:   {lgb_test_metrics['R2']:.4f}")
    
    lgb_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    for idx, row in lgb_importance.head(10).iterrows():
        print(f"  {row['feature']:.<50} {row['importance']:.4f}")
    
    joblib.dump(lgb_model, "../models/lightgbm.pkl")
    lgb_importance.to_csv("../outputs/lightgbm_feature_importance.csv", index=False)
    print("✓ Model and feature importance saved")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

# Collect all test results
all_test_results = [lr_test_metrics, ridge_test_metrics, rf_test_metrics]
if XGBOOST_AVAILABLE:
    all_test_results.append(xgb_test_metrics)
if LIGHTGBM_AVAILABLE:
    all_test_results.append(lgb_test_metrics)

results_df = pd.DataFrame(all_test_results)
results_df = results_df.sort_values('R2', ascending=False)

print("\nTest Set Performance (Ranked by R²):")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
os.makedirs("../outputs", exist_ok=True)
results_df.to_csv("../outputs/final_test_results.csv", index=False)

# Find best model
best_model = results_df.iloc[0]
print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)
print(f"Model: {best_model['model'].upper()}")
print(f"  MAE:  ${best_model['MAE']:.2f}")
print(f"  RMSE: ${best_model['RMSE']:.2f}")
print(f"  R²:   {best_model['R2']:.4f} ({best_model['R2']*100:.2f}%)")
print(f"  MAPE: {best_model['MAPE']:.1f}%")

# Create predictions file with best model
print("\n" + "="*80)
print("CREATING PREDICTION REPORTS")
print("="*80)

best_model_name = best_model['model'].lower().replace(' ', '_')
best_model_obj = joblib.load(f"../models/{best_model_name}.pkl")
best_predictions = best_model_obj.predict(X_test)
best_predictions = np.maximum(best_predictions, 0)

predictions_df = pd.DataFrame({
    'customer_id': test_df['customer_id'].values,
    'date': test_df['date'].values,
    'segment': test_df['segment'].values,
    'actual_deposit': y_test.values,
    'predicted_deposit': best_predictions,
    'absolute_error': np.abs(y_test.values - best_predictions),
    'percentage_error': np.where(
        y_test.values > 0,
        100 * np.abs(y_test.values - best_predictions) / y_test.values,
        0
    )
})

predictions_df.to_csv("../outputs/best_model_predictions.csv", index=False)
print("✓ Predictions saved to: ../outputs/best_model_predictions.csv")

# High-value predictions
high_value = predictions_df[predictions_df['predicted_deposit'] >= 100.0].sort_values('predicted_deposit', ascending=False)
high_value.to_csv("../outputs/high_value_customers.csv", index=False)
print(f"✓ High-value customers ({len(high_value)}): ../outputs/high_value_customers.csv")

# Segment performance
print("\n" + "="*80)
print("SEGMENT-WISE PERFORMANCE")
print("="*80)

segment_perf = predictions_df.groupby('segment').apply(
    lambda x: pd.Series({
        'count': len(x),
        'actual_mean': x['actual_deposit'].mean(),
        'predicted_mean': x['predicted_deposit'].mean(),
        'mae': x['absolute_error'].mean(),
        'r2': r2_score(x['actual_deposit'], x['predicted_deposit'])
    })
).round(2)

print("\n" + segment_perf.to_string())
segment_perf.to_csv("../outputs/segment_performance.csv")
print("\n✓ Segment performance saved")

print("\n" + "="*80)
print("✅ ALL MODELS TRAINED AND EVALUATED SUCCESSFULLY!")
print("="*80)

print(f"\n📁 Files Generated:")
print(f"  Models: 5 files in ../models/")
print(f"  Results: ../outputs/final_test_results.csv")
print(f"  Predictions: ../outputs/best_model_predictions.csv")
print(f"  High-Value: ../outputs/high_value_customers.csv")
print(f"  Segments: ../outputs/segment_performance.csv")
print(f"  Feature Importance: 3 files in ../outputs/")

print(f"\n🎯 Best Model: {best_model['model']} with R² = {best_model['R2']:.4f}")
print(f"✅ Project Complete and Ready for GitHub!")
