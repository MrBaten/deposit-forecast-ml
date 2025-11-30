"""
Combined Phase 3, 4, 5 Execution Script
Model Development, Evaluation, and Production Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from model_training import DepositForecaster
from model_evaluation import ModelEvaluator
from deployment_pipeline import DeploymentPipeline, BatchPredictor

print("="*80)
print("  PHASES 3, 4, 5: MODEL DEVELOPMENT & DEPLOYMENT")
print("="*80)

# ============================================================================
# PHASE 3: MODEL DEVELOPMENT
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: MODEL DEVELOPMENT")
print("="*80)

# Load splits
print("\nLoading train/validation/test data...")
train_df = pd.read_csv("../data/train_data.csv")
val_df = pd.read_csv("../data/val_data.csv")
test_df = pd.read_csv("../data/test_data.csv")

print(f"  Train: {train_df.shape}")
print(f"  Validation: {val_df.shape}")
print(f"  Test: {test_df.shape}")

# Prepare features and target
exclude_cols = ['customer_id', 'date', 'segment']

def prepare_features_target(df):
    feature_cols = [col for col in df.columns if col not in exclude_cols + ['deposit_amount']]
    X = df[feature_cols]
    y = df['deposit_amount']
    return X, y

X_train, y_train = prepare_features_target(train_df)
X_val, y_val = prepare_features_target(val_df)
X_test, y_test = prepare_features_target(test_df)

print(f"\nFeature matrix shapes:")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

# Initialize forecaster
forecaster = DepositForecaster()

# Train all models
print("\n" + "="*80)
print("TRAINING MODELS...")
print("="*80)

forecaster.train_all_models(X_train, y_train, X_val, y_val)

# Save models
forecaster.save_models("../models/")

# Save metrics
forecaster.save_metrics("../outputs/model_metrics.csv")

# Get best model
best_name, best_model, best_metrics = forecaster.get_best_model()
print("\n" + "="*80)
print("BEST MODEL")
print("="*80)
print(f"Model: {best_name}")
print(f"Validation R²: {best_metrics['val']['R2']:.4f}")
print(f"Validation MAE: ${best_metrics['val']['MAE']:.2f}")
print(f"Validation RMSE: ${best_metrics['val']['RMSE']:.2f}")

# ============================================================================
# PHASE 4: MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: MODEL EVALUATION")
print("="*80)

evaluator = ModelEvaluator()

# Evaluate all models on test set
print("\nEvaluating models on test set...")
test_results = []

for model_name, model in forecaster.models.items():
    test_metrics = evaluator.evaluate_on_test(model, X_test, y_test, model_name)
    test_results.append(test_metrics)
    
    print(f"\n{model_name.upper()}:")
    print(f"  MAE:  ${test_metrics['MAE']:.2f}")
    print(f"  RMSE: ${test_metrics['RMSE']:.2f}")
    print(f"  R²:   {test_metrics['R2']:.4f}")
    print(f"  MAPE: {test_metrics['MAPE']:.1f}%")

# Save test results
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv("../outputs/test_results.csv", index=False)
print("\n✓ Test results saved to: ../outputs/test_results.csv")

# Create visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

os.makedirs("../visualizations", exist_ok=True)

# Model comparison
metrics_df = pd.read_csv("../outputs/model_metrics.csv")
evaluator.plot_model_comparison(metrics_df, "../visualizations/model_comparison.png")
print("✓ Model comparison plot saved")

# Predictions vs actual for best model
evaluator.plot_predictions_vs_actual(best_name, f"../visualizations/{best_name}_predictions.png")
print(f"✓ {best_name} predictions plot saved")

# Time series for sample customers
for customer_id in [1, 50, 100]:
    try:
        evaluator.plot_time_series_sample(
            test_df, best_name, customer_id,
            f"../visualizations/{best_name}_customer_{customer_id}.png"
        )
        print(f"✓ Customer {customer_id} time series saved")
    except:
        pass

# Feature importance for tree-based models
if best_name in forecaster.feature_importance:
    evaluator.plot_feature_importance(
        forecaster.feature_importance[best_name],
        best_name,
        top_n=20,
        save_path=f"../visualizations/{best_name}_feature_importance.png"
    )
    print(f"✓ Feature importance plot saved")
    
    # Save feature importance to CSV
    forecaster.feature_importance[best_name].to_csv(
        f"../outputs/{best_name}_feature_importance.csv",
        index=False
    )

# ============================================================================
# PHASE 5: PRODUCTION PIPELINE
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: PRODUCTION PIPELINE")
print("="*80)

# Create prediction report
print("\nGenerating prediction report...")
best_model_pred = evaluator.predictions[best_name]
pipeline = DeploymentPipeline(f"../models/{best_name}.pkl")
report_df = pipeline.create_prediction_report(test_df, best_model_pred)

# High-value predictions
print("\nIdentifying high-value customers...")
high_value = pipeline.get_high_value_predictions(report_df, threshold=100.0)
high_value.to_csv("../outputs/high_value_predictions.csv", index=False)
print(f"✓ Found {len(high_value)} high-value predictions (>${100})")
print(f"  Total predicted volume: ${high_value['predicted_deposit'].sum():,.2f}")

# Batch prediction for next day
print("\nGenerating batch predictions for all customers...")
batch_predictor = BatchPredictor(f"../models/{best_name}.pkl")
batch_predictions = batch_predictor.predict_next_day_all_customers(
    "../data/customer_deposits_featured.csv",
    "../outputs/next_day_predictions.csv"
)

# Segment-wise analysis
print("\n" + "="*80)
print("SEGMENT-WISE PERFORMANCE ANALYSIS")
print("="*80)

test_df_with_pred = test_df.copy()
test_df_with_pred['predicted'] = best_model_pred

segment_performance = test_df_with_pred.groupby('segment').apply(
    lambda x: pd.Series({
        'count': len(x),
        'actual_mean': x['deposit_amount'].mean(),
        'predicted_mean': x['predicted'].mean(),
        'mae': np.mean(np.abs(x['deposit_amount'] - x['predicted'])),
        'rmse': np.sqrt(np.mean((x['deposit_amount'] - x['predicted'])**2))
    })
)

print("\nPerformance by customer segment:")
print(segment_performance.to_string())

segment_performance.to_csv("../outputs/segment_performance.csv")
print("\n✓ Segment performance saved to: ../outputs/segment_performance.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROJECT COMPLETION SUMMARY")
print("="*80)

summary = {
    'Total Customers': train_df['customer_id'].nunique(),
    'Training Records': len(train_df),
    'Test Records': len(test_df),
    'Features Used': X_train.shape[1],
    'Models Trained': len(forecaster.models),
    'Best Model': best_name,
    'Test R² Score': f"{test_results_df[test_results_df['model'] == best_name]['R2'].values[0]:.4f}",
    'Test MAE': f"${test_results_df[test_results_df['model'] == best_name]['MAE'].values[0]:.2f}",
    'Test RMSE': f"${test_results_df[test_results_df['model'] == best_name]['RMSE'].values[0]:.2f}",
}

for key, value in summary.items():
    print(f"{key:.<40} {value}")

print("\n" + "="*80)
print("✓ ALL PHASES COMPLETED SUCCESSFULLY!")
print("="*80)

print("\nGenerated Files:")
print("\nModels:")
print("  • ../models/*.pkl (trained models)")

print("\nMetrics & Results:")
print("  • ../outputs/model_metrics.csv (train/val metrics)")
print("  • ../outputs/test_results.csv (test set evaluation)")
print("  • ../outputs/predictions.csv (detailed predictions)")
print("  • ../outputs/high_value_predictions.csv (high-value customers)")
print("  • ../outputs/next_day_predictions.csv (batch predictions)")
print("  • ../outputs/segment_performance.csv (segment analysis)")
if best_name in forecaster.feature_importance:
    print(f"  • ../outputs/{best_name}_feature_importance.csv")

print("\nVisualizations:")
print("  • ../visualizations/model_comparison.png")
print(f"  • ../visualizations/{best_name}_predictions.png")
print(f"  • ../visualizations/{best_name}_customer_*.png")
if best_name in forecaster.feature_importance:
    print(f"  • ../visualizations/{best_name}_feature_importance.png")

print("\n" + "="*80)
print("PROJECT READY FOR PRODUCTION DEPLOYMENT!")
print("="*80)

print("\nNext Steps:")
print("  1. Review model performance and visualizations")
print("  2. Deploy prediction API for real-time forecasting")
print("  3. Schedule batch predictions for daily operations")
print("  4. Set up monitoring and retraining pipeline")
print("  5. Push code to GitHub repository")

print("\n✅ Customer Deposit Forecasting System Complete!")
