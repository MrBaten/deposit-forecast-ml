"""
Model Evaluation Module
Comprehensive evaluation, visualization, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate and visualize model performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.predictions = {}
        self.actuals = {}
        
    def evaluate_on_test(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                        model_name: str) -> Dict:
        """Evaluate model on test set."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # No negative predictions
        
        # Store for later analysis
        self.predictions[model_name] = y_pred
        self.actuals[model_name] = y_test.values
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # MAPE for non-zero values
        non_zero_mask = y_test > 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / 
                                 y_test[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        return {
            'model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def plot_predictions_vs_actual(self, model_name: str, save_path: str = None):
        """Plot predicted vs actual values."""
        y_true = self.actuals[model_name]
        y_pred = self.predictions[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Deposit Amount ($)')
        axes[0].set_ylabel('Predicted Deposit Amount ($)')
        axes[0].set_title(f'{model_name}: Predicted vs Actual', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residuals histogram
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residual (Actual - Predicted)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residuals Distribution', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series_sample(self, test_df: pd.DataFrame, model_name: str,
                                customer_id: int = 1, save_path: str = None):
        """Plot time series for sample customer."""
        customer_data = test_df[test_df['customer_id'] == customer_id].copy()
        
        if len(customer_data) == 0:
            print(f"Warning: Customer {customer_id} not found in test set")
            return
        
        # Get predictions for this customer
        customer_indices = test_df[test_df['customer_id'] == customer_id].index
        y_pred = self.predictions[model_name][customer_indices]
        y_true = self.actuals[model_name][customer_indices]
        
        plt.figure(figsize=(14, 6))
        plt.plot(customer_data['date'].values, y_true, marker='o', linewidth=2, 
                label='Actual', color='blue')
        plt.plot(customer_data['date'].values, y_pred, marker='s', linewidth=2,
                label='Predicted', color='red', alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Deposit Amount ($)')
        plt.title(f'{model_name}: Customer {customer_id} - Actual vs Predicted', 
                 fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, metrics_df: pd.DataFrame, save_path: str = None):
        """Plot comparison of all models."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = metrics_df['model'].unique()
        x = np.arange(len(models))
        width = 0.35
        
        # MAE comparison
        train_mae = []
        val_mae = []
        for model in models:
            train_mae.append(metrics_df[(metrics_df['model'] == model) & 
                                       (metrics_df['split'] == 'train')]['MAE'].values[0])
            val_mae.append(metrics_df[(metrics_df['model'] == model) & 
                                     (metrics_df['split'] == 'val')]['MAE'].values[0])
        
        axes[0, 0].bar(x - width/2, train_mae, width, label='Train', alpha=0.8)
        axes[0, 0].bar(x + width/2, val_mae, width, label='Validation', alpha=0.8)
        axes[0, 0].set_ylabel('MAE ($)')
        axes[0, 0].set_title('Mean Absolute Error Comparison', fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        # RMSE comparison
        train_rmse = []
        val_rmse = []
        for model in models:
            train_rmse.append(metrics_df[(metrics_df['model'] == model) & 
                                        (metrics_df['split'] == 'train')]['RMSE'].values[0])
            val_rmse.append(metrics_df[(metrics_df['model'] == model) & 
                                      (metrics_df['split'] == 'val')]['RMSE'].values[0])
        
        axes[0, 1].bar(x - width/2, train_rmse, width, label='Train', alpha=0.8)
        axes[0, 1].bar(x + width/2, val_rmse, width, label='Validation', alpha=0.8)
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].set_title('Root Mean Squared Error Comparison', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        # R² comparison
        train_r2 = []
        val_r2 = []
        for model in models:
            train_r2.append(metrics_df[(metrics_df['model'] == model) & 
                                      (metrics_df['split'] == 'train')]['R2'].values[0])
            val_r2.append(metrics_df[(metrics_df['model'] == model) & 
                                    (metrics_df['split'] == 'val')]['R2'].values[0])
        
        axes[1, 0].bar(x - width/2, train_r2, width, label='Train', alpha=0.8)
        axes[1, 0].bar(x + width/2, val_r2, width, label='Validation', alpha=0.8)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('R² Score Comparison', fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3, axis='y')
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        # MAPE comparison (validation only, for non-zero values)
        val_mape = []
        for model in models:
            val_mape.append(metrics_df[(metrics_df['model'] == model) & 
                                      (metrics_df['split'] == 'val')]['MAPE'].values[0])
        
        axes[1, 1].bar(x, val_mape, alpha=0.8, color='teal')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].set_title('Mean Absolute Percentage Error (Validation)', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                               model_name: str, top_n: int = 20, save_path: str = None):
        """Plot feature importance."""
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values, alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'{model_name}: Top {top_n} Feature Importance', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("Model evaluation module loaded successfully!")
