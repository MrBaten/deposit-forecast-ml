"""
Model Training Module for Customer Deposit Forecasting
Implements multiple model types: Linear, Tree-based, and Deep Learning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try importing advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


class DepositForecaster:
    """Multi-model forecaster for customer deposits."""
    
    def __init__(self):
        """Initialize forecaster."""
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Handle negative predictions
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        non_zero_mask = y_true > 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        metrics = {
            'model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train linear regression model."""
        print("\n" + "="*80)
        print("TRAINING: Linear Regression")
        print("="*80)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train.values, train_pred, 'Linear Regression (Train)')
        val_metrics = self.calculate_metrics(y_val.values, val_pred, 'Linear Regression (Val)')
        
        print(f"\nTraining Metrics:")
        print(f"  MAE:  ${train_metrics['MAE']:.2f}")
        print(f"  RMSE: ${train_metrics['RMSE']:.2f}")
        print(f"  R²:   {train_metrics['R2']:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"  MAE:  ${val_metrics['MAE']:.2f}")
        print(f"  RMSE: ${val_metrics['RMSE']:.2f}")
        print(f"  R²:   {val_metrics['R2']:.4f}")
        
        # Store
        self.models['linear_regression'] = model
        self.metrics['linear_regression'] = {'train': train_metrics, 'val': val_metrics}
        
        return val_metrics
    
    def train_ridge_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series, alpha: float = 1.0) -> Dict:
        """Train Ridge regression model."""
        print("\n" + "="*80)
        print(f"TRAINING: Ridge Regression (alpha={alpha})")
        print("="*80)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train.values, train_pred, 'Ridge (Train)')
        val_metrics = self.calculate_metrics(y_val.values, val_pred, 'Ridge (Val)')
        
        print(f"\nValidation Metrics:")
        print(f"  MAE:  ${val_metrics['MAE']:.2f}")
        print(f"  RMSE: ${val_metrics['RMSE']:.2f}")
        print(f"  R²:   {val_metrics['R2']:.4f}")
        
        # Store
        self.models['ridge_regression'] = model
        self.metrics['ridge_regression'] = {'train': train_metrics, 'val': val_metrics}
        
        return val_metrics
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_estimators: int = 100, max_depth: int = 20) -> Dict:
        """Train Random Forest model."""
        print("\n" + "="*80)
        print(f"TRAINING: Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
        print("="*80)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("Training Random Forest...")
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train.values, train_pred, 'Random Forest (Train)')
        val_metrics = self.calculate_metrics(y_val.values, val_pred, 'Random Forest (Val)')
        
        print(f"\nValidation Metrics:")
        print(f"  MAE:  ${val_metrics['MAE']:.2f}")
        print(f"  RMSE: ${val_metrics['RMSE']:.2f}")
        print(f"  R²:   {val_metrics['R2']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:.<50} {row['importance']:.4f}")
        
        # Store
        self.models['random_forest'] = model
        self.metrics['random_forest'] = {'train': train_metrics, 'val': val_metrics}
        self.feature_importance['random_forest'] = feature_importance
        
        return val_metrics
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost not available. Skipping...")
            return {}
        
        print("\n" + "="*80)
        print("TRAINING: XGBoost")
        print("="*80)
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        print("Training XGBoost...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train.values, train_pred, 'XGBoost (Train)')
        val_metrics = self.calculate_metrics(y_val.values, val_pred, 'XGBoost (Val)')
        
        print(f"\nValidation Metrics:")
        print(f"  MAE:  ${val_metrics['MAE']:.2f}")
        print(f"  RMSE: ${val_metrics['RMSE']:.2f}")
        print(f"  R²:   {val_metrics['R2']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:.<50} {row['importance']:.4f}")
        
        # Store
        self.models['xgboost'] = model
        self.metrics['xgboost'] = {'train': train_metrics, 'val': val_metrics}
        self.feature_importance['xgboost'] = feature_importance
        
        return val_metrics
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            print("\nLightGBM not available. Skipping...")
            return {}
        
        print("\n" + "="*80)
        print("TRAINING: LightGBM")
        print("="*80)
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        print("Training LightGBM...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        train_metrics = self.calculate_metrics(y_train.values, train_pred, 'LightGBM (Train)')
        val_metrics = self.calculate_metrics(y_val.values, val_pred, 'LightGBM (Val)')
        
        print(f"\nValidation Metrics:")
        print(f"  MAE:  ${val_metrics['MAE']:.2f}")
        print(f"  RMSE: ${val_metrics['RMSE']:.2f}")
        print(f"  R²:   {val_metrics['R2']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:.<50} {row['importance']:.4f}")
        
        # Store
        self.models['lightgbm'] = model
        self.metrics['lightgbm'] = {'train': train_metrics, 'val': val_metrics}
        self.feature_importance['lightgbm'] = feature_importance
        
        return val_metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series):
        """Train all available models."""
        print("="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        # Baseline models
        self.train_linear_regression(X_train, y_train, X_val, y_val)
        self.train_ridge_regression(X_train, y_train, X_val, y_val, alpha=1.0)
        
        # Tree-based models
        self.train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=20)
        
        # Gradient boosting
        if XGBOOST_AVAILABLE:
            self.train_xgboost(X_train, y_train, X_val, y_val)
        
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        print("\n" + "="*80)
        print("✓ ALL MODELS TRAINED")
        print("="*80)
    
    def get_best_model(self) -> Tuple[str, Any, Dict]:
        """Get best model based on validation R²."""
        best_r2 = -np.inf
        best_model_name = None
        
        for model_name, metrics in self.metrics.items():
            val_r2 = metrics['val']['R2']
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_model_name = model_name
        
        return best_model_name, self.models[best_model_name], self.metrics[best_model_name]
    
    def save_models(self, output_dir: str = "../models/"):
        """Save all trained models."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nSaving models...")
        for model_name, model in self.models.items():
            filepath = f"{output_dir}{model_name}.pkl"
            joblib.dump(model, filepath)
            print(f"  ✓ Saved: {filepath}")
    
    def save_metrics(self, output_path: str = "../outputs/model_metrics.csv"):
        """Save metrics to CSV."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        metrics_list = []
        for model_name, metrics in self.metrics.items():
            for split in ['train', 'val']:
                if split in metrics:
                    row = metrics[split].copy()
                    row['split'] = split
                    metrics_list.append(row)
        
        df = pd.DataFrame(metrics_list)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Metrics saved to: {output_path}")


if __name__ == "__main__":
    print("Model training module loaded successfully!")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
