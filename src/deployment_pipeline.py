"""
Production Prediction Pipeline
Deploy-ready prediction API and batch processing.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DeploymentPipeline:
    """Production-ready prediction pipeline."""
    
    def __init__(self, model_path: str):
        """
        Initialize pipeline with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        self.model_name = model_path.split('/')[-1].replace('.pkl', '')
        print(f"✓ Loaded model: {self.model_name}")
    
    def predict_single_customer(self, customer_features: Dict) -> float:
        """
        Predict next-day deposit for single customer.
        
        Args:
            customer_features: Dictionary of feature values
            
        Returns:
            Predicted deposit amount
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([customer_features])
        
        # Predict
        prediction = self.model.predict(features_df)[0]
        prediction = max(0, prediction)  # No negative predictions
        
        return prediction
    
    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict for batch of customers.
        
        Args:
            features_df: DataFrame with features for multiple customers
            
        Returns:
            Array of predictions
        """
        predictions = self.model.predict(features_df)
        predictions = np.maximum(predictions, 0)  # No negative predictions
        
        return predictions
    
    def create_prediction_report(self, test_df: pd.DataFrame, predictions: np.ndarray,
                                output_path: str = "../outputs/predictions.csv"):
        """
        Create detailed prediction report.
        
        Args:
            test_df: Test DataFrame with customer_id, date, actual amounts
            predictions: Model predictions
            output_path: Where to save the report
        """
        report_df = pd.DataFrame({
            'customer_id': test_df['customer_id'].values,
            'date': test_df['date'].values,
            'actual_deposit': test_df['deposit_amount'].values,
            'predicted_deposit': predictions,
            'absolute_error': np.abs(test_df['deposit_amount'].values - predictions),
            'percentage_error': np.where(
                test_df['deposit_amount'].values > 0,
                100 * np.abs(test_df['deposit_amount'].values - predictions) / test_df['deposit_amount'].values,
                0
            ),
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        report_df.to_csv(output_path, index=False)
        print(f"✓ Prediction report saved to: {output_path}")
        
        return report_df
    
    def get_high_value_predictions(self, report_df: pd.DataFrame, 
                                   threshold: float = 100.0) -> pd.DataFrame:
        """
        Filter predictions for high-value deposits.
        
        Args:
            report_df: Prediction report DataFrame
            threshold: Minimum predicted amount
            
        Returns:
            Filtered DataFrame
        """
        high_value = report_df[report_df['predicted_deposit'] >= threshold].copy()
        high_value = high_value.sort_values('predicted_deposit', ascending=False)
        
        return high_value
    
    def get_prediction_confidence_intervals(self, predictions: np.ndarray,
                                           confidence: float = 0.95) -> Dict:
        """
        Calculate confidence intervals (simplified approach).
        
        Args:
            predictions: Array of predictions
            confidence: Confidence level
            
        Returns:
            Dictionary with confidence bounds
        """
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Using normal approximation
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_score * std_pred / np.sqrt(len(predictions))
        
        return {
            'mean_prediction': mean_pred,
            'confidence_level': confidence,
            'lower_bound': mean_pred - margin,
            'upper_bound': mean_pred + margin
        }


class BatchPredictor:
    """Batch prediction for all customers."""
    
    def __init__(self, model_path: str):
        """Initialize batch predictor."""
        self.pipeline = DeploymentPipeline(model_path)
    
    def predict_next_day_all_customers(self, featured_data_path: str,
                                       output_path: str = "../outputs/batch_predictions.csv"):
        """
        Predict next-day deposits for all customers.
        
        Args:
            featured_data_path: Path to featured dataset
            output_path: Where to save predictions
        """
        print("Loading data for batch prediction...")
        df = pd.read_csv(featured_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Get the latest date for each customer
        latest_data = df.sort_values('date').groupby('customer_id').tail(1).copy()
        
        # Prepare features
        exclude_cols = ['customer_id', 'date', 'segment', 'deposit_amount']
        feature_cols = [col for col in latest_data.columns if col not in exclude_cols]
        
        X = latest_data[feature_cols]
        
        # Predict
        print(f"Predicting for {len(latest_data)} customers...")
        predictions = self.pipeline.predict_batch(X)
        
        # Create output
        output_df = pd.DataFrame({
            'customer_id': latest_data['customer_id'].values,
            'segment': latest_data['segment'].values,
            'last_known_date': latest_data['date'].values,
            'next_day_prediction': predictions,
            'prediction_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        output_df = output_df.sort_values('next_day_prediction', ascending=False)
        output_df.to_csv(output_path, index=False)
        
        print(f"✓ Batch predictions saved to: {output_path}")
        print(f"\nSummary:")
        print(f"  Total customers: {len(output_df)}")
        print(f"  Total predicted volume: ${output_df['next_day_prediction'].sum():,.2f}")
        print(f"  Average prediction: ${output_df['next_day_prediction'].mean():.2f}")
        print(f"  Customers with predicted deposits > $0: {(output_df['next_day_prediction'] > 1).sum()}")
        
        return output_df


def create_prediction_api_example():
    """Example of how to use the prediction API."""
    
    example_code = '''
# Example: How to use the Prediction API

from deployment_pipeline import DeploymentPipeline

# 1. Initialize pipeline with trained model
pipeline = DeploymentPipeline(model_path="../models/xgboost.pkl")

# 2. Single customer prediction
customer_features = {
    'deposit_lag_1d': 100.0,
    'deposit_lag_7d': 150.0,
    'rolling_mean_7d': 120.0,
    'rolling_std_7d': 30.0,
    'wow_change': 10.0,
    'day_of_week': 5,  # Saturday
    'is_weekend': 1,
    # ... include all 51 features
}

prediction = pipeline.predict_single_customer(customer_features)
print(f"Predicted deposit: ${prediction:.2f}")

# 3. Batch prediction for all customers
from deployment_pipeline import BatchPredictor

batch_predictor = BatchPredictor(model_path="../models/xgboost.pkl")
predictions = batch_predictor.predict_next_day_all_customers(
    featured_data_path="../data/customer_deposits_featured.csv",
    output_path="../outputs/tomorrows_predictions.csv"
)

# 4. Get high-value customers
high_value = pipeline.get_high_value_predictions(predictions, threshold=100.0)
print(f"High-value customers: {len(high_value)}")
'''
    
    return example_code


if __name__ == "__main__":
    print("="*80)
    print("DEPLOYMENT PIPELINE")
    print("="*80)
    print("\nThis module provides production-ready prediction capabilities:")
    print("  1. Single customer predictions")
    print("  2. Batch predictions for all customers")
    print("  3. Prediction reports with confidence intervals")
    print("  4. High-value customer filtering")
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print(create_prediction_api_example())
