"""Comprehensive model evaluation utilities."""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model comparison and evaluation reporting."""
    
    METRICS = ['mae', 'rmse', 'r2', 'mape']
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    @staticmethod
    def cross_validate(model, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'neg_mean_absolute_error') -> Dict[str, float]:
        """Perform k-fold cross-validation."""
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            'mean_score': -scores.mean(),  # Convert back to positive for MAE
            'std_score': scores.std(),
            'scores': -scores  # Individual fold scores
        }
    
    @staticmethod
    def compare_models(models: Dict[str, any], X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5) -> pd.DataFrame:
        """Compare multiple models across all metrics."""
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Cross-validation
            cv_result = ModelEvaluator.cross_validate(model, X, y, cv=cv)
            
            # Train/test split for detailed metrics
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = ModelEvaluator.calculate_metrics(y_test, y_pred)
            metrics.update({
                'model': name,
                'cv_mae': cv_result['mean_score'],
                'cv_std': cv_result['std_score']
            })
            results.append(metrics)
        
        return pd.DataFrame(results).sort_values('cv_mae')
    
    @staticmethod
    def plot_comparison(results_df: pd.DataFrame, metric: str = 'mae'):
        """Generate comparison visualization."""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x='model', y=metric, palette='viridis')
        plt.title(f'Model Comparison by {metric.upper()}')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predicted vs Actual"):
        """Generate prediction scatter plot with identity line."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
               'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig