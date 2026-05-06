"""Abstract base class for all prediction models."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class F1PredictionModel(ABC, BaseEstimator):
    """Abstract interface for F1 race prediction models."""
    
    def __init__(self, model_name: str, target_col: str = "finishing_position"):
        self.model_name = model_name
        self.target_col = target_col
        self.model_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
    @abstractmethod
    def _create_model(self):
        """Initialize the underlying ML model."""
        pass
    
    @abstractmethod
    def _get_param_grid(self) -> dict:
        """Return parameter grid for hyperparameter tuning."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Train the model with optional hyperparameter tuning."""
        self.feature_names_ = X.columns.tolist() if hasattr(X, 'columns') else None
        
        self.model_ = self._create_model()
        
        # Optional hyperparameter tuning
        if kwargs.get('tune', False):
            self._hyperparameter_tune(X, y, **kwargs)
        else:
            self.model_.fit(X, y)
        
        self.is_fitted_ = True
        logger.info(f"{self.model_name} fitted successfully")
        return self
    
    def _hyperparameter_tune(self, X, y, cv=5, scoring='neg_mean_absolute_error', **kwargs):
        """Perform grid search cross-validation for hyperparameter tuning."""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = self._get_param_grid()
        grid_search = GridSearchCV(
            self.model_, param_grid, cv=cv, scoring=scoring, 
            n_jobs=-1, verbose=1, return_train_score=True
        )
        grid_search.fit(X, y)
        
        self.model_ = grid_search.best_estimator_
        logger.info(f"Best params for {self.model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point predictions."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model_.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_bootstrap: int = 100) -> dict:
        """Generate predictions with confidence intervals via bootstrap."""
        point_preds = self.predict(X)
        
        # Bootstrap for uncertainty estimation
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
            try:
                boot_pred = self.model_.predict(X_boot)
                bootstrap_preds.append(boot_pred)
            except:
                continue
        
        if not bootstrap_preds:
            return {"point": point_preds, "lower": point_preds, "upper": point_preds}
        
        bootstrap_preds = np.array(bootstrap_preds)
        lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper = np.percentile(bootstrap_preds, 97.5, axis=0)
        
        return {
            "point": point_preds,
            "lower": lower,
            "upper": upper,
            "std": np.std(bootstrap_preds, axis=0)
        }
    
    def save(self, filepath: str):
        """Save trained model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model_,
            'feature_names': self.feature_names_,
            'config': self.get_params()
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'F1PredictionModel':
        """Load trained model from disk."""
        data = joblib.load(filepath)
        instance = cls(**data['config'])
        instance.model_ = data['model']
        instance.feature_names_ = data['feature_names']
        instance.is_fitted_ = True
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_feature_importance(self) -> pd.Series:
        """Extract feature importances if available."""
        if hasattr(self.model_, 'feature_importances_'):
            return pd.Series(self.model_.feature_importances_, index=self.feature_names_)
        elif hasattr(self.model_, 'coef_'):
            return pd.Series(np.abs(self.model_.coef_[0]), index=self.feature_names_)
        return pd.Series(dtype=float)