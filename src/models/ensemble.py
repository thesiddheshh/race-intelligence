"""Ensemble model combining multiple predictors."""
from .base import F1PredictionModel
from .rf import RandomForestModel
from .xgb import XGBoostModel
from .nn import NeuralNetworkModel
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression

class EnsembleModel(F1PredictionModel):
    """Stacking ensemble of multiple F1 prediction models."""
    
    def __init__(self, ensemble_type: str = "stacking", **kwargs):
        self.ensemble_type = ensemble_type
        super().__init__(model_name=f"Ensemble_{ensemble_type}", **kwargs)
    
    def _create_model(self):
        # Initialize base models
        base_models = [
            ('rf', RandomForestModel().model_),
            ('xgb', XGBoostModel().model_),
            ('nn', NeuralNetworkModel().model_)
        ]
        
        if self.ensemble_type == "voting":
            return VotingRegressor(estimators=base_models, weights=[0.4, 0.4, 0.2])
        elif self.ensemble_type == "stacking":
            return StackingRegressor(
                estimators=base_models,
                final_estimator=LinearRegression(),
                cv=5,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
    
    def _get_param_grid(self) -> dict:
        # Ensemble tuning focuses on weights and final estimator
        return {
            'weights': [[0.33, 0.33, 0.34], [0.4, 0.4, 0.2], [0.5, 0.3, 0.2]],
            'final_estimator__fit_intercept': [True, False]
        } if self.ensemble_type == "voting" else {}