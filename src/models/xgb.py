"""XGBoost/LightGBM implementation."""
from .base import F1PredictionModel
import numpy as np

class XGBoostModel(F1PredictionModel):
    """XGBoost model for F1 performance prediction."""
    
    def __init__(self, use_lightgbm: bool = False, **kwargs):
        self.use_lightgbm = use_lightgbm
        model_type = "LightGBM" if use_lightgbm else "XGBoost"
        super().__init__(model_name=model_type, **kwargs)
    
    def _create_model(self):
        if self.use_lightgbm:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        else:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
    
    def _get_param_grid(self) -> dict:
        base_params = {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [6, 8, 10],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        if self.use_lightgbm:
            base_params.update({
                'num_leaves': [20, 31, 40],
                'min_child_samples': [10, 20, 30]
            })
        else:
            base_params.update({
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            })
        return base_params