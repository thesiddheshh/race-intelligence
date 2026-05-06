"""Random Forest Regressor implementation."""
from .base import F1PredictionModel
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(F1PredictionModel):
    """Random Forest model for F1 performance prediction."""
    
    def __init__(self, **kwargs):
        super().__init__(model_name="RandomForest", **kwargs)
    
    def _create_model(self):
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    def _get_param_grid(self) -> dict:
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }