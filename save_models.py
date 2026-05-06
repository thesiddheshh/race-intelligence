import joblib
from pathlib import Path

# Ensure these variables exist from your training pipeline
# trained_model = your_trained_model
# feature_scaler = your_fitted_scaler

model_data = {
    'model': trained_model,
    'scaler': feature_scaler,
    'feature_names': ['lap_time_mean', 'LapConsistency', 'SectorTotal_s', 
                      'TeamPerformance', 'WeatherPenalty', 'TempAdjustment',
                      'RecentForm', 'CompoundFactor', 'QualiProxy', 'lap_count']
}

Path("models").mkdir(exist_ok=True)
joblib.dump(model_data, "models/gradient_boosting_2026.pkl")
print("✅ Model saved successfully to models/gradient_boosting_2026.pkl")