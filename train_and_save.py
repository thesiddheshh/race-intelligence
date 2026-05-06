"""
F1 2026 Model Training & Export Script
Trains ML models and saves them in thesis_viz.py compatible format.

Usage: python train_and_save.py
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import fastf1
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CACHE_DIR = Path(__file__).parent / ".f1_cache"
MODEL_DIR = Path(__file__).parent / "models"
GRID_FILE = Path(__file__).parent / "data" / "grid_2026.json"

CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ==================== DATA LOADING ====================
def load_grid():
    if not GRID_FILE.exists():
        raise FileNotFoundError(f"Grid file missing: {GRID_FILE}")
    with open(GRID_FILE, 'r') as f:
        return json.load(f)

def get_team_performance(grid):
    points = {
        "Red Bull Racing": 860, "McLaren": 650, "Ferrari": 610,
        "Mercedes": 480, "Aston Martin": 92, "Williams": 28,
        "RB": 46, "Haas": 54, "Alpine": 13, "Audi": 0, "Cadillac": 0
    }
    team_scores = {}
    for team in grid["teams"].keys():
        key = team
        if team not in points:
            if "Red Bull" in team: key = "Red Bull Racing"
            elif "Sauber" in team or "Audi" in team: key = "Audi"
            elif "Visa Cash App" in team or team == "RB": key = "RB"
        team_scores[team] = points.get(key, 10)
    max_pts = max(team_scores.values())
    return {t: pts/max_pts for t, pts in team_scores.items()}

def load_or_generate_data(grid):
    """Try FastF1, fall back to realistic synthetic data for training."""
    try:
        print("📡 Attempting to load real FastF1 race data...")
        schedule = fastf1.get_event_schedule(2025)
        event = schedule[schedule['EventFormat'] == 'conventional'].iloc[0]
        session = fastf1.get_session(2025, event['EventName'], 'R')
        session.load(telemetry=False, weather=False)
        laps = session.laps.copy()
        laps = laps[['Driver', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'Position']]
        for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
            laps[f'{col}_s'] = laps[col].dt.total_seconds()
        laps = laps.dropna(subset=['LapTime_s'])
        laps = laps[laps['LapTime_s'] > 60]
        print(f"✅ Loaded {len(laps)} real laps from {event['EventName']}")
        return laps
    except Exception as e:
        print(f"⚠️ FastF1 failed: {e}")
        print("🔄 Generating realistic synthetic training data...")
        np.random.seed(42)
        drivers = list(grid["drivers"].keys())
        n_laps_per_driver = 15
        data = {
            'Driver': np.repeat(drivers, n_laps_per_driver),
            'LapTime_s': np.random.normal(95.5, 2.5, len(drivers)*n_laps_per_driver),
            'Sector1Time_s': np.random.normal(30.2, 0.8, len(drivers)*n_laps_per_driver),
            'Sector2Time_s': np.random.normal(32.1, 0.9, len(drivers)*n_laps_per_driver),
            'Sector3Time_s': np.random.normal(26.4, 0.7, len(drivers)*n_laps_per_driver),
            'Compound': np.random.choice(['SOFT', 'MEDIUM', 'HARD'], len(drivers)*n_laps_per_driver),
            'Position': np.random.uniform(1, 22, len(drivers)*n_laps_per_driver)
        }
        return pd.DataFrame(data)

# ==================== FEATURE ENGINEERING ====================
def engineer_features(laps, grid, weather=None):
    if weather is None:
        weather = {'rain_probability': 0.15, 'temperature': 22.0}
        
    driver_to_team = {code: info["team"] for code, info in grid["drivers"].items()}
    team_perf = get_team_performance(grid)
    
    driver_stats = laps.groupby('Driver').agg(
        lap_time_mean=('LapTime_s', 'mean'),
        lap_time_std=('LapTime_s', 'std'),
        lap_time_min=('LapTime_s', 'min'),
        lap_count=('LapTime_s', 'count'),
        sector1_mean=('Sector1Time_s', 'mean'),
        sector2_mean=('Sector2Time_s', 'mean'),
        sector3_mean=('Sector3Time_s', 'mean'),
        avg_position=('Position', 'mean'),
        compound=('Compound', lambda x: x.mode()[0] if not x.mode().empty else 'MEDIUM')
    ).reset_index()
    
    driver_stats['SectorTotal_s'] = driver_stats[['sector1_mean', 'sector2_mean', 'sector3_mean']].sum(axis=1)
    driver_stats['LapConsistency'] = driver_stats['lap_time_std'].fillna(2.0)
    driver_stats['Team'] = driver_stats['Driver'].map(driver_to_team)
    driver_stats['TeamPerformance'] = driver_stats['Team'].map(team_perf).fillna(0.5)
    driver_stats['WeatherPenalty'] = weather['rain_probability'] * 2.5
    driver_stats['TempAdjustment'] = np.abs(weather['temperature'] - 22) * 0.05
    driver_stats['RecentForm'] = np.clip(driver_stats['lap_count'] / 20, 0.5, 1.0)
    
    compound_map = {'SOFT': 0.95, 'MEDIUM': 1.0, 'HARD': 1.05, 'INTERMEDIATE': 1.15, 'WET': 1.25}
    driver_stats['CompoundFactor'] = driver_stats['compound'].map(compound_map).fillna(1.0)
    driver_stats['QualiProxy'] = driver_stats['avg_position'].rank(method='min')
    
    feature_cols = ['lap_time_mean', 'LapConsistency', 'SectorTotal_s', 'TeamPerformance', 
                    'WeatherPenalty', 'TempAdjustment', 'RecentForm', 'CompoundFactor', 
                    'QualiProxy', 'lap_count']
    
    features = driver_stats[['Driver'] + feature_cols].copy().dropna(subset=feature_cols)
    return features, feature_cols

# ==================== MODEL TRAINING ====================
def train_and_save_models(features, feature_cols, target_col='lap_time_mean'):
    X = features[feature_cols]
    y = features[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'random_forest': RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42),
        'ridge_regression': Ridge(alpha=1.0)
    }
    
    print("\n🏋️ Training models...")
    for name, model in models.items():
        model.fit(X_scaled, y)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        mae = -cv_scores.mean()
        print(f"  ✅ {name.replace('_', ' ').title():<20} | CV MAE: {mae:.3f}s")
        
        # Save in thesis_viz.py expected format
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'config': {
                'cv_mae': mae,
                'cv_std': cv_scores.std(),
                'n_features': len(feature_cols),
                'n_samples': len(y),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        save_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model_data, save_path)
        print(f"     💾 Saved to {save_path}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("🏁 F1 2026 Model Training Pipeline\n" + "="*40)
    
    grid = load_grid()
    laps = load_or_generate_data(grid)
    features, feature_cols = engineer_features(laps, grid)
    
    print(f"📊 Dataset: {len(features)} drivers | {len(feature_cols)} features")
    train_and_save_models(features, feature_cols)
    
    print("\n🎉 Training complete! Run `streamlit run thesis_viz.py` to generate thesis figures.")