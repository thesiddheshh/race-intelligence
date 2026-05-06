"""
F1 2026 Race Prediction System
Final Year Computer Science Thesis Project
Real data • Multiple ML models • Monte Carlo simulation • Streamlit dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import fastf1
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
CACHE_DIR = Path(__file__).parent / ".f1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

GRID_FILE = Path(__file__).parent / "data" / "grid_2026.json"

# Palette
C_BG       = "#08080f"
C_SURFACE  = "#0e0e1a"
C_GLASS    = "rgba(255,255,255,0.04)"
C_BORDER   = "rgba(255,255,255,0.08)"
C_RED      = "#e8001e"
C_RED2     = "#ff2d46"
C_GOLD     = "#f5c542"
C_SILVER   = "#c0c0c0"
C_BRONZE   = "#cd7f32"
C_ACCENT   = "#00d4ff"
C_TEXT     = "#f0f0f5"
C_MUTED    = "#6b6b85"

DRIVER_TICKER = [
    "VER", "NOR", "LEC", "PIA", "SAI", "RUS", "HAM", "ALO",
    "ANT", "HUL", "TSU", "ALB", "GAS", "OCO", "STR", "MAG",
    "BEA", "DOO", "BOR", "HAD"
]

# ==================== GRID LOADER ====================
def load_grid():
    if not GRID_FILE.exists():
        st.error(f"Grid file not found: {GRID_FILE}")
        st.stop()
    with open(GRID_FILE, 'r') as f:
        return json.load(f)

def get_driver_to_team_map(grid):
    return {code: info["team"] for code, info in grid["drivers"].items()}

def get_team_performance(grid, historical_points=None):
    points = historical_points or {
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
            elif "Kick" in team: key = "Audi"
        team_scores[team] = points.get(key, 10)
    max_pts = max(team_scores.values())
    return {t: pts/max_pts for t, pts in team_scores.items()}

# ==================== DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_race_data(year, event_name):
    try:
        schedule = fastf1.get_event_schedule(year)
        event = schedule[schedule['EventName'].str.contains(event_name, case=False, na=False)]
        if event.empty:
            return None, f"No event found: {event_name}"
        session = fastf1.get_session(year, event['EventName'].iloc[0], 'R')
        session.load(telemetry=False, weather=False)
        laps = session.laps.copy()
        if laps.empty:
            return None, "No lap data available"
        cols = ['Driver', 'Team', 'LapTime', 'Sector1Time', 'Sector2Time',
                'Sector3Time', 'Compound', 'Stint', 'Position']
        laps = laps[[c for c in cols if c in laps.columns]]
        for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
            if col in laps.columns:
                laps[f'{col}_s'] = laps[col].dt.total_seconds()
        laps = laps.dropna(subset=['LapTime_s'])
        laps = laps[laps['LapTime_s'] > 60]
        return laps, None
    except Exception as e:
        return None, str(e)

def get_weather_fallback():
    return {'rain_probability': 0.15, 'temperature': 22.0, 'humidity': 55, 'wind_speed': 3.2}

# ==================== FEATURE ENGINEERING ====================
def engineer_features(laps, grid, weather, track_name):
    driver_to_team = get_driver_to_team_map(grid)
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
    driver_stats['SectorTotal_s'] = (
        driver_stats['sector1_mean'] + driver_stats['sector2_mean'] + driver_stats['sector3_mean'])
    driver_stats['LapConsistency'] = driver_stats['lap_time_std'].fillna(2.0)
    driver_stats['Team'] = driver_stats['Driver'].map(driver_to_team)
    driver_stats['TeamPerformance'] = driver_stats['Team'].map(team_perf).fillna(0.5)
    driver_stats['WeatherPenalty'] = weather['rain_probability'] * 2.5
    driver_stats['TempAdjustment'] = np.abs(weather['temperature'] - 22) * 0.05
    driver_stats['RecentForm'] = np.clip(driver_stats['lap_count'] / 20, 0.5, 1.0)
    compound_map = {'SOFT': 0.95, 'MEDIUM': 1.0, 'HARD': 1.05, 'INTERMEDIATE': 1.15, 'WET': 1.25}
    driver_stats['CompoundFactor'] = driver_stats['compound'].map(compound_map).fillna(1.0)
    driver_stats['QualiProxy'] = driver_stats['avg_position'].rank(method='min')
    feature_cols = [
        'lap_time_mean', 'LapConsistency', 'SectorTotal_s',
        'TeamPerformance', 'WeatherPenalty', 'TempAdjustment',
        'RecentForm', 'CompoundFactor', 'QualiProxy', 'lap_count'
    ]
    features = driver_stats[['Driver'] + feature_cols].copy()
    features = features.dropna(subset=feature_cols)
    return features, driver_stats

# ==================== ML MODELS ====================
def train_models(X, y, model_type='all'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models = {}
    results = {}
    if model_type in ['all', 'rf']:
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        models['rf'] = {'model': rf, 'scaler': scaler, 'cv_mae': -cv_scores.mean()}
        results['Random Forest'] = {'cv_mae': -cv_scores.mean(), 'cv_std': cv_scores.std(),
            'feature_importance': pd.Series(rf.feature_importances_, index=X.columns)}
    if model_type in ['all', 'gb']:
        gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
        gb.fit(X_scaled, y)
        cv_scores = cross_val_score(gb, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        models['gb'] = {'model': gb, 'scaler': scaler, 'cv_mae': -cv_scores.mean()}
        results['Gradient Boosting'] = {'cv_mae': -cv_scores.mean(), 'cv_std': cv_scores.std(),
            'feature_importance': pd.Series(gb.feature_importances_, index=X.columns)}
    if model_type in ['all', 'ridge']:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        cv_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        models['ridge'] = {'model': ridge, 'scaler': scaler, 'cv_mae': -cv_scores.mean()}
        results['Ridge Regression'] = {'cv_mae': -cv_scores.mean(), 'cv_std': cv_scores.std()}
    return models, results

def predict_with_uncertainty(model_dict, X, feature_cols, n_bootstrap=50):
    model = model_dict['model']
    scaler = model_dict['scaler']
    X_scaled = scaler.transform(X[feature_cols])
    point_pred = model.predict(X_scaled)
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_boot = X_scaled[indices]
        try:
            pred = model.predict(X_boot)
            bootstrap_preds.append(pred)
        except:
            continue
    if bootstrap_preds:
        bootstrap_preds = np.array(bootstrap_preds)
        lower = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper = np.percentile(bootstrap_preds, 97.5, axis=0)
        std = np.std(bootstrap_preds, axis=0)
    else:
        lower, upper, std = point_pred, point_pred, np.zeros_like(point_pred)
    return {'point': point_pred, 'lower': lower, 'upper': upper, 'std': std}

# ==================== RACE SIMULATION ====================
def simulate_race(driver_features, predictions, n_simulations=500):
    results = []
    for sim in range(n_simulations):
        sim_positions = []
        for idx, row in driver_features.iterrows():
            driver = row['Driver']
            base_time = predictions['point'][idx]
            consistency = row['LapConsistency']
            noise = np.random.normal(0, consistency * 0.5)
            lap_time = base_time + noise
            lap_time *= (1 + row['WeatherPenalty'] * 0.1)
            lap_time *= row['CompoundFactor']
            dnf_prob = 0.02 + (1 - row['RecentForm']) * 0.03
            if np.random.random() < dnf_prob:
                sim_positions.append({'driver': driver, 'position': np.inf, 'dnf': True})
                continue
            grid_pos = row['QualiProxy']
            position = grid_pos + np.random.normal(0, 1.5)
            sim_positions.append({'driver': driver, 'position': max(1, position), 'dnf': False, 'predicted_time': lap_time})
        df = pd.DataFrame(sim_positions)
        df['finishing_position'] = np.nan
        valid = df[df['position'] != np.inf]
        if not valid.empty:
            ranks = valid['position'].rank(method='min').astype(int)
            df.loc[valid.index, 'finishing_position'] = ranks
        results.append(df)
    all_results = pd.concat(results, ignore_index=True)
    summary = []
    for driver in driver_features['Driver'].unique():
        driver_sims = all_results[all_results['driver'] == driver]
        positions = driver_sims['finishing_position'].dropna()
        if len(positions) == 0:
            continue
        summary.append({
            'driver': driver,
            'expected_position': positions.mean(),
            'position_std': positions.std(),
            'podium_prob': (positions <= 3).mean(),
            'win_prob': (positions == 1).mean(),
            'dnf_prob': driver_sims['dnf'].mean(),
            'top5_prob': (positions <= 5).mean()
        })
    return pd.DataFrame(summary).sort_values('expected_position')

# ==================== STREAMLIT DASHBOARD ====================
def inject_css():
    st.markdown(f"""
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {{
        font-family: 'Syne', sans-serif !important;
        color: {C_TEXT} !important;
    }}
    .stApp, .main, section.main > div {{
        background: {C_BG} !important;
    }}
    .block-container {{
        padding: 0 2rem 4rem 2rem !important;
        max-width: 1400px !important;
    }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: {C_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {C_RED}; border-radius: 2px; }}

    /* ── Racing ticker ── */
    .ticker-wrapper {{
        width: 100%;
        background: linear-gradient(90deg, {C_RED} 0%, #8b0000 50%, {C_RED} 100%);
        background-size: 200% 100%;
        animation: shimmer 3s linear infinite;
        overflow: hidden;
        padding: 6px 0;
        position: sticky;
        top: 0;
        z-index: 9999;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    @keyframes shimmer {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}
    .ticker-track {{
        display: flex;
        width: max-content;
        animation: ticker 28s linear infinite;
    }}
    .ticker-track:hover {{ animation-play-state: paused; }}
    @keyframes ticker {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    .ticker-item {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0 32px;
        font-family: 'Orbitron', monospace !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        color: rgba(255,255,255,0.95) !important;
        white-space: nowrap;
    }}
    .ticker-dot {{
        width: 5px; height: 5px;
        background: rgba(255,255,255,0.6);
        border-radius: 50%;
        display: inline-block;
    }}
    .car-icon {{
        display: inline-block;
        animation: bounce 0.5s ease-in-out infinite alternate;
        font-size: 13px;
    }}
    @keyframes bounce {{
        from {{ transform: translateY(0px); }}
        to   {{ transform: translateY(-2px); }}
    }}

    /* ── Hero header ── */
    .hero {{
        padding: 3.5rem 0 2.5rem 0;
        text-align: left;
        position: relative;
    }}
    .hero-eyebrow {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10px !important;
        letter-spacing: 5px !important;
        color: {C_RED} !important;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-family: 'Orbitron', monospace !important;
        font-size: clamp(2rem, 5vw, 3.8rem) !important;
        font-weight: 900 !important;
        line-height: 1.05 !important;
        letter-spacing: -1px !important;
        color: {C_TEXT} !important;
        margin: 0 0 1rem 0;
    }}
    .hero-title span {{
        color: {C_RED};
    }}
    .hero-sub {{
        font-family: 'Syne', sans-serif !important;
        font-size: 0.95rem !important;
        color: {C_MUTED} !important;
        font-weight: 400 !important;
    }}
    .hero-line {{
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, {C_RED}, transparent);
        margin: 1.5rem 0;
        border-radius: 2px;
    }}

    /* ── Section headers ── */
    .section-label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 9px !important;
        letter-spacing: 4px !important;
        color: {C_RED} !important;
        text-transform: uppercase !important;
        margin-bottom: 0.4rem !important;
    }}
    .section-title {{
        font-family: 'Orbitron', monospace !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
        margin-bottom: 1.5rem !important;
    }}

    /* ── Glass cards ── */
    .glass-card {{
        background: {C_GLASS};
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: border-color 0.3s ease, transform 0.2s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(232,0,30,0.3);
        transform: translateY(-2px);
    }}

    /* ── Podium cards ── */
    .podium-wrap {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin-bottom: 2.5rem;
    }}
    .podium-card {{
        position: relative;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        overflow: hidden;
        transition: transform 0.3s ease;
    }}
    .podium-card:hover {{ transform: translateY(-6px); }}
    .podium-card::before {{
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 1px;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
    }}
    .p1 {{
        background: linear-gradient(135deg, rgba(245,197,66,0.12) 0%, rgba(245,197,66,0.04) 100%);
        border: 1px solid rgba(245,197,66,0.35);
    }}
    .p2 {{
        background: linear-gradient(135deg, rgba(192,192,192,0.10) 0%, rgba(192,192,192,0.03) 100%);
        border: 1px solid rgba(192,192,192,0.25);
    }}
    .p3 {{
        background: linear-gradient(135deg, rgba(205,127,50,0.10) 0%, rgba(205,127,50,0.03) 100%);
        border: 1px solid rgba(205,127,50,0.25);
    }}
    .podium-rank {{
        font-family: 'Orbitron', monospace !important;
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        line-height: 1 !important;
        margin-bottom: 0.75rem !important;
    }}
    .p1 .podium-rank {{ color: {C_GOLD}; text-shadow: 0 0 30px rgba(245,197,66,0.4); }}
    .p2 .podium-rank {{ color: {C_SILVER}; text-shadow: 0 0 30px rgba(192,192,192,0.3); }}
    .p3 .podium-rank {{ color: {C_BRONZE}; text-shadow: 0 0 30px rgba(205,127,50,0.3); }}
    .podium-code {{
        font-family: 'Orbitron', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 900 !important;
        color: {C_TEXT} !important;
        letter-spacing: 3px !important;
        margin-bottom: 0.3rem !important;
    }}
    .podium-name {{
        font-size: 0.78rem !important;
        color: {C_MUTED} !important;
        margin-bottom: 0.2rem !important;
        font-weight: 500 !important;
    }}
    .podium-team {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        color: {C_RED} !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        margin-bottom: 1.2rem !important;
    }}
    .podium-stat-row {{
        display: flex;
        justify-content: space-around;
        gap: 0.5rem;
        margin-top: 1rem;
    }}
    .podium-stat {{
        flex: 1;
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.5rem 0.3rem;
    }}
    .podium-stat-val {{
        font-family: 'Orbitron', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
    }}
    .podium-stat-lbl {{
        font-size: 0.6rem !important;
        color: {C_MUTED} !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        margin-top: 2px;
    }}

    /* ── Driver standings table ── */
    .standings-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }}
    .standings-table th {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 2px !important;
        color: {C_MUTED} !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1rem !important;
        text-align: left;
        border-bottom: 1px solid {C_BORDER};
        background: transparent;
    }}
    .standings-table td {{
        padding: 0.7rem 1rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
        vertical-align: middle;
        font-family: 'Syne', sans-serif !important;
        color: {C_TEXT} !important;
        background: transparent !important;
    }}
    .standings-table tr:hover td {{
        background: rgba(255,255,255,0.03) !important;
    }}
    .pos-num {{
        font-family: 'Orbitron', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: {C_MUTED};
    }}
    .pos-num.top3 {{ color: {C_GOLD}; }}
    .pos-num.top10 {{ color: {C_ACCENT}; }}
    .driver-code-cell {{
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 2px !important;
    }}
    .prob-bar-wrap {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .prob-bar {{
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, {C_RED}, {C_RED2});
        opacity: 0.9;
        min-width: 2px;
    }}
    .prob-text {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        color: {C_TEXT};
        min-width: 36px;
    }}
    .team-badge {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: {C_MUTED} !important;
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
        padding: 2px 6px;
    }}
    .dnf-badge {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        color: {C_RED};
    }}

    /* ── Metric tiles ── */
    .metric-tile {{
        background: {C_GLASS};
        border: 1px solid {C_BORDER};
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        transition: border-color 0.25s;
    }}
    .metric-tile:hover {{ border-color: rgba(232,0,30,0.4); }}
    .metric-lbl {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.62rem !important;
        letter-spacing: 3px !important;
        color: {C_MUTED} !important;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }}
    .metric-val {{
        font-family: 'Orbitron', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
        line-height: 1;
    }}
    .metric-val.red {{ color: {C_RED} !important; }}
    .metric-sub {{
        font-size: 0.7rem !important;
        color: {C_MUTED} !important;
        margin-top: 0.2rem;
    }}

    /* ── Status pill ── */
    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 100px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase;
    }}
    .pill-green {{
        background: rgba(0,255,136,0.08);
        border: 1px solid rgba(0,255,136,0.2);
        color: #00ff88 !important;
    }}
    .pill-red {{
        background: rgba(232,0,30,0.08);
        border: 1px solid rgba(232,0,30,0.2);
        color: {C_RED} !important;
    }}
    .pill-dot {{
        width: 5px; height: 5px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse-dot 1.5s ease infinite;
    }}
    @keyframes pulse-dot {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(0.7); }}
    }}

    /* ── Sidebar override ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a0a14 0%, #080810 100%) !important;
        border-right: 1px solid {C_BORDER} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 2rem;
    }}
    .sidebar-logo {{
        font-family: 'Orbitron', monospace !important;
        font-size: 1rem !important;
        font-weight: 900 !important;
        color: {C_RED} !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        padding: 0 1.2rem 1.5rem 1.2rem;
        border-bottom: 1px solid {C_BORDER};
        margin-bottom: 1.5rem;
        display: block;
    }}
    .sidebar-section {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6rem !important;
        letter-spacing: 3px !important;
        color: {C_MUTED} !important;
        text-transform: uppercase !important;
        padding: 0 0 0.5rem 0 !important;
        margin-top: 1.5rem !important;
        border-bottom: 1px solid {C_BORDER} !important;
        margin-bottom: 0.8rem !important;
        display: block;
    }}
    .sidebar-stat {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
    }}
    .sidebar-stat-k {{
        font-size: 0.72rem !important;
        color: {C_MUTED} !important;
        font-weight: 500;
    }}
    .sidebar-stat-v {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        color: {C_TEXT} !important;
        font-weight: 600;
    }}

    /* ── Streamlit widget overrides ── */
    .stSelectbox label, .stRadio label, .stSlider label,
    .stCheckbox label, .stNumberInput label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 2px !important;
        color: {C_MUTED} !important;
        text-transform: uppercase !important;
    }}
    .stSelectbox > div > div {{
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 10px !important;
        color: {C_TEXT} !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.875rem !important;
    }}
    .stSelectbox > div > div:focus-within {{
        border-color: {C_RED} !important;
        box-shadow: 0 0 0 1px {C_RED} !important;
    }}
    .stRadio > div {{
        gap: 0.5rem !important;
    }}
    .stRadio > div > label {{
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid {C_BORDER} !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        color: {C_TEXT} !important;
    }}
    .stRadio > div > label:hover {{
        border-color: rgba(232,0,30,0.4) !important;
        background: rgba(232,0,30,0.06) !important;
    }}
    [data-baseweb="radio"] input:checked + div {{
        border-color: {C_RED} !important;
    }}
    .stSlider > div {{
        padding-top: 0.2rem;
    }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background: {C_RED} !important;
        border-color: {C_RED} !important;
    }}
    .stCheckbox > label {{
        font-family: 'Syne', sans-serif !important;
        font-size: 0.8rem !important;
        letter-spacing: 0 !important;
        text-transform: none !important;
        color: {C_TEXT} !important;
    }}
    div[data-baseweb="checkbox"] div {{
        border-color: {C_BORDER} !important;
        background: transparent !important;
    }}
    div[data-baseweb="checkbox"] input:checked ~ div {{
        background: {C_RED} !important;
        border-color: {C_RED} !important;
    }}

    /* ── Tabs ── */
    [data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: 1px solid {C_BORDER} !important;
        gap: 0 !important;
    }}
    [data-baseweb="tab"] {{
        background: transparent !important;
        border: none !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: {C_MUTED} !important;
        padding: 0.6rem 1.5rem !important;
        transition: color 0.2s !important;
    }}
    [data-baseweb="tab"]:hover {{ color: {C_TEXT} !important; }}
    [aria-selected="true"][data-baseweb="tab"] {{
        color: {C_RED} !important;
        border-bottom: 2px solid {C_RED} !important;
    }}
    [data-baseweb="tab-highlight"] {{
        background: {C_RED} !important;
        height: 2px !important;
    }}

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {{
        background: transparent !important;
        border: 1px solid {C_RED} !important;
        color: {C_RED} !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.25s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {C_RED} !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(232,0,30,0.3) !important;
        transform: translateY(-1px) !important;
    }}

    /* ── Alert / info boxes ── */
    .stInfo, [data-baseweb="notification"] {{
        background: rgba(0,212,255,0.05) !important;
        border: 1px solid rgba(0,212,255,0.15) !important;
        border-radius: 10px !important;
        color: {C_TEXT} !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.8rem !important;
    }}
    .stWarning {{
        background: rgba(245,197,66,0.05) !important;
        border: 1px solid rgba(245,197,66,0.15) !important;
    }}
    .stSuccess {{
        background: rgba(0,255,136,0.05) !important;
        border: 1px solid rgba(0,255,136,0.15) !important;
    }}

    /* ── Spinner ── */
    .stSpinner > div {{
        border-top-color: {C_RED} !important;
    }}

    /* ── Plotly charts ── */
    .js-plotly-plot {{ border-radius: 12px; overflow: hidden; }}

    /* ── Divider ── */
    hr {{
        border: none !important;
        border-top: 1px solid {C_BORDER} !important;
        margin: 2.5rem 0 !important;
    }}

    /* ── Head-to-head section ── */
    .h2h-card {{
        background: {C_GLASS};
        border: 1px solid {C_BORDER};
        border-radius: 16px;
        padding: 1.5rem;
    }}
    .h2h-bar-wrap {{
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0.5rem 0;
    }}
    .h2h-label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 1px !important;
        color: {C_MUTED} !important;
        min-width: 50px;
        text-transform: uppercase;
    }}
    .h2h-bar-outer {{
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }}
    .h2h-bar-inner {{
        height: 100%;
        background: linear-gradient(90deg, {C_RED}, {C_RED2});
        border-radius: 3px;
        transition: width 0.6s ease;
    }}
    .h2h-val {{
        font-family: 'Orbitron', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        color: {C_TEXT} !important;
        min-width: 45px;
        text-align: right;
    }}

    /* ── Footer ── */
    .footer {{
        text-align: center;
        padding: 2.5rem 0;
        border-top: 1px solid {C_BORDER};
        margin-top: 3rem;
    }}
    .footer-brand {{
        font-family: 'Orbitron', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 4px !important;
        color: {C_RED} !important;
        margin-bottom: 0.5rem;
    }}
    .footer-sub {{
        font-size: 0.72rem !important;
        color: {C_MUTED} !important;
        line-height: 1.8;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        display: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_ticker():
    drivers = DRIVER_TICKER
    items_html = ""
    for i, d in enumerate(drivers * 2):   # double for seamless loop
        items_html += f"""
        <span class="ticker-item">
            <span class="car-icon">&#x1F3CE;</span>
            {d}
            <span class="ticker-dot"></span>
        </span>"""
    st.markdown(f"""
    <div class="ticker-wrapper">
      <div class="ticker-track">{items_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_hero(race_name):
    st.markdown(f"""
    <div class="hero">
        <p class="hero-eyebrow">FORMULA ONE  ·  2026 SEASON  ·  PREDICTION SYSTEM</p>
        <h1 class="hero-title">RACE<br><span>INTELLIGENCE</span></h1>
        <div class="hero-line"></div>
        <p class="hero-sub">{race_name} &nbsp;·&nbsp; Multi-model ML &nbsp;·&nbsp; Monte Carlo simulation &nbsp;·&nbsp; Real FastF1 data</p>
    </div>
    """, unsafe_allow_html=True)


def render_section(label, title):
    st.markdown(f"""
    <p class="section-label">{label}</p>
    <p class="section-title">{title}</p>
    """, unsafe_allow_html=True)


def render_podium(simulation_results, grid, show_uncertainty):
    podium = simulation_results.head(3)
    cols = st.columns(3, gap="medium")
    classes = ["p1", "p2", "p3"]
    ranks   = ["01", "02", "03"]
    for i, (_, row) in enumerate(podium.iterrows()):
        driver  = row['driver']
        dinfo   = grid["drivers"].get(driver, {})
        name    = dinfo.get("name", driver)
        team    = dinfo.get("team", "—")
        win_p   = row['win_prob'] * 100
        pod_p   = row['podium_prob'] * 100
        exp_p   = row['expected_position']
        std_p   = row['position_std']
        with cols[i]:
            st.markdown(f"""
            <div class="podium-card {classes[i]}">
                <div class="podium-rank">{ranks[i]}</div>
                <div class="podium-code">{driver}</div>
                <div class="podium-name">{name}</div>
                <div class="podium-team">{team}</div>
                <div class="podium-stat-row">
                    <div class="podium-stat">
                        <div class="podium-stat-val">{win_p:.1f}%</div>
                        <div class="podium-stat-lbl">Win</div>
                    </div>
                    <div class="podium-stat">
                        <div class="podium-stat-val">{pod_p:.1f}%</div>
                        <div class="podium-stat-lbl">Podium</div>
                    </div>
                    <div class="podium-stat">
                        <div class="podium-stat-val">{exp_p:.2f}</div>
                        <div class="podium-stat-lbl">Expected</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_standings(simulation_results, grid, show_uncertainty):
    rows_html = ""
    for rank, (_, row) in enumerate(simulation_results.iterrows(), 1):
        driver = row['driver']
        dinfo  = grid["drivers"].get(driver, {})
        name   = dinfo.get("name", driver)
        team   = dinfo.get("team", "—")
        exp_p  = row['expected_position']
        std_p  = row['position_std']
        pod_p  = row['podium_prob']
        win_p  = row['win_prob']
        dnf_p  = row['dnf_prob']

        pos_class = "top3" if rank <= 3 else ("top10" if rank <= 10 else "")
        pos_label = f"{rank:02d}"

        exp_str = f"{exp_p:.2f} ± {std_p:.2f}" if show_uncertainty else f"{exp_p:.2f}"
        pod_bar_w = int(pod_p * 120)
        win_bar_w = int(win_p * 120)

        rows_html += f"""
        <tr>
            <td><span class="pos-num {pos_class}">{pos_label}</span></td>
            <td><span class="driver-code-cell">{driver}</span></td>
            <td style="color: #a0a0b8; font-size:0.83rem;">{name}</td>
            <td><span class="team-badge">{team[:14]}</span></td>
            <td style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#d0d0e0;">{exp_str}</td>
            <td>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{pod_bar_w}px;"></div>
                    <span class="prob-text">{pod_p:.1%}</span>
                </div>
            </td>
            <td>
                <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{win_bar_w}px; background: linear-gradient(90deg, #f5c542, #e88b00);"></div>
                    <span class="prob-text">{win_p:.1%}</span>
                </div>
            </td>
            <td><span class="dnf-badge">{dnf_p:.1%}</span></td>
        </tr>"""

    st.markdown(f"""
    <div class="glass-card" style="padding: 0; overflow: hidden; border-radius: 16px;">
    <table class="standings-table">
        <thead>
            <tr>
                <th>Pos</th>
                <th>Code</th>
                <th>Driver</th>
                <th>Team</th>
                <th>Expected</th>
                <th>Podium %</th>
                <th>Win %</th>
                <th>DNF</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


def make_plotly_theme():
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Syne, sans-serif", color=C_TEXT, size=11),
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )


def render_h2h(simulation_results, grid):
    drivers_list = simulation_results['driver'].tolist()
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.selectbox("Driver A", drivers_list, key="cmp1",
                          format_func=lambda d: f"{d} — {grid['drivers'].get(d,{}).get('name',d)}")
    with col2:
        d2 = st.selectbox("Driver B", drivers_list, index=min(1, len(drivers_list)-1), key="cmp2",
                          format_func=lambda d: f"{d} — {grid['drivers'].get(d,{}).get('name',d)}")

    if d1 == d2:
        return

    d1r = simulation_results[simulation_results['driver'] == d1].iloc[0]
    d2r = simulation_results[simulation_results['driver'] == d2].iloc[0]

    metrics = [
        ("Win",    d1r['win_prob'],    d2r['win_prob']),
        ("Podium", d1r['podium_prob'], d2r['podium_prob']),
        ("Top 5",  d1r['top5_prob'],   d2r['top5_prob']),
    ]

    def bars(driver, stats, label_side="left"):
        html = f"<div style='flex:1;'>"
        html += f"<div style='font-family:Orbitron,monospace; font-weight:900; font-size:1.1rem; color:#f0f0f5; margin-bottom:1rem; letter-spacing:2px;'>{driver}</div>"
        for lbl, val, _ in stats:
            w = int(val * 140)
            html += f"""
            <div style="margin-bottom:0.9rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:2px; color:{C_MUTED}; text-transform:uppercase;">{lbl}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; font-weight:700; color:{C_TEXT};">{val:.1%}</span>
                </div>
                <div style="height:5px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;">
                    <div style="height:100%; width:{w}px; background:linear-gradient(90deg,{C_RED},{C_RED2}); border-radius:3px; max-width:100%;"></div>
                </div>
            </div>"""
        html += "</div>"
        return html

    d1_stats = [(lbl, d1v, d2v) for lbl, d1v, d2v in metrics]
    d2_stats = [(lbl, d2v, d1v) for lbl, d1v, d2v in metrics]

    gap = d1r['expected_position'] - d2r['expected_position']
    leader = d1 if gap < 0 else d2

    st.markdown(f"""
    <div class="glass-card">
        <div style="display:flex; gap:3rem; align-items:flex-start;">
            {bars(d1, d1_stats)}
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:80px; padding-top:0.5rem;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:2px; color:{C_MUTED}; text-transform:uppercase; margin-bottom:0.5rem;">Gap</div>
                <div style="font-family:Orbitron,monospace; font-size:1.3rem; font-weight:900; color:{C_RED};">{abs(gap):.2f}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:{C_MUTED}; margin-top:0.25rem;">positions</div>
                <div style="font-family:Orbitron,monospace; font-size:0.65rem; font-weight:700; color:#00ff88; margin-top:1rem; letter-spacing:1px;">ADV: {leader}</div>
            </div>
            {bars(d2, d2_stats)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def setup_page():
    st.set_page_config(
        page_title="F1 2026 Prediction System",
        page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏎</text></svg>",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def main():
    setup_page()
    inject_css()
    render_ticker()

    # ── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<span class="sidebar-logo">F1 &nbsp;·&nbsp; 2026</span>', unsafe_allow_html=True)

        st.markdown('<span class="sidebar-section">Grand Prix</span>', unsafe_allow_html=True)

        grid = load_grid()

        year = st.selectbox("Season", [2024, 2025, 2026], index=1)

        try:
            schedule = fastf1.get_event_schedule(year)
            race_options = schedule[schedule['EventFormat'] == 'conventional']['EventName'].tolist()
        except:
            race_options = [
                "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
                "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
                "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
                "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
                "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
                "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
                "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
                "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
            ]

        selected_race = st.selectbox("Race", race_options)

        st.markdown('<span class="sidebar-section">Model</span>', unsafe_allow_html=True)
        model_choice = st.radio(
            "Algorithm",
            ["Random Forest", "Gradient Boosting", "Ridge Regression", "Ensemble"],
            index=1,
            label_visibility="collapsed"
        )

        st.markdown('<span class="sidebar-section">Simulation</span>', unsafe_allow_html=True)
        n_sims = st.slider("Monte Carlo runs", 100, 2000, 500, step=100)
        show_uncertainty = st.checkbox("Confidence intervals", value=True)

        st.markdown('<span class="sidebar-section">Weather</span>', unsafe_allow_html=True)
        use_custom = st.checkbox("Override conditions")
        if use_custom:
            rain = st.slider("Rain probability", 0, 100, 15) / 100
            temp = st.slider("Temperature °C", -10, 50, 22)
            weather = {'rain_probability': rain, 'temperature': temp, 'humidity': 55, 'wind_speed': 3.2}
        else:
            weather = get_weather_fallback()

        st.markdown('<span class="sidebar-section">Grid</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-stat"><span class="sidebar-stat-k">Teams</span><span class="sidebar-stat-v">{len(grid['teams'])}</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-k">Drivers</span><span class="sidebar-stat-v">{len(grid['drivers'])}</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-k">Season</span><span class="sidebar-stat-v">2026</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-k">Rain</span><span class="sidebar-stat-v">{weather['rain_probability']*100:.0f}%</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-k">Temp</span><span class="sidebar-stat-v">{weather['temperature']}°C</span></div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding: 1rem; background: rgba(232,0,30,0.06); border: 1px solid rgba(232,0,30,0.15); border-radius: 12px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:2px; color:#e8001e; text-transform:uppercase; margin-bottom:0.4rem;">Stack</div>
            <div style="font-size:0.72rem; color:#6b6b85; line-height:2;">FastF1 · scikit-learn<br>Plotly · Streamlit<br>Random Forest · GB<br>Monte Carlo</div>
        </div>
        """, unsafe_allow_html=True)

    # ── HERO ────────────────────────────────────────────────────────────────
    render_hero(selected_race)

    # ── LOAD DATA ───────────────────────────────────────────────────────────
    with st.spinner("Loading race data..."):
        laps, error = load_race_data(year, selected_race)
        data_source = "FastF1"

        if error or laps is None or laps.empty:
            data_source = "Synthetic fallback"
            driver_codes = list(grid["drivers"].keys())
            np.random.seed(42)
            laps = pd.DataFrame({
                'Driver': driver_codes * 10,
                'LapTime_s': np.random.uniform(92, 98, len(driver_codes) * 10),
                'Sector1Time_s': np.random.uniform(28, 32, len(driver_codes) * 10),
                'Sector2Time_s': np.random.uniform(30, 34, len(driver_codes) * 10),
                'Sector3Time_s': np.random.uniform(24, 28, len(driver_codes) * 10),
                'Compound': np.random.choice(['SOFT', 'MEDIUM', 'HARD'], len(driver_codes) * 10),
                'Position': np.random.uniform(1, 22, len(driver_codes) * 10)
            })

    # ── STATUS METRICS ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-lbl">Data Source</div>
            <div class="metric-val" style="font-size:1rem;">{data_source}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-lbl">Model</div>
            <div class="metric-val red" style="font-size:1rem;">{model_choice}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-lbl">Simulations</div>
            <div class="metric-val">{n_sims:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-tile">
            <div class="metric-lbl">Rain Risk</div>
            <div class="metric-val">{weather['rain_probability']*100:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FEATURE ENGINEERING ─────────────────────────────────────────────────
    with st.spinner("Engineering features..."):
        features, driver_stats = engineer_features(laps, grid, weather, selected_race)
        if features.empty:
            st.error("Feature engineering failed.")
            st.stop()

    feature_cols = [c for c in features.columns if c != 'Driver']
    X = features[feature_cols]
    y = features['lap_time_mean']

    # ── TRAIN ───────────────────────────────────────────────────────────────
    with st.spinner("Training model..."):
        model_key = {'Random Forest': 'rf', 'Gradient Boosting': 'gb',
                     'Ridge Regression': 'ridge', 'Ensemble': 'all'}[model_choice]
        models, eval_results = train_models(X, y, model_type=model_key)

        if model_choice == "Ensemble":
            preds = np.zeros(len(X))
            stds  = np.zeros(len(X))
            for k in ['rf', 'gb', 'ridge']:
                if k in models:
                    p = predict_with_uncertainty(models[k], features, feature_cols)
                    preds += p['point']; stds += p['std']
            n = len([k for k in ['rf','gb','ridge'] if k in models])
            preds /= n; stds /= n
            predictions = {'point': preds, 'lower': preds - stds, 'upper': preds + stds, 'std': stds}
        else:
            mk = {'Random Forest': 'rf', 'Gradient Boosting': 'gb', 'Ridge Regression': 'ridge'}[model_choice]
            predictions = predict_with_uncertainty(models[mk], features, feature_cols)

    # ── SIMULATE ────────────────────────────────────────────────────────────
    with st.spinner(f"Running {n_sims} simulations..."):
        simulation_results = simulate_race(features, predictions, n_simulations=n_sims)

    # ── PODIUM ──────────────────────────────────────────────────────────────
    st.markdown("---")
    render_section("Prediction output", "Predicted Podium")
    render_podium(simulation_results, grid, show_uncertainty)

    # ── STANDINGS ───────────────────────────────────────────────────────────
    render_section("Full classification", "Driver Rankings")
    render_standings(simulation_results, grid, show_uncertainty)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ──────────────────────────────────────────────────────────────
    render_section("Analytics", "Prediction Insights")

    tab1, tab2, tab3 = st.tabs(["POSITION DISTRIBUTION", "PROBABILITIES", "MODEL INSIGHTS"])
    theme = make_plotly_theme()

    with tab1:
        top_d = simulation_results.head(12)['driver'].tolist()
        fig = go.Figure()
        for driver in top_d:
            nm   = grid["drivers"].get(driver, {}).get("name", driver)
            row  = simulation_results[simulation_results['driver'] == driver].iloc[0]
            pos  = np.clip(np.random.normal(row['expected_position'], row['position_std'], 200), 1, 22)
            fig.add_trace(go.Violin(
                x=[nm]*len(pos), y=pos, name=nm,
                box_visible=True, meanline_visible=True,
                opacity=0.8,
                fillcolor=f"rgba(232,0,30,0.15)",
                line_color=C_RED,
                meanline_color=C_GOLD,
            ))
        fig.update_layout(
            **theme,
            title=dict(text="Finishing Position Distribution", font=dict(family="Orbitron", size=13, color=C_TEXT)),
            yaxis_autorange='reversed',
            yaxis_title="Finishing Position",
            showlegend=False,
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        prob_df = simulation_results.copy()
        prob_df['driver_name'] = prob_df['driver'].map(lambda d: grid["drivers"].get(d,{}).get("name",d))
        prob_df = prob_df.sort_values('podium_prob', ascending=True).tail(14)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=prob_df['driver_name'],
            x=prob_df['podium_prob'],
            orientation='h',
            name='Podium',
            marker=dict(
                color=prob_df['podium_prob'],
                colorscale=[[0, "rgba(232,0,30,0.3)"], [1, C_RED]],
                line=dict(width=0),
            ),
        ))
        fig.add_trace(go.Bar(
            y=prob_df['driver_name'],
            x=prob_df['win_prob'],
            orientation='h',
            name='Win',
            marker=dict(
                color=prob_df['win_prob'],
                colorscale=[[0, "rgba(245,197,66,0.3)"], [1, C_GOLD]],
                line=dict(width=0),
            ),
        ))
        fig.update_layout(
            **theme,
            title=dict(text="Podium & Win Probability", font=dict(family="Orbitron", size=13, color=C_TEXT)),
            barmode='overlay',
            xaxis_tickformat='.0%',
            height=460,
            legend=dict(
                font=dict(family="JetBrains Mono", size=9, color=C_MUTED),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom", y=1.02,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        eval_df = pd.DataFrame([
            {'Model': name, 'MAE': d['cv_mae'], 'Std': d['cv_std']}
            for name, d in eval_results.items()
        ]).sort_values('MAE')

        fig = go.Figure(go.Bar(
            x=eval_df['Model'], y=eval_df['MAE'],
            error_y=dict(type='data', array=eval_df['Std'], visible=True, color=C_MUTED),
            marker=dict(
                color=eval_df['MAE'],
                colorscale=[[0, C_RED], [0.5, "#ff6b35"], [1, C_GOLD]],
                line=dict(width=0),
            )
        ))
        fig.update_layout(
            **theme,
            title=dict(text="Cross-Validated MAE  —  Lower is Better", font=dict(family="Orbitron", size=13, color=C_TEXT)),
            yaxis_title="MAE (seconds)",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'Random Forest' in eval_results and 'feature_importance' in eval_results['Random Forest']:
            fi = eval_results['Random Forest']['feature_importance'].nlargest(8)
            fig2 = go.Figure(go.Bar(
                y=fi.index, x=fi.values, orientation='h',
                marker=dict(
                    color=fi.values,
                    colorscale=[[0, "rgba(0,212,255,0.3)"], [1, C_ACCENT]],
                    line=dict(width=0),
                )
            ))
            fig2.update_layout(
                **theme,
                title=dict(text="Feature Importance — Random Forest", font=dict(family="Orbitron", size=13, color=C_TEXT)),
                xaxis_title="Importance Score",
                height=320,
                margin=dict(l=10, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── HEAD-TO-HEAD ─────────────────────────────────────────────────────────
    st.markdown("---")
    render_section("Comparison", "Head-to-Head")
    render_h2h(simulation_results, grid)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    csv = simulation_results.to_csv(index=False)
    c1, c2 = st.columns([5, 1])
    with c2:
        st.download_button(
            "Export CSV",
            data=csv,
            file_name=f"{selected_race.replace(' ', '_')}_predictions.csv",
            mime="text/csv"
        )

    st.markdown(f"""
    <div class="footer">
        <div class="footer-brand">F1 INTELLIGENCE — 2026</div>
        <div class="footer-sub">
            Final Year Computer Science Thesis &nbsp;·&nbsp; FastF1 + scikit-learn + Plotly + Streamlit<br>
            Predictions are probabilistic estimates for educational and research purposes only.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()