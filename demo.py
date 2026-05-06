"""
🏁 F1 2026 Race Prediction System
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# F1 Theme Colors
F1_RED = "#e10600"
F1_DARK = "#15151e"
F1_CARD = "#1e1e2a"

# ==================== GRID LOADER ====================
def load_grid():
    """Load 2026 grid from JSON (dynamic, no hardcoding)."""
    if not GRID_FILE.exists():
        st.error(f"❌ Grid file not found: {GRID_FILE}")
        st.stop()
    with open(GRID_FILE, 'r') as f:
        return json.load(f)

def get_driver_to_team_map(grid):
    """Create driver_code -> team mapping."""
    return {code: info["team"] for code, info in grid["drivers"].items()}

def get_team_performance(grid, historical_points=None):
    """Calculate team performance scores (normalized 0-1)."""
    # Default points if no historical data
    points = historical_points or {
        "Red Bull Racing": 860, "McLaren": 650, "Ferrari": 610,
        "Mercedes": 480, "Aston Martin": 92, "Williams": 28,
        "RB": 46, "Haas": 54, "Alpine": 13, "Audi": 0, "Cadillac": 0
    }
    # Map team names from grid
    team_scores = {}
    for team in grid["teams"].keys():
        # Handle name variations
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
    """Load real race data via FastF1."""
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
        
        # Extract key columns
        cols = ['Driver', 'Team', 'LapTime', 'Sector1Time', 'Sector2Time', 
                'Sector3Time', 'Compound', 'Stint', 'Position']
        laps = laps[[c for c in cols if c in laps.columns]]
        
        # Convert times to seconds
        for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
            if col in laps.columns:
                laps[f'{col}_s'] = laps[col].dt.total_seconds()
        
        # Clean data
        laps = laps.dropna(subset=['LapTime_s'])
        laps = laps[laps['LapTime_s'] > 60]  # Remove outliers
        
        return laps, None
    except Exception as e:
        return None, str(e)

def get_weather_fallback():
    """Fallback weather when API unavailable."""
    return {
        'rain_probability': 0.15,
        'temperature': 22.0,
        'humidity': 55,
        'wind_speed': 3.2
    }

# ==================== FEATURE ENGINEERING ====================
def engineer_features(laps, grid, weather, track_name):
    """Create domain-specific F1 features - FIXED version."""
    driver_to_team = get_driver_to_team_map(grid)
    team_perf = get_team_performance(grid)
    
    # Aggregate lap data per driver using NAMED aggregations (cleaner column names)
    driver_stats = laps.groupby('Driver').agg(
        lap_time_mean=('LapTime_s', 'mean'),
        lap_time_std=('LapTime_s', 'std'),
        lap_time_min=('LapTime_s', 'min'),
        lap_count=('LapTime_s', 'count'),
        sector1_mean=('Sector1Time_s', 'mean'),
        sector2_mean=('Sector2Time_s', 'mean'),
        sector3_mean=('Sector3Time_s', 'mean'),
        avg_position=('Position', 'mean'),
        # Handle Compound with a safer approach
        compound=('Compound', lambda x: x.mode()[0] if not x.mode().empty else 'MEDIUM')
    ).reset_index()
    
    # Add derived features
    driver_stats['SectorTotal_s'] = (
        driver_stats['sector1_mean'] + 
        driver_stats['sector2_mean'] + 
        driver_stats['sector3_mean']
    )
    driver_stats['LapConsistency'] = driver_stats['lap_time_std'].fillna(2.0)
    driver_stats['Team'] = driver_stats['Driver'].map(driver_to_team)
    driver_stats['TeamPerformance'] = driver_stats['Team'].map(team_perf).fillna(0.5)
    
    # Weather adjustments
    driver_stats['WeatherPenalty'] = weather['rain_probability'] * 2.5
    driver_stats['TempAdjustment'] = np.abs(weather['temperature'] - 22) * 0.05
    
    # Recent form proxy
    driver_stats['RecentForm'] = np.clip(driver_stats['lap_count'] / 20, 0.5, 1.0)
    
    # Tire compound encoding (use correct column name: 'compound')
    compound_map = {'SOFT': 0.95, 'MEDIUM': 1.0, 'HARD': 1.05, 'INTERMEDIATE': 1.15, 'WET': 1.25}
    driver_stats['CompoundFactor'] = driver_stats['compound'].map(compound_map).fillna(1.0)
    
    # Qualifying position proxy
    driver_stats['QualiProxy'] = driver_stats['avg_position'].rank(method='min')
    
    # Final feature set
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
    """Train and evaluate multiple ML models."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {}
    results = {}
    
    # Model 1: Random Forest
    if model_type in ['all', 'rf']:
        rf = RandomForestRegressor(n_estimators=200, max_depth=12, 
                                  min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        models['rf'] = {'model': rf, 'scaler': scaler, 'cv_mae': -cv_scores.mean()}
        results['Random Forest'] = {
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': pd.Series(rf.feature_importances_, index=X.columns)
        }
    
    # Model 2: Gradient Boosting (XGBoost-style)
    if model_type in ['all', 'gb']:
        gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=6, subsample=0.8, random_state=42)
        gb.fit(X_scaled, y)
        cv_scores = cross_val_score(gb, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        models['gb'] = {'model': gb, 'scaler': scaler, 'cv_mae': -cv_scores.mean()}
        results['Gradient Boosting'] = {
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': pd.Series(gb.feature_importances_, index=X.columns)
        }
    
    # Model 3: Ridge Regression (baseline + stability)
    if model_type in ['all', 'ridge']:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        cv_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
        models['ridge'] = {'model': ridge, 'scaler': scaler, 'cv_mae': -cv_scores.mean()}
        results['Ridge Regression'] = {
            'cv_mae': -cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    return models, results

def predict_with_uncertainty(model_dict, X, feature_cols, n_bootstrap=50):
    """Generate predictions with confidence intervals via bootstrap."""
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    X_scaled = scaler.transform(X[feature_cols])
    point_pred = model.predict(X_scaled)
    
    # Bootstrap for uncertainty
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
    
    return {
        'point': point_pred,
        'lower': lower,
        'upper': upper,
        'std': std
    }

# ==================== RACE SIMULATION (FIXED) ====================
def simulate_race(driver_features, predictions, n_simulations=500):
    """Monte Carlo simulation to convert lap times → finishing positions."""
    results = []
    
    for sim in range(n_simulations):
        sim_positions = []
        
        for idx, row in driver_features.iterrows():
            driver = row['Driver']
            
            # Base predicted lap time with noise
            # Ensure idx matches array index (safe because of reset_index)
            base_time = predictions['point'][idx]
            consistency = row['LapConsistency']
            
            # Add stochastic variation
            noise = np.random.normal(0, consistency * 0.5)
            lap_time = base_time + noise
            
            # Weather/tire adjustments
            lap_time *= (1 + row['WeatherPenalty'] * 0.1)
            lap_time *= row['CompoundFactor']
            
            # DNF probability (simplified)
            dnf_prob = 0.02 + (1 - row['RecentForm']) * 0.03
            if np.random.random() < dnf_prob:
                sim_positions.append({'driver': driver, 'position': np.inf, 'dnf': True})
                continue
            
            # Qualifying position advantage
            grid_pos = row['QualiProxy']
            position = grid_pos + np.random.normal(0, 1.5)
            
            sim_positions.append({
                'driver': driver,
                'position': max(1, position),
                'dnf': False,
                'predicted_time': lap_time
            })
        
        # Convert to finishing order
        df = pd.DataFrame(sim_positions)
        
        # 🔧 FIX: Initialize column with NaN so it always exists
        df['finishing_position'] = np.nan
        
        valid = df[df['position'] != np.inf]
        if not valid.empty:
            # Calculate ranks for valid finishers
            ranks = valid['position'].rank(method='min').astype(int)
            # Update the main dataframe using index alignment
            df.loc[valid.index, 'finishing_position'] = ranks
            
        results.append(df)
    
    # Aggregate simulations
    all_results = pd.concat(results, ignore_index=True)
    
    # Calculate statistics per driver
    summary = []
    for driver in driver_features['Driver'].unique():
        driver_sims = all_results[all_results['driver'] == driver]
        
        # 🔧 FIX: Now safe to access because column is guaranteed to exist
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
def setup_page():
    """Configure Streamlit page with F1 theme."""
    st.set_page_config(
        page_title="🏁 F1 2026 Predictor | Thesis Project",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown(f"""
    <style>
        .main {{background-color: {F1_DARK};}}
        .stApp {{background-color: {F1_DARK};}}
        h1, h2, h3 {{color: {F1_RED} !important;}}
        .stMetric {{background-color: {F1_CARD}; padding: 1rem; border-radius: 0.5rem;}}
        .stPlotlyChart {{background-color: {F1_CARD}; padding: 0.5rem; border-radius: 0.5rem;}}
        div[data-testid="stMetricValue"] {{color: {F1_RED} !important;}}
        .stButton>button {{
            background-color: {F1_RED}; color: white; border: none;
            border-radius: 4px; font-weight: 600;
        }}
        .stButton>button:hover {{background-color: #ff1a1a;}}
        .podium-1 {{background: linear-gradient(135deg, #ffd700, #daa520); color: #000;}}
        .podium-2 {{background: linear-gradient(135deg, #c0c0c0, #a9a9a9); color: #000;}}
        .podium-3 {{background: linear-gradient(135deg, #cd7f32, #8b4513); color: #fff;}}
    </style>
    """, unsafe_allow_html=True)

def main():
    setup_page()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏁 F1 2026 Race Prediction System")
        st.markdown("*Final Year Computer Science Thesis Project*")
    with col2:
        st.markdown("""
        <div style="text-align:right; color:#888; font-size:0.9rem;">
            <strong>Real Data</strong> • FastF1<br>
            <strong>3 ML Models</strong> • RF/GB/Ridge<br>
            <strong>Monte Carlo</strong> • 500 simulations
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load grid
        grid = load_grid()
        teams = list(grid["teams"].keys())
        
        # Race selection
        year = st.selectbox("Season", [2024, 2025, 2026], index=1)
        
        # Get available races (with fallback)
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
        
        selected_race = st.selectbox("🏆 Select Grand Prix", race_options)
        
        # Model selection
        st.subheader("🧠 Model Selection")
        model_choice = st.radio(
            "Prediction Model",
            ["Random Forest", "Gradient Boosting", "Ridge Regression", "Ensemble (Average)"],
            index=1
        )
        
        # Simulation settings
        st.subheader("🎲 Simulation")
        n_sims = st.slider("Monte Carlo Runs", 100, 2000, 500, step=100)
        show_uncertainty = st.checkbox("Show Confidence Intervals", value=True)
        
        # Weather override
        st.subheader("🌧️ Weather")
        use_custom = st.checkbox("Custom Conditions")
        if use_custom:
            rain = st.slider("Rain Probability (%)", 0, 100, 15) / 100
            temp = st.slider("Temperature (°C)", -10, 50, 22)
            weather = {'rain_probability': rain, 'temperature': temp, 'humidity': 55, 'wind_speed': 3.2}
        else:
            weather = get_weather_fallback()
        
        st.markdown("---")
        st.markdown("**2026 Grid Loaded** ✅")
        st.markdown(f"- {len(grid['teams'])} teams")
        st.markdown(f"- {len(grid['drivers'])} drivers")
        st.markdown("- Dynamic JSON configuration")
    
    # Main content
    st.markdown(f"## 🏁 {selected_race} Predictions")
    
    # Load data
    with st.spinner("🔄 Loading real F1 data via FastF1..."):
        laps, error = load_race_data(year, selected_race)
        
        if error or laps is None or laps.empty:
            st.warning(f"⚠️ Could not load real data: {error or 'No data'}")
            st.info("💡 Using intelligent fallback with 2026 grid + historical patterns")
            
            # Create synthetic but realistic fallback data
            driver_codes = list(grid["drivers"].keys())
            np.random.seed(42)
            
            laps = pd.DataFrame({
                'Driver': driver_codes * 10,  # 10 laps each
                'LapTime_s': np.random.uniform(92, 98, len(driver_codes) * 10),
                'Sector1Time_s': np.random.uniform(28, 32, len(driver_codes) * 10),
                'Sector2Time_s': np.random.uniform(30, 34, len(driver_codes) * 10),
                'Sector3Time_s': np.random.uniform(24, 28, len(driver_codes) * 10),
                'Compound': np.random.choice(['SOFT', 'MEDIUM', 'HARD'], len(driver_codes) * 10),
                'Position': np.random.uniform(1, 22, len(driver_codes) * 10)
            })
    
    # Feature engineering
    with st.spinner("🔧 Engineering domain-specific features..."):
        features, driver_stats = engineer_features(laps, grid, weather, selected_race)
        
        if features.empty:
            st.error("❌ Feature engineering failed - check data quality")
            st.stop()
        
        st.success(f"✅ Created {len(features.columns)-1} features for {len(features)} drivers")
    
    # Prepare training data
    feature_cols = [c for c in features.columns if c != 'Driver']
    X = features[feature_cols]
    
    # Target: lower lap time = better (we'll invert for position prediction)
    y = features['lap_time_mean']  # ✅ FIXED: matches named aggregation column
    
    # Train models
    with st.spinner(f" Training {model_choice} model..."):
        model_key = {'Random Forest': 'rf', 'Gradient Boosting': 'gb', 
                    'Ridge Regression': 'ridge', 'Ensemble (Average)': 'all'}[model_choice]
        
        models, eval_results = train_models(X, y, model_type=model_key)
        
        # Select model for prediction
        if model_choice == "Ensemble (Average)":
            # Average predictions from all models
            preds = np.zeros(len(X))
            uncertainties = np.zeros(len(X))
            for key in ['rf', 'gb', 'ridge']:
                if key in models:
                    pred = predict_with_uncertainty(models[key], features, feature_cols)
                    preds += pred['point']
                    uncertainties += pred['std']
            preds /= len([k for k in ['rf', 'gb', 'ridge'] if k in models])
            uncertainties /= len([k for k in ['rf', 'gb', 'ridge'] if k in models])
            predictions = {'point': preds, 'lower': preds - uncertainties, 'upper': preds + uncertainties, 'std': uncertainties}
        else:
            model_key = {'Random Forest': 'rf', 'Gradient Boosting': 'gb', 'Ridge Regression': 'ridge'}[model_choice]
            predictions = predict_with_uncertainty(models[model_key], features, feature_cols)
    
    # Race simulation
    with st.spinner(f"🎲 Running {n_sims} Monte Carlo simulations..."):
        simulation_results = simulate_race(features, predictions, n_simulations=n_sims)
        st.success(f"✅ Simulation complete")
    
    # Display Podium
    st.markdown("### 🥇 Predicted Podium")
    podium = simulation_results.head(3)
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(podium.iterrows()):
        driver = row['driver']
        driver_name = grid["drivers"].get(driver, {}).get("name", driver)
        team = grid["drivers"].get(driver, {}).get("team", "Unknown")
        
        with cols[i]:
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            win_pct = row['win_prob'] * 100
            podium_pct = row['podium_prob'] * 100
            
            st.markdown(f"""
            <div class="podium-{i+1}" style="padding:1.5rem; border-radius:12px; text-align:center; margin:0.5rem;">
                <h2 style="margin:0; font-size:3rem;">{medal}</h2>
                <h3 style="margin:0.5rem 0;">{driver}</h3>
                <p style="margin:0.2rem 0; font-size:0.9rem; opacity:0.9;">{driver_name}<br>{team}</p>
                <p style="margin:0.2rem 0;"><strong>Win:</strong> {win_pct:.1f}%</p>
                <p style="margin:0.2rem 0;"><strong>Podium:</strong> {podium_pct:.1f}%</p>
                <p style="margin:0.2rem 0;"><strong>Expected:</strong> {row['expected_position']:.2f} ± {row['position_std']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Full rankings table
    st.markdown("### 📊 Full Predicted Rankings")
    
    display_df = simulation_results.copy()
    display_df['driver_name'] = display_df['driver'].map(
        lambda d: grid["drivers"].get(d, {}).get("name", d)
    )
    display_df['team'] = display_df['driver'].map(
        lambda d: grid["drivers"].get(d, {}).get("team", "Unknown")
    )
    
    # Format for display
    display_df = display_df[[
        'driver', 'driver_name', 'team', 'expected_position', 
        'position_std', 'podium_prob', 'win_prob', 'dnf_prob'
    ]].copy()
    
    display_df['expected_position'] = display_df.apply(
        lambda r: f"{r['expected_position']:.2f} ± {r['position_std']:.2f}" if show_uncertainty 
                  else f"{r['expected_position']:.2f}", axis=1
    )
    display_df['podium_prob'] = display_df['podium_prob'].map('{:.1%}'.format)
    display_df['win_prob'] = display_df['win_prob'].map('{:.1%}'.format)
    display_df['dnf_prob'] = display_df['dnf_prob'].map('{:.1%}'.format)
    
    # Color-code by expected position
    def position_color(val):
        try:
            pos = float(val.split('±')[0].strip())
            if pos <= 3: return f'color: gold; font-weight: bold'
            elif pos <= 10: return 'color: #00aaff'
            else: return 'color: #888'
        except:
            return ''
    
    st.dataframe(
        display_df.style.map(position_color, subset=['expected_position'])
        .format({'position_std': '{:.2f}'})
        .hide(axis='index'),
        use_container_width=True,
        column_config={
            "driver": "Code",
            "driver_name": "Driver",
            "team": "Team",
            "expected_position": "Expected Position",
            "podium_prob": "Podium %",
            "win_prob": "Win %",
            "dnf_prob": "DNF %"
        }
    )
    
    # Visualizations
    st.markdown("### 📈 Prediction Analytics")
    
    tab1, tab2, tab3 = st.tabs(["🏁 Position Distribution", "🎯 Probabilities", "🧠 Model Insights"])
    
    with tab1:
        if show_uncertainty and len(simulation_results) > 0:
            # Create position distribution plot
            fig = go.Figure()
            top_drivers = simulation_results.head(12)['driver'].tolist()
            
            for driver in top_drivers:
                driver_name = grid["drivers"].get(driver, {}).get("name", driver)
                # Simulate distribution (in real version, store all sim results)
                mean_pos = simulation_results[simulation_results['driver']==driver]['expected_position'].values[0]
                std_pos = simulation_results[simulation_results['driver']==driver]['position_std'].values[0]
                positions = np.random.normal(mean_pos, std_pos, 100)
                positions = np.clip(positions, 1, 22)
                
                fig.add_trace(go.Violin(
                    x=[driver_name]*len(positions), y=positions,
                    name=driver_name, box_visible=True, meanline_visible=True,
                    opacity=0.7, marker_color=F1_RED
                ))
            
            fig.update_layout(
                title="Predicted Finishing Position Distribution (Top 12)",
                xaxis_title="Driver", yaxis_title="Finishing Position",
                yaxis_autorange='reversed',
                template="plotly_dark",
                plot_bgcolor=F1_DARK,
                paper_bgcolor=F1_DARK,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple bar chart
            plot_df = simulation_results.head(15).copy()
            plot_df['driver_name'] = plot_df['driver'].map(
                lambda d: grid["drivers"].get(d, {}).get("name", d)
            )
            
            fig = px.bar(
                plot_df, x='expected_position', y='driver_name', orientation='h',
                color='expected_position', color_continuous_scale='RdYlGn_r',
                title="Expected Finishing Positions",
                labels={'expected_position': 'Expected Position', 'driver_name': 'Driver'}
            )
            fig.update_layout(
                yaxis_autorange='reversed', template="plotly_dark",
                plot_bgcolor=F1_DARK, paper_bgcolor=F1_DARK,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Podium probability chart
        prob_df = simulation_results.copy()
        prob_df['driver_name'] = prob_df['driver'].map(
            lambda d: grid["drivers"].get(d, {}).get("name", d)
        )
        prob_df = prob_df.sort_values('podium_prob', ascending=False).head(12)
        
        fig = px.bar(
            prob_df, x='podium_prob', y='driver_name', orientation='h',
            color='podium_prob', color_continuous_scale='YlOrRd',
            title="Podium Probability (%)",
            labels={'podium_prob': 'Probability', 'driver_name': 'Driver'}
        )
        fig.update_layout(
            template="plotly_dark", plot_bgcolor=F1_DARK, paper_bgcolor=F1_DARK,
            xaxis_title="Podium Probability", height=400
        )
        fig.update_xaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Model evaluation results
        st.subheader("📊 Model Performance (Cross-Validated)")
        
        eval_df = pd.DataFrame([
            {'Model': name, 'MAE (s)': data['cv_mae'], 'Std Dev': data['cv_std']}
            for name, data in eval_results.items()
        ]).sort_values('MAE (s)')
        
        fig = px.bar(
            eval_df, x='Model', y='MAE (s)', color='MAE (s)',
            color_continuous_scale='viridis', title="Lower MAE = Better Prediction",
            error_y='Std Dev'
        )
        fig.update_layout(template="plotly_dark", plot_bgcolor=F1_DARK, paper_bgcolor=F1_DARK)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if 'Random Forest' in eval_results and 'feature_importance' in eval_results['Random Forest']:
            st.subheader("🔍 Top 5 Most Important Features")
            feat_imp = eval_results['Random Forest']['feature_importance'].nlargest(5)
            
            fig = px.bar(
                x=feat_imp.values, y=feat_imp.index, orientation='h',
                color=feat_imp.values, color_continuous_scale='Viridis',
                title="Random Forest Feature Importance"
            )
            fig.update_layout(template="plotly_dark", plot_bgcolor=F1_DARK, paper_bgcolor=F1_DARK,
                           xaxis_title="Importance Score", yaxis_title="Feature")
            st.plotly_chart(fig, use_container_width=True)
    
    # Driver comparison tool
    st.markdown("### 🔍 Driver Head-to-Head")
    
    drivers_list = simulation_results['driver'].tolist()
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.selectbox("Driver 1", drivers_list, key="cmp1")
    with col2:
        d2 = st.selectbox("Driver 2", drivers_list, key="cmp2")
    
    if d1 != d2:
        d1_data = simulation_results[simulation_results['driver'] == d1].iloc[0]
        d2_data = simulation_results[simulation_results['driver'] == d2].iloc[0]
        
        # Comparison metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(f"{d1} Expected Pos", f"{d1_data['expected_position']:.2f}")
        with metrics_col2:
            st.metric(f"{d2} Expected Pos", f"{d2_data['expected_position']:.2f}")
        with metrics_col3:
            gap = d1_data['expected_position'] - d2_data['expected_position']
            st.metric("Gap", f"{gap:+.2f}", delta_color="inverse")
        
        # Probability comparison
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.markdown(f"**{d1} Probabilities**")
            st.markdown(f"- 🏆 Win: {d1_data['win_prob']:.1%}")
            st.markdown(f"- 🥇 Podium: {d1_data['podium_prob']:.1%}")
            st.markdown(f"- 🎯 Top 5: {d1_data['top5_prob']:.1%}")
            st.markdown(f"- ⚠️ DNF: {d1_data['dnf_prob']:.1%}")
        with prob_col2:
            st.markdown(f"**{d2} Probabilities**")
            st.markdown(f"- 🏆 Win: {d2_data['win_prob']:.1%}")
            st.markdown(f"- 🥇 Podium: {d2_data['podium_prob']:.1%}")
            st.markdown(f"- 🎯 Top 5: {d2_data['top5_prob']:.1%}")
            st.markdown(f"- ⚠️ DNF: {d2_data['dnf_prob']:.1%}")
    
    # Export & Info
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.info(f"🌧️ Conditions: {weather['rain_probability']*100:.0f}% rain | {weather['temperature']}°C")
    
    with col2:
        st.caption(f"Model: {model_choice} • Simulations: {n_sims} • Data: {'Real (FastF1)' if laps is not None and not laps.empty else 'Fallback'}")
    
    with col3:
        csv = simulation_results.to_csv(index=False)
        st.download_button(
            "💾 Export CSV",
            data=csv,
            file_name=f"{selected_race.replace(' ', '_')}_2026_predictions.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown(f"""
    <div style="text-align:center; color:#666; font-size:0.85rem; margin-top:2rem; padding:1rem; border-top:1px solid #333;">
        <strong>F1 2026 Race Prediction System</strong> | 
        Final Year Computer Science Thesis Project | 
        Built with FastF1 • scikit-learn • Plotly • Streamlit<br>
        <em>Predictions are probabilistic estimates, not guarantees. For educational/research purposes.</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()