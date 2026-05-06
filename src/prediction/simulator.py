"""Monte Carlo race simulation engine for finishing order prediction."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RaceSimulator:
    """Simulates F1 race outcomes using predicted lap times and stochastic elements."""
    
    def __init__(self, 
                 n_simulations: int = 1000,
                 pit_stop_variance: float = 2.5,
                 reliability_factors: Dict[str, float] = None):
        self.n_simulations = n_simulations
        self.pit_stop_variance = pit_stop_variance
        # Default DNF probabilities by component (per race)
        self.reliability_factors = reliability_factors or {
            'engine': 0.02, 'gearbox': 0.015, 'hydraulics': 0.01,
            'collision': 0.03, 'driver_error': 0.02
        }
    
    def simulate_race(self, 
                     driver_data: pd.DataFrame,
                     track_info: Dict) -> pd.DataFrame:
        """
        Run Monte Carlo simulation to predict finishing positions.
        
        Args:
            driver_data: DataFrame with predicted lap times, consistency, etc.
            track_info: Dict with track characteristics (overtaking_difficulty, etc.)
        
        Returns:
            DataFrame with finishing position distributions per driver
        """
        results = []
        
        for sim in range(self.n_simulations):
            sim_result = self._run_single_simulation(driver_data, track_info, sim)
            results.append(sim_result)
        
        # Aggregate results
        all_positions = pd.DataFrame(results)
        summary = all_positions.groupby('driver').agg({
            'position': ['mean', 'std', lambda x: x.mode()[0]],
            'dnf': 'mean',
            'fastest_lap': 'sum'
        }).round(3)
        summary.columns = ['avg_position', 'position_std', 'mode_position', 'dnf_probability', 'fastest_lap_count']
        
        # Calculate podium probabilities
        for pos in [1, 2, 3]:
            summary[f'p_podium_{pos}'] = (all_positions.groupby('driver')['position'] <= pos).mean()
        summary['podium_probability'] = summary[[f'p_podium_{pos}' for pos in [1,2,3]]].max(axis=1)
        
        return summary.sort_values('avg_position')
    
    def _run_single_simulation(self, 
                              driver_data: pd.DataFrame,
                              track_info: Dict,
                              seed: int) -> pd.DataFrame:
        """Run one race simulation instance."""
        np.random.seed(seed)
        sim_results = []
        
        # Base race distance (laps)
        race_laps = track_info.get('race_laps', 58)
        overtaking_difficulty = track_info.get('overtaking_difficulty', 0.5)  # 0=easy, 1=hard
        
        for _, driver in driver_data.iterrows():
            # Check for DNF
            dnf_prob = sum(self.reliability_factors.values())
            if np.random.random() < dnf_prob:
                sim_results.append({
                    'driver': driver['driver_code'],
                    'position': np.nan,
                    'dnf': True,
                    'fastest_lap': False,
                    'total_time': np.inf
                })
                continue
            
            # Generate lap times with consistency and strategy
            base_lap_time = driver['predicted_lap_time_s']
            consistency = driver.get('lap_consistency', 1.5)  # std dev in seconds
            
            # Tire degradation model (simplified)
            stint_length = np.random.choice([15, 20, 25, 30])  # Random strategy
            degradation_rate = 0.02 if driver['compound'] == 'SOFT' else 0.01
            
            lap_times = []
            for lap in range(race_laps):
                # Add degradation
                degraded_time = base_lap_time * (1 + degradation_rate * (lap % stint_length) / stint_length)
                # Add stochastic variation
                noise = np.random.normal(0, consistency)
                lap_times.append(max(degraded_time + noise, base_lap_time * 0.95))  # Floor at 95% of base
            
            # Pit stop time loss (stochastic)
            n_pits = race_laps // stint_length
            pit_loss = np.random.normal(22, self.pit_stop_variance, n_pits).sum()
            
            total_time = sum(lap_times) + pit_loss
            
            # Overtaking adjustment (simplified)
            # Drivers with better qualifying positions have advantage
            grid_position = driver.get('qualifying_position', 10)
            position_adjustment = np.random.normal(0, overtaking_difficulty * 2)
            final_position = grid_position + position_adjustment
            
            sim_results.append({
                'driver': driver['driver_code'],
                'position': final_position,
                'dnf': False,
                'fastest_lap': min(lap_times) < base_lap_time * 0.98,
                'total_time': total_time
            })
        
        # Convert positions to actual finishing order
        df = pd.DataFrame(sim_results)
        valid = df.dropna(subset=['position'])
        if len(valid) > 0:
            valid['finishing_position'] = valid['position'].rank(method='min').astype(int)
            df.update(valid[['finishing_position']])
        
        return df[['driver', 'finishing_position' if 'finishing_position' in df.columns else 'position', 'dnf', 'fastest_lap', 'total_time']]