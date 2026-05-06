"""Calculate podium/DNF probabilities from simulation results."""
import pandas as pd
import numpy as np
from typing import Dict

def calculate_podium_probabilities(simulation_results: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate detailed probability metrics from race simulations.
    
    Returns dict like:
    {
        'VER': {
            'win_probability': 0.42,
            'podium_probability': 0.89,
            'top5_probability': 0.97,
            'dnf_probability': 0.03,
            'expected_position': 2.1,
            'position_std': 1.8
        },
        ...
    }
    """
    probabilities = {}
    
    for driver in simulation_results['driver'].unique():
        driver_sims = simulation_results[simulation_results['driver'] == driver]
        
        positions = driver_sims['finishing_position'].dropna()
        dnf_count = driver_sims['dnf'].sum()
        total_sims = len(driver_sims)
        
        probabilities[driver] = {
            'win_probability': (positions == 1).mean(),
            'podium_probability': (positions <= 3).mean(),
            'top5_probability': (positions <= 5).mean(),
            'points_probability': (positions <= 10).mean(),  # Assuming top 10 score points
            'dnf_probability': dnf_count / total_sims,
            'expected_position': positions.mean(),
            'position_std': positions.std(),
            'most_likely_position': positions.mode().iloc[0] if not positions.mode().empty else None
        }
    
    return probabilities