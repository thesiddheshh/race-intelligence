"""Data cleaning and validation pipeline."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class F1DataPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for F1 data cleaning."""
    
    def __init__(self, 
                 min_laps: int = 10,
                 time_threshold_s: float = 120.0,
                 impute_strategy: str = 'median'):
        self.min_laps = min_laps
        self.time_threshold_s = time_threshold_s
        self.impute_strategy = impute_strategy
        self.valid_drivers_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Identify valid drivers based on minimum lap count."""
        lap_counts = X.groupby('Driver').size()
        self.valid_drivers_ = set(lap_counts[lap_counts >= self.min_laps].index)
        logger.info(f"Valid drivers (≥{self.min_laps} laps): {len(self.valid_drivers_)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations."""
        df = X.copy()
        
        # Filter to valid drivers only
        df = df[df['Driver'].isin(self.valid_drivers_)].copy()
        
        # Remove outlier lap times (>2x median for driver)
        for driver in df['Driver'].unique():
            mask = df['Driver'] == driver
            median_time = df.loc[mask, 'LapTime_s'].median()
            df.loc[mask, 'IsOutlier'] = df.loc[mask, 'LapTime_s'] > (median_time * 2)
        df = df[~df['IsOutlier']].drop(columns=['IsOutlier'])
        
        # Impute missing sector times with driver median
        sector_cols = ['Sector1Time_s', 'Sector2Time_s', 'Sector3Time_s']
        for col in sector_cols:
            if df[col].isna().any():
                df[col] = df.groupby('Driver')[col].transform(
                    lambda x: x.fillna(x.median() if x.notna().any() else self.time_threshold_s/3)
                )
        
        # Ensure compound is categorical with known values
        valid_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        df['Compound'] = df['Compound'].apply(
            lambda x: x if x in valid_compounds else 'UNKNOWN'
        )
        
        logger.info(f"Cleaned dataset: {len(df)} laps, {df['Driver'].nunique()} drivers")
        return df.reset_index(drop=True)
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that DataFrame has required columns and types."""
        required = {
            'Driver': 'object',
            'LapTime_s': 'float64',
            'Sector1Time_s': 'float64',
            'Sector2Time_s': 'float64',
            'Sector3Time_s': 'float64',
            'Compound': 'object',
            'Position': 'float64'
        }
        for col, dtype in required.items():
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
            if df[col].dtype != dtype and not pd.api.types.is_numeric_dtype(df[col]) and dtype == 'float64':
                logger.warning(f"Column {col} has dtype {df[col].dtype}, expected {dtype}")
        return True