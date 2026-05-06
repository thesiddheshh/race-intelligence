"""FastF1 data ingestion module with caching and error handling."""
import logging
import pandas as pd
import fastf1
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class FastF1DataLoader:
    """Handles loading and caching of F1 session data via FastF1."""
    
    def __init__(self, cache_dir: str = "/tmp/f1_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        logger.info(f"FastF1 cache enabled at {self.cache_dir}")
    
    def get_session(self, year: int, event_name: str, session_type: str = "R"):
        """Get a specific session, with error handling and logging."""
        try:
            schedule = fastf1.get_event_schedule(year)
            event = schedule[schedule['EventName'].str.contains(event_name, case=False, na=False)]
            if event.empty:
                raise ValueError(f"No event found matching '{event_name}' in {year}")
            
            session = fastf1.get_session(year, event['EventName'].iloc[0], session_type)
            session.load(telemetry=False, weather=False)  # Load core data first
            logger.info(f"Loaded {session_type} session for {event['EventName'].iloc[0]}")
            return session
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            raise
    
    def extract_lap_data(self, session) -> pd.DataFrame:
        """Extract cleaned lap-level data from a session."""
        laps = session.laps.copy()
        
        # Select and rename critical columns
        lap_cols = [
            'Driver', 'Team', 'LapNumber', 'LapTime', 'Sector1Time', 
            'Sector2Time', 'Sector3Time', 'Stint', 'Compound', 'PitOutTime',
            'PitInTime', 'Position', 'IsPersonalBest'
        ]
        laps = laps[[c for c in lap_cols if c in laps.columns]]
        
        # Convert time columns to seconds
        time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
                    'PitOutTime', 'PitInTime']
        for col in time_cols:
            if col in laps.columns:
                laps[f'{col}_s'] = laps[col].dt.total_seconds()
        
        # Drop laps with missing critical times
        laps = laps.dropna(subset=['LapTime_s', 'Sector1Time_s', 'Sector2Time_s', 'Sector3Time_s'])
        
        # Add derived columns
        laps['SectorTotal_s'] = laps[['Sector1Time_s', 'Sector2Time_s', 'Sector3Time_s']].sum(axis=1)
        laps['LapConsistency'] = laps.groupby('Driver')['LapTime_s'].transform('std')
        
        logger.info(f"Extracted {len(laps)} valid laps")
        return laps.reset_index(drop=True)
    
    def get_historical_driver_performance(self, driver: str, years: List[int] = [2024, 2025]) -> pd.DataFrame:
        """Get aggregated historical performance metrics for a driver."""
        all_laps = []
        for year in years:
            try:
                schedule = fastf1.get_event_schedule(year)
                for _, event in schedule.iterrows():
                    if event['EventFormat'] == 'conventional':
                        try:
                            session = self.get_session(year, event['EventName'], 'R')
                            laps = self.extract_lap_data(session)
                            laps['Year'] = year
                            laps['EventName'] = event['EventName']
                            all_laps.append(laps[laps['Driver'] == driver])
                        except:
                            continue  # Skip events that fail to load
            except:
                continue
        
        if not all_laps:
            return pd.DataFrame()
        
        df = pd.concat(all_laps, ignore_index=True)
        if df.empty:
            return df
        
        # Aggregate performance metrics
        agg = df.groupby(['Year', 'EventName']).agg({
            'LapTime_s': ['mean', 'std', 'min'],
            'SectorTotal_s': 'mean',
            'Position': 'mean',
            'Compound': lambda x: x.mode()[0] if not x.mode().empty else 'UNKNOWN'
        }).reset_index()
        agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
        
        return agg