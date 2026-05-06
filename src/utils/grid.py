"""Dynamic loading of 2026 F1 grid configuration."""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class GridManager:
    """Manages 2026 F1 grid data from external configuration."""
    
    GRID_FILE = Path("data/grid_2026.json")
    
    @classmethod
    def load_grid(cls) -> Dict[str, Dict]:
        """Load grid configuration from JSON file."""
        if not cls.GRID_FILE.exists():
            logger.warning(f"Grid file not found at {cls.GRID_FILE}, using embedded fallback")
            return cls._get_fallback_grid()
        
        try:
            with open(cls.GRID_FILE, 'r') as f:
                grid = json.load(f)
            logger.info(f"Loaded 2026 grid: {len(grid['teams'])} teams, {len(grid['drivers'])} drivers")
            return grid
        except Exception as e:
            logger.error(f"Failed to load grid file: {e}")
            return cls._get_fallback_grid()
    
    @classmethod
    def _get_fallback_grid(cls) -> Dict[str, Dict]:
        """Fallback grid matching user specifications exactly."""
        return {
            "teams": {
                "Red Bull Racing": ["Max Verstappen", "Isack Hadjar"],
                "Ferrari": ["Charles Leclerc", "Lewis Hamilton"],
                "Mercedes": ["George Russell", "Kimi Antonelli"],
                "McLaren": ["Lando Norris", "Oscar Piastri"],
                "Aston Martin": ["Fernando Alonso", "Lance Stroll"],
                "Alpine": ["Pierre Gasly", "Franco Colapinto"],
                "Williams": ["Carlos Sainz", "Alexander Albon"],
                "Haas": ["Esteban Ocon", "Oliver Bearman"],
                "RB": ["Liam Lawson", "Yuki Tsunoda"],
                "Audi": ["Nico Hulkenberg", "Gabriel Bortoleto"],
                "Cadillac": ["Sergio Perez", "Valtteri Bottas"]
            },
            "drivers": {
                "VER": {"name": "Max Verstappen", "team": "Red Bull Racing"},
                "HAD": {"name": "Isack Hadjar", "team": "Red Bull Racing"},
                "LEC": {"name": "Charles Leclerc", "team": "Ferrari"},
                "HAM": {"name": "Lewis Hamilton", "team": "Ferrari"},
                "RUS": {"name": "George Russell", "team": "Mercedes"},
                "ANT": {"name": "Kimi Antonelli", "team": "Mercedes"},
                "NOR": {"name": "Lando Norris", "team": "McLaren"},
                "PIA": {"name": "Oscar Piastri", "team": "McLaren"},
                "ALO": {"name": "Fernando Alonso", "team": "Aston Martin"},
                "STR": {"name": "Lance Stroll", "team": "Aston Martin"},
                "GAS": {"name": "Pierre Gasly", "team": "Alpine"},
                "COL": {"name": "Franco Colapinto", "team": "Alpine"},
                "SAI": {"name": "Carlos Sainz", "team": "Williams"},
                "ALB": {"name": "Alexander Albon", "team": "Williams"},
                "OCO": {"name": "Esteban Ocon", "team": "Haas"},
                "BEA": {"name": "Oliver Bearman", "team": "Haas"},
                "LAW": {"name": "Liam Lawson", "team": "RB"},
                "TSU": {"name": "Yuki Tsunoda", "team": "RB"},
                "HUL": {"name": "Nico Hulkenberg", "team": "Audi"},
                "BOR": {"name": "Gabriel Bortoleto", "team": "Audi"},
                "PER": {"name": "Sergio Perez", "team": "Cadillac"},
                "BOT": {"name": "Valtteri Bottas", "team": "Cadillac"}
            }
        }
    
    @classmethod
    def get_driver_to_team_map(cls) -> Dict[str, str]:
        """Get mapping from driver code to team name."""
        grid = cls.load_grid()
        return {code: info["team"] for code, info in grid["drivers"].items()}
    
    @classmethod
    def get_team_drivers(cls, team: str) -> List[str]:
        """Get list of driver codes for a team."""
        grid = cls.load_grid()
        return [code for code, info in grid["drivers"].items() if info["team"] == team]
    
    @classmethod
    def validate_grid_integrity(cls) -> bool:
        """Ensure no driver is assigned to multiple teams."""
        grid = cls.load_grid()
        team_counts = {}
        for code, info in grid["drivers"].items():
            team = info["team"]
            if team not in team_counts:
                team_counts[team] = []
            if code in team_counts[team]:
                logger.error(f"Duplicate driver {code} in team {team}")
                return False
            team_counts[team].append(code)
        
        # Verify each team has exactly 2 drivers
        for team, drivers in grid["teams"].items():
            if len(drivers) != 2:
                logger.error(f"Team {team} has {len(drivers)} drivers, expected 2")
                return False
        
        logger.info("Grid integrity validated successfully")
        return True