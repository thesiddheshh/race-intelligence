"""Weather data integration for race conditions."""
import requests
import pandas as pd
import logging
from typing import Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WeatherDataLoader:
    """Handles weather data fetching from OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"
    
    def __init__(self, api_key: str, units: str = "metric"):
        self.api_key = api_key
        self.units = units
        if not api_key:
            logger.warning("Weather API key not provided - using fallback values")
    
    def get_forecast(self, lat: float, lon: float, target_time: datetime) -> dict:
        """Get weather forecast for specific coordinates and time."""
        if not self.api_key:
            return self._get_fallback_weather()
        
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': self.units,
                'cnt': 40  # 5 days * 8 intervals/day
            }
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Find closest forecast to target time
            target_ts = target_time.timestamp()
            forecasts = data['list']
            closest = min(forecasts, key=lambda x: abs(x['dt'] - target_ts))
            
            return {
                'rain_probability': closest.get('pop', 0),  # 0-1
                'temperature': closest['main']['temp'],
                'humidity': closest['main']['humidity'],
                'wind_speed': closest['wind']['speed'],
                'conditions': closest['weather'][0]['main'],
                'pressure': closest['main']['pressure']
            }
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self._get_fallback_weather()
    
    def _get_fallback_weather(self) -> dict:
        """Return sensible fallback values when API fails."""
        return {
            'rain_probability': 0.15,
            'temperature': 22.0,
            'humidity': 55,
            'wind_speed': 3.2,
            'conditions': 'Clear',
            'pressure': 1013
        }
    
    def get_track_coordinates(self, event_name: str) -> Tuple[float, float]:
        """Get latitude/longitude for F1 track (simplified mapping)."""
        # In production, this would use a proper geocoding service or static mapping
        track_coords = {
            'Bahrain Grand Prix': (26.0325, 50.5106),
            'Saudi Arabian Grand Prix': (21.6319, 39.1044),
            'Australian Grand Prix': (-37.8497, 144.9680),
            'Japanese Grand Prix': (34.8431, 136.5407),
            'Chinese Grand Prix': (31.3389, 121.2197),
            'Miami Grand Prix': (25.9581, -80.2389),
            'Emilia Romagna Grand Prix': (44.3439, 11.7167),
            'Monaco Grand Prix': (43.7347, 7.4206),
            'Canadian Grand Prix': (45.5000, -73.5228),
            'Spanish Grand Prix': (41.5700, 2.2611),
            'Austrian Grand Prix': (47.2197, 14.7647),
            'British Grand Prix': (52.0786, -1.0169),
            'Hungarian Grand Prix': (47.5789, 19.2486),
            'Belgian Grand Prix': (50.4372, 5.9714),
            'Dutch Grand Prix': (52.3888, 4.5409),
            'Italian Grand Prix': (45.6156, 9.2811),
            'Azerbaijan Grand Prix': (40.3725, 49.8533),
            'Singapore Grand Prix': (1.2914, 103.8644),
            'United States Grand Prix': (30.1328, -97.6411),
            'Mexico City Grand Prix': (19.4042, -99.0907),
            'São Paulo Grand Prix': (-23.7036, -46.6997),
            'Las Vegas Grand Prix': (36.1147, -115.1728),
            'Qatar Grand Prix': (25.4889, 51.4542),
            'Abu Dhabi Grand Prix': (24.4672, 54.6031),
        }
        return track_coords.get(event_name, (45.5017, -73.5663))  # Default: Montreal