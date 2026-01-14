"""
Shadow and sun position analysis module.
Calculates sun angles, daily exposure, and shade percentages.
"""

from pysolar import solar
from datetime import datetime, timedelta
import pytz
import math

def calculate_sun_angles(lat, lon, date=None):
    """
    Calculate sun altitude and azimuth for a specific location and time.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        date: datetime object (UTC timezone recommended). If None, uses current time.
    
    Returns:
        dict: Contains altitude (degrees above horizon), azimuth (degrees from north), and date
    """
    if date is None:
        date = datetime.now(pytz.UTC)
    elif date.tzinfo is None:
        # If no timezone provided, assume UTC
        date = pytz.UTC.localize(date)
    
    altitude = solar.get_altitude(lat, lon, date)
    azimuth = solar.get_azimuth(lat, lon, date)
    
    return {
        'altitude': altitude,  # degrees above horizon (0-90)
        'azimuth': azimuth,    # degrees from north (0-360)
        'date': date,
        'is_daytime': altitude > 0
    }

def calculate_daily_sun_exposure(lat, lon, date=None):
    """
    Calculate hourly sun positions for a full day.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        date: datetime object for the day. If None, uses current date.
    
    Returns:
        list of dict: Each entry contains hour, altitude, azimuth, and estimated radiation
    """
    if date is None:
        date = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    elif date.tzinfo is None:
        date = pytz.UTC.localize(date.replace(hour=0, minute=0, second=0, microsecond=0))
    else:
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    sun_data = []
    for hour in range(24):
        current_time = date + timedelta(hours=hour)
        altitude = solar.get_altitude(lat, lon, current_time)
        
        if altitude > 0:  # Sun is above horizon
            azimuth = solar.get_azimuth(lat, lon, current_time)
            
            # Get direct radiation (W/m²)
            try:
                radiation = solar.radiation.get_radiation_direct(current_time, altitude)
            except:
                # If radiation calculation fails, estimate based on altitude
                radiation = 1000 * math.sin(math.radians(altitude))
            
            sun_data.append({
                'hour': hour,
                'altitude': altitude,
                'azimuth': azimuth,
                'radiation': max(0, radiation)
            })
    
    return sun_data

def calculate_shade_percentage(lat, lon, obstacle_height=0, distance_to_obstacle=10, date=None):
    """
    Estimate shade percentage based on nearby obstacles.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        obstacle_height: Height of nearby building/tree in meters
        distance_to_obstacle: Distance to obstacle in meters
        date: datetime object. If None, uses current time.
    
    Returns:
        float: Shade percentage (0-100)
    """
    if date is None:
        date = datetime.now(pytz.UTC)
    elif date.tzinfo is None:
        date = pytz.UTC.localize(date)
    
    altitude = solar.get_altitude(lat, lon, date)
    
    if altitude <= 0:
        return 100.0  # Night time = 100% shade
    
    if obstacle_height <= 0 or distance_to_obstacle <= 0:
        return 0.0  # No obstacle = no shade
    
    # Calculate shadow angle (angle from horizontal to top of obstacle)
    shadow_angle = math.degrees(math.atan(obstacle_height / distance_to_obstacle))
    
    if altitude < shadow_angle:
        return 100.0  # Completely shaded
    else:
        # Partial shade calculation (simplified model)
        # As sun gets higher, shade decreases
        shade_factor = max(0, (shadow_angle / altitude) * 100)
        return min(100.0, shade_factor)

def calculate_yearly_sun_hours(lat, lon, year=None):
    """
    Calculate total daylight hours per month for a year.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        year: Year to analyze. If None, uses current year.
    
    Returns:
        dict: Monthly daylight hours
    """
    if year is None:
        year = datetime.now().year
    
    monthly_hours = {}
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    
    for month_num in range(1, 13):
        # Sample middle of each month
        sample_date = datetime(year, month_num, 15, tzinfo=pytz.UTC)
        daily_data = calculate_daily_sun_exposure(lat, lon, sample_date)
        
        # Count hours with sun above horizon
        daylight_hours = len(daily_data)
        monthly_hours[months[month_num - 1]] = daylight_hours
    
    return monthly_hours

def get_optimal_panel_angle(lat):
    """
    Calculate optimal solar panel tilt angle for a given latitude.
    
    Args:
        lat: Latitude in degrees
    
    Returns:
        dict: Recommended tilt angles for different seasons
    """
    # General rule: tilt = latitude for year-round optimization
    year_round = abs(lat)
    
    # Summer: latitude - 15 degrees
    summer = max(0, abs(lat) - 15)
    
    # Winter: latitude + 15 degrees
    winter = min(90, abs(lat) + 15)
    
    return {
        'year_round_optimal': round(year_round, 1),
        'summer_optimal': round(summer, 1),
        'winter_optimal': round(winter, 1),
        'note': 'These are general guidelines. Site-specific factors may vary.'
    }
