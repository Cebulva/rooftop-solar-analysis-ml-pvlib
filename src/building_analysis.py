"""
Building footprint and rooftop analysis module.
Uses OpenStreetMap data to extract building information.
"""

import osmnx as ox
from shapely.geometry import Point
import math

def get_building_at_location(lat, lon, radius=50):
    """
    Get building data for a specific location from OpenStreetMap.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        radius: Search radius in meters (default: 50m)
    
    Returns:
        dict: Building properties including geometry, area, height, type
        None: If no building found or error occurred
    """
    try:
        # Create a point for the location
        point = Point(lon, lat)
        
        # Download building footprints from OpenStreetMap
        buildings = ox.features_from_point(
            (lat, lon), 
            tags={'building': True}, 
            dist=radius
        )
        
        if buildings.empty:
            return None
        
        # Find the closest building to the point
        buildings['distance'] = buildings.geometry.distance(point)
        closest_idx = buildings['distance'].idxmin()
        closest = buildings.loc[closest_idx]
        
        # Calculate area (rough conversion from degrees to meters²)
        # More accurate calculation using UTM projection
        area_degrees = closest.geometry.area
        # Approximate: 1 degree ≈ 111km at equator
        # This is a rough estimate; for precise area, reproject to UTM
        area_m2 = area_degrees * (111320 ** 2) * math.cos(math.radians(lat))
        
        # Extract building properties
        result = {
            'geometry': closest.geometry,
            'area_m2': area_m2,
            'building_type': closest.get('building', 'unknown'),
            'height': None,
            'levels': None,
            'name': closest.get('name', 'Unknown'),
            'osm_id': closest.name if hasattr(closest, 'name') else None
        }
        
        # Try to get height information
        if 'height' in closest.index:
            try:
                height_str = str(closest['height'])
                # Remove 'm' or 'meters' if present
                height_str = height_str.replace('m', '').replace('meters', '').strip()
                result['height'] = float(height_str)
            except:
                pass
        
        # Try to get number of levels/floors
        if 'building:levels' in closest.index:
            try:
                result['levels'] = int(closest['building:levels'])
            except:
                pass
        
        # Estimate height if not available but levels are known
        if result['height'] is None and result['levels'] is not None:
            # Assume ~3 meters per floor
            result['height'] = float(result['levels']) * 3.0
        
        # If still no height, use a default based on building type
        if result['height'] is None:
            building_type = result['building_type']
            if building_type in ['house', 'residential', 'detached', 'bungalow']:
                result['height'] = 6.0  # Typical single-story house
            elif building_type in ['apartments', 'commercial']:
                result['height'] = 15.0  # Typical 5-story building
            else:
                result['height'] = 9.0  # Default ~3 stories
        
        return result
        
    except Exception as e:
        print(f"Error fetching building data: {e}")
        return None

def estimate_roof_area(lat, lon, usable_factor=0.7):
    """
    Estimate usable roof area for solar panel installation.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        usable_factor: Percentage of roof area suitable for panels (default: 0.7 = 70%)
    
    Returns:
        dict: Contains total area, usable area, height, and building type
        None: If no building found
    """
    building = get_building_at_location(lat, lon)
    
    if building is None:
        return None
    
    # Calculate usable area (accounting for vents, chimneys, shading, edges, etc.)
    usable_area = building['area_m2'] * usable_factor
    
    return {
        'total_roof_area': round(building['area_m2'], 2),
        'usable_roof_area': round(usable_area, 2),
        'building_height': building['height'],
        'building_type': building['building_type'],
        'building_name': building['name'],
        'estimated_levels': building['levels'],
        'usable_factor': usable_factor
    }

def calculate_panel_capacity(roof_area_m2, panel_efficiency=0.20, panel_size_m2=1.7):
    """
    Calculate potential solar panel capacity for a given roof area.
    
    Args:
        roof_area_m2: Available roof area in square meters
        panel_efficiency: Solar panel efficiency (default: 0.20 = 20%)
        panel_size_m2: Size of one solar panel in m² (default: 1.7 m²)
    
    Returns:
        dict: Panel count, total capacity in kW, and estimated annual production
    """
    # Calculate number of panels that fit
    num_panels = int(roof_area_m2 / panel_size_m2)
    
    # Calculate total capacity
    # Standard panel: ~300-400W, using 350W average
    panel_wattage = 350  # Watts per panel
    total_capacity_kw = (num_panels * panel_wattage) / 1000
    
    # Estimate annual production (rough estimate: 1000-1500 kWh per kW installed)
    # Using conservative 1200 kWh/kW/year
    estimated_annual_kwh = total_capacity_kw * 1200
    
    return {
        'num_panels': num_panels,
        'total_capacity_kw': round(total_capacity_kw, 2),
        'estimated_annual_production_kwh': round(estimated_annual_kwh, 0),
        'panel_efficiency': panel_efficiency,
        'panel_size_m2': panel_size_m2,
        'watts_per_panel': panel_wattage
    }

def analyze_building_solar_potential(lat, lon):
    """
    Comprehensive building and solar potential analysis.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
    
    Returns:
        dict: Complete analysis including building info, roof area, and solar capacity
        None: If no building found
    """
    # Get roof area estimate
    roof_data = estimate_roof_area(lat, lon)
    
    if roof_data is None:
        return None
    
    # Calculate solar panel capacity
    panel_data = calculate_panel_capacity(roof_data['usable_roof_area'])
    
    # Combine all information
    result = {
        'building_info': {
            'type': roof_data['building_type'],
            'name': roof_data['building_name'],
            'height': roof_data['building_height'],
            'levels': roof_data['estimated_levels']
        },
        'roof_analysis': {
            'total_area_m2': roof_data['total_roof_area'],
            'usable_area_m2': roof_data['usable_roof_area'],
            'usable_percentage': roof_data['usable_factor'] * 100
        },
        'solar_potential': panel_data,
        'location': {
            'latitude': lat,
            'longitude': lon
        }
    }
    
    return result
