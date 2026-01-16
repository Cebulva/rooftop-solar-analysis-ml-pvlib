"""
Convert OpenStreetMap building geometry to roof segmentation and generate heatmap.
Integrates building_analysis with roof_solar_heatmap.
"""

import numpy as np
import cv2
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd
from building_analysis import get_building_at_location
from roof_solar_heatmap import RoofSolarHeatmap
import math

class OSMRoofHeatmap:
    """
    Generate solar heatmaps directly from OpenStreetMap building data.
    No manual image upload required!
    """
    
    def __init__(self, lat, lon, image_size=512, meters_per_pixel=0.5):
        """
        Initialize with location and desired resolution.
        
        Args:
            lat: Latitude
            lon: Longitude
            image_size: Size of generated image (pixels)
            meters_per_pixel: Ground sample distance (resolution)
        """
        self.lat = lat
        self.lon = lon
        self.image_size = image_size
        self.meters_per_pixel = meters_per_pixel
        
        # Calculate coverage area in meters
        self.coverage_meters = image_size * meters_per_pixel
        
    def lat_lon_to_meters(self, lat, lon):
        """
        Convert lat/lon to approximate meters from reference point.
        Uses simple equirectangular projection (good for small areas).
        """
        # Meters per degree at this latitude
        meters_per_lat = 111320
        meters_per_lon = 111320 * math.cos(math.radians(self.lat))
        
        # Calculate offset from center point
        y = (lat - self.lat) * meters_per_lat
        x = (lon - self.lon) * meters_per_lon
        
        return x, y
    
    def meters_to_pixels(self, x_meters, y_meters):
        """
        Convert meters to pixel coordinates.
        Origin at center of image.
        """
        # Convert to pixels
        x_pixels = (x_meters / self.meters_per_pixel) + (self.image_size / 2)
        y_pixels = (self.image_size / 2) - (y_meters / self.meters_per_pixel)  # Y is flipped
        
        return int(x_pixels), int(y_pixels)
    
    def geometry_to_pixel_coords(self, geometry):
        """
        Convert shapely geometry (in lat/lon) to pixel coordinates.
        """
        if geometry.is_empty:
            return None
        
        coords = []
        
        if isinstance(geometry, Polygon):
            # Get exterior coordinates
            for lon, lat in geometry.exterior.coords:
                x_m, y_m = self.lat_lon_to_meters(lat, lon)
                x_px, y_px = self.meters_to_pixels(x_m, y_m)
                coords.append([x_px, y_px])
        
        elif isinstance(geometry, MultiPolygon):
            # Handle multiple polygons
            all_coords = []
            for poly in geometry.geoms:
                poly_coords = []
                for lon, lat in poly.exterior.coords:
                    x_m, y_m = self.lat_lon_to_meters(lat, lon)
                    x_px, y_px = self.meters_to_pixels(x_m, y_m)
                    poly_coords.append([x_px, y_px])
                all_coords.append(np.array(poly_coords, dtype=np.int32))
            return all_coords
        
        return [np.array(coords, dtype=np.int32)]
    
    def create_roof_segmentation_from_osm(self, search_radius=100):
        """
        Create a binary roof segmentation image from OpenStreetMap data.
        
        Args:
            search_radius: Radius in meters to search for buildings
        
        Returns:
            numpy array: Binary image (255 = roof, 0 = background)
            list: Building information for each detected building
        """
        # Create black background
        roof_image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Get building at clicked location
        building = get_building_at_location(self.lat, self.lon, radius=search_radius)
        
        buildings_info = []
        
        if building is None:
            return roof_image, buildings_info
        
        # Convert building geometry to pixel coordinates
        pixel_coords = self.geometry_to_pixel_coords(building['geometry'])
        
        if pixel_coords is None:
            return roof_image, buildings_info
        
        # Draw building on image (white)
        if isinstance(pixel_coords[0], list):
            # Multiple polygons
            for coords in pixel_coords:
                cv2.fillPoly(roof_image, [coords], 255)
        else:
            # Single polygon
            cv2.fillPoly(roof_image, pixel_coords, 255)
        
        # Store building info
        buildings_info.append({
            'type': building.get('building_type', 'unknown'),
            'height': building.get('height', 0),
            'area_m2': building.get('area_m2', 0),
            'geometry': building['geometry']
        })
        
        return roof_image, buildings_info
    
    def create_roof_segmentation_multi_building(self, search_radius=100):
        """
        Create roof segmentation with ALL buildings in the area.
        
        Args:
            search_radius: Radius in meters to search for buildings
        
        Returns:
            numpy array: Binary image with all buildings
            list: Information for all buildings
        """
        import osmnx as ox
        
        # Create black background
        roof_image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        try:
            # Get all buildings in the area
            buildings = ox.features_from_point(
                (self.lat, self.lon),
                tags={'building': True},
                dist=search_radius
            )
            
            if buildings.empty:
                return roof_image, []
            
            buildings_info = []
            
            # Draw each building
            for idx, building in buildings.iterrows():
                geometry = building.geometry
                
                # Convert to pixel coordinates
                pixel_coords = self.geometry_to_pixel_coords(geometry)
                
                if pixel_coords is None:
                    continue
                
                # Draw building
                if isinstance(pixel_coords[0], list):
                    for coords in pixel_coords:
                        cv2.fillPoly(roof_image, [coords], 255)
                else:
                    cv2.fillPoly(roof_image, pixel_coords, 255)
                
                # Store info
                buildings_info.append({
                    'type': building.get('building', 'unknown'),
                    'height': building.get('height', None),
                    'name': building.get('name', 'Unknown'),
                    'geometry': geometry
                })
            
            return roof_image, buildings_info
            
        except Exception as e:
            print(f"Error fetching buildings: {e}")
            return roof_image, []
    
    def generate_heatmap_from_osm(self, heatmap_type='yearly', multi_building=False, 
                                   search_radius=100, samples_per_month=2):
        """
        Complete workflow: OSM data → roof segmentation → solar heatmap.
        
        Args:
            heatmap_type: 'daily' or 'yearly'
            multi_building: If True, includes all buildings in area
            search_radius: Search radius for buildings in meters
            samples_per_month: For yearly heatmaps
        
        Returns:
            dict: Contains roof_image, heatmap, buildings_info, and generator
        """
        # Step 1: Create roof segmentation from OSM
        if multi_building:
            roof_image, buildings_info = self.create_roof_segmentation_multi_building(search_radius)
        else:
            roof_image, buildings_info = self.create_roof_segmentation_from_osm(search_radius)
        
        # Check if any buildings were found
        if np.sum(roof_image) == 0:
            return {
                'success': False,
                'error': 'No buildings found in this area',
                'roof_image': roof_image,
                'buildings_info': buildings_info
            }
        
        # Step 2: Save temporary image
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            cv2.imwrite(tmp.name, roof_image)
            temp_path = tmp.name
        
        try:
            # Step 3: Generate heatmap
            generator = RoofSolarHeatmap(self.lat, self.lon, temp_path)
            
            if heatmap_type == 'daily':
                heatmap = generator.create_daily_heatmap()
            else:
                heatmap = generator.create_yearly_heatmap(samples_per_month=samples_per_month)
            
            # Step 4: Analyze zones
            zones = generator.analyze_roof_zones(heatmap)
            
            # Step 5: Find optimal panel locations
            optimal_locations = generator.find_optimal_panel_locations(heatmap, panel_count=10)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return {
                'success': True,
                'roof_image': roof_image,
                'heatmap': heatmap,
                'buildings_info': buildings_info,
                'generator': generator,
                'zones': zones,
                'optimal_locations': optimal_locations,
                'num_buildings': len(buildings_info)
            }
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return {
                'success': False,
                'error': str(e),
                'roof_image': roof_image,
                'buildings_info': buildings_info
            }


def quick_osm_heatmap(lat, lon, heatmap_type='yearly', multi_building=False):
    """
    Quick function to generate heatmap from OSM data.
    
    Args:
        lat: Latitude
        lon: Longitude
        heatmap_type: 'daily' or 'yearly'
        multi_building: Include all buildings in area
    
    Returns:
        dict: Results including heatmap and building info
    """
    osm_heatmap = OSMRoofHeatmap(lat, lon, image_size=512, meters_per_pixel=0.5)
    results = osm_heatmap.generate_heatmap_from_osm(
        heatmap_type=heatmap_type,
        multi_building=multi_building,
        search_radius=100
    )
    return results


# Example usage
if __name__ == "__main__":
    # Test with a location
    lat, lon = 37.7749, -122.4194  # San Francisco
    
    print("Generating heatmap from OpenStreetMap data...")
    results = quick_osm_heatmap(lat, lon, heatmap_type='yearly', multi_building=False)
    
    if results['success']:
        print(f"✅ Success!")
        print(f"Found {results['num_buildings']} building(s)")
        print(f"Heatmap shape: {results['heatmap'].shape}")
        
        # Visualize
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(results['roof_image'], cmap='gray')
        axes[0].set_title('Roof Segmentation (from OSM)')
        axes[0].axis('off')
        
        axes[1].imshow(results['heatmap'], cmap='hot')
        axes[1].set_title('Solar Exposure Heatmap')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ Failed: {results['error']}")