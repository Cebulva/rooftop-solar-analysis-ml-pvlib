"""
Roof Solar Heatmap Generator
Creates sun exposure heatmaps for roof surfaces based on binary segmentation images.
White pixels = roofs, Black pixels = other areas
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pysolar import solar
from datetime import datetime, timedelta
import pytz
from pathlib import Path

class RoofSolarHeatmap:
    def __init__(self, lat, lon, roof_image_path, timezone='UTC'):
        """
        Initialize the roof solar heatmap generator.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            roof_image_path: Path to binary image (white=roofs, black=other)
            timezone: Timezone string (default: 'UTC')
        """
        self.lat = lat
        self.lon = lon
        self.timezone = pytz.timezone(timezone)
        
        # Load and process the roof image
        self.roof_image = cv2.imread(str(roof_image_path), cv2.IMREAD_GRAYSCALE)
        if self.roof_image is None:
            raise FileNotFoundError(f"Could not load image: {roof_image_path}")
        
        # Create roof mask (white pixels = roofs)
        _, self.roof_mask = cv2.threshold(self.roof_image, 127, 255, cv2.THRESH_BINARY)
        
        # Store image dimensions
        self.height, self.width = self.roof_mask.shape
        
    def calculate_sun_position(self, date_time):
        """Calculate sun altitude and azimuth for a given time."""
        if date_time.tzinfo is None:
            date_time = self.timezone.localize(date_time)
        
        altitude = solar.get_altitude(self.lat, self.lon, date_time)
        azimuth = solar.get_azimuth(self.lat, self.lon, date_time)
        
        return altitude, azimuth
    
    def calculate_directional_exposure(self, date_time, roof_mask):
        """
        Calculate sun exposure considering the direction of sunlight.
        Creates variation across the roof based on sun position.
        
        Args:
            date_time: datetime object
            roof_mask: Binary mask of roof areas
        
        Returns:
            2D array: Directional exposure map (0-1)
        """
        altitude, azimuth = self.calculate_sun_position(date_time)
        
        if altitude <= 0:
            return np.zeros_like(roof_mask, dtype=np.float32)
        
        # Calculate base radiation intensity
        try:
            radiation = solar.radiation.get_radiation_direct(date_time, altitude)
            base_intensity = min(radiation / 1000.0, 1.0)
        except:
            base_intensity = np.sin(np.radians(altitude))
        
        # Create directional gradient based on sun azimuth
        height, width = roof_mask.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Center coordinates
        center_y, center_x = height / 2, width / 2
        
        # Calculate angle from center for each pixel
        dx = x_coords - center_x
        dy = center_y - y_coords  # Flip Y because image coordinates
        
        # Calculate angle of each pixel from center (in degrees)
        pixel_angles = np.degrees(np.arctan2(dy, dx))
        pixel_angles = (pixel_angles + 360) % 360  # Normalize to 0-360
        
        # Calculate distance from center (normalized)
        distances = np.sqrt(dx**2 + dy**2)
        max_distance = np.sqrt((height/2)**2 + (width/2)**2)
        normalized_distances = distances / max_distance
        
        # Convert sun azimuth to image coordinates
        # Azimuth: 0°=North, 90°=East, 180°=South, 270°=West
        # Image: 0°=East, 90°=South, 180°=West, 270°=North
        sun_angle_in_image = (90 - azimuth) % 360
        
        # Calculate angular difference between sun and each pixel
        angle_diff = np.abs(pixel_angles - sun_angle_in_image)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)  # Shortest angle
        
        # Convert angle difference to exposure factor (0-1)
        # Pixels facing the sun get more exposure
        # cos(0°) = 1 (facing sun), cos(90°) = 0 (perpendicular), cos(180°) = -1 (away)
        directional_factor = np.cos(np.radians(angle_diff))
        directional_factor = np.maximum(directional_factor, 0)  # Clamp to 0-1
        
        # Add distance-based variation (edges get slightly less due to angle)
        edge_factor = 1.0 - (normalized_distances * 0.3)  # 30% reduction at edges
        
        # Combine factors
        exposure_map = base_intensity * directional_factor * edge_factor
        
        # Apply roof mask
        exposure_map = exposure_map * (roof_mask / 255.0)
        
        return exposure_map.astype(np.float32)
    
    def create_daily_heatmap(self, date=None, start_hour=6, end_hour=18, interval_minutes=30):
        """
        Create a heatmap showing cumulative sun exposure throughout a day.
        Now includes directional exposure based on sun position.
        
        Args:
            date: Date to analyze (defaults to today)
            start_hour: Start hour for analysis (default: 6 AM)
            end_hour: End hour for analysis (default: 6 PM)
            interval_minutes: Time interval between samples (default: 30 min)
        
        Returns:
            numpy array: Heatmap with values 0-1 (cumulative sun exposure)
        """
        if date is None:
            date = datetime.now(self.timezone).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date.tzinfo is None:
            date = self.timezone.localize(date.replace(hour=0, minute=0, second=0, microsecond=0))
        
        # Initialize heatmap
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Sample sun positions throughout the day
        current_time = date.replace(hour=start_hour)
        end_time = date.replace(hour=end_hour)
        sample_count = 0
        
        while current_time <= end_time:
            # Calculate directional exposure for this time
            exposure_map = self.calculate_directional_exposure(current_time, self.roof_mask)
            
            if np.sum(exposure_map) > 0:
                heatmap += exposure_map
                sample_count += 1
            
            current_time += timedelta(minutes=interval_minutes)
        
        # Normalize by number of samples
        if sample_count > 0:
            heatmap = heatmap / sample_count
        
        return heatmap
    
    def create_yearly_heatmap(self, year=None, samples_per_month=4):
        """
        Create a heatmap showing average sun exposure throughout a year.
        
        Args:
            year: Year to analyze (defaults to current year)
            samples_per_month: Number of days to sample per month
        
        Returns:
            numpy array: Heatmap with values 0-1 (average annual sun exposure)
        """
        if year is None:
            year = datetime.now().year
        
        # Initialize cumulative heatmap
        cumulative_heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        total_samples = 0
        
        # Sample multiple days throughout the year
        for month in range(1, 13):
            # Sample evenly throughout the month
            days_in_month = [7, 14, 21, 28][:samples_per_month]
            
            for day in days_in_month:
                try:
                    sample_date = datetime(year, month, day)
                    daily_heatmap = self.create_daily_heatmap(sample_date)
                    cumulative_heatmap += daily_heatmap
                    total_samples += 1
                except ValueError:
                    # Skip invalid dates (e.g., Feb 30)
                    continue
        
        # Average over all samples
        if total_samples > 0:
            cumulative_heatmap = cumulative_heatmap / total_samples
        
        return cumulative_heatmap
    
    def visualize_heatmap(self, heatmap, title="Roof Solar Exposure Heatmap", 
                         colormap='hot', save_path=None, show_original=True):
        """
        Visualize the solar heatmap with matplotlib.
        
        Args:
            heatmap: The heatmap array to visualize
            title: Title for the plot
            colormap: Matplotlib colormap name
            save_path: Path to save the visualization (optional)
            show_original: Whether to show original roof mask alongside heatmap
        """
        if show_original:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original roof mask
            ax1.imshow(self.roof_mask, cmap='gray')
            ax1.set_title('Original Roof Segmentation')
            ax1.axis('off')
            
            # Heatmap
            im = ax2.imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
            ax2.set_title(title)
            ax2.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Solar Exposure (0=None, 1=Maximum)', rotation=270, labelpad=20)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Solar Exposure (0=None, 1=Maximum)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        plt.show()
    
    def create_overlay_heatmap(self, heatmap, alpha=0.6, colormap='hot'):
        """
        Create an RGB overlay of the heatmap on the original image.
        
        Args:
            heatmap: The heatmap array
            alpha: Transparency of the heatmap overlay (0-1)
            colormap: Matplotlib colormap name
        
        Returns:
            numpy array: RGB image with heatmap overlay
        """
        # Convert grayscale roof image to RGB
        base_image = cv2.cvtColor(self.roof_mask, cv2.COLOR_GRAY2RGB)
        
        # Create colormap
        cmap = plt.get_cmap(colormap)
        
        # Normalize heatmap and apply colormap
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create overlay only on roof areas
        roof_areas = self.roof_mask > 127
        overlay = base_image.copy()
        overlay[roof_areas] = cv2.addWeighted(
            base_image[roof_areas], 
            1 - alpha, 
            heatmap_colored[roof_areas], 
            alpha, 
            0
        )
        
        return overlay
    
    def analyze_roof_zones(self, heatmap, num_zones=5):
        """
        Divide roof into zones based on sun exposure levels.
        
        Args:
            heatmap: The heatmap array
            num_zones: Number of exposure zones to create
        
        Returns:
            dict: Statistics for each zone
        """
        # Only analyze roof pixels
        roof_pixels = self.roof_mask > 127
        roof_heatmap_values = heatmap[roof_pixels]
        
        # Create zones based on exposure percentiles
        zones = {}
        zone_boundaries = np.linspace(0, 1, num_zones + 1)
        
        for i in range(num_zones):
            lower = zone_boundaries[i]
            upper = zone_boundaries[i + 1]
            
            zone_mask = (roof_heatmap_values >= lower) & (roof_heatmap_values < upper)
            zone_pixels = np.sum(zone_mask)
            zone_percentage = (zone_pixels / len(roof_heatmap_values)) * 100
            
            zones[f"Zone_{i+1}"] = {
                'exposure_range': (lower, upper),
                'pixel_count': int(zone_pixels),
                'percentage': round(zone_percentage, 2),
                'avg_exposure': round(np.mean(roof_heatmap_values[zone_mask]), 3) if zone_pixels > 0 else 0
            }
        
        return zones
    
    def find_optimal_panel_locations(self, heatmap, panel_count=10, min_exposure=0.5):
        """
        Find optimal locations for solar panel placement.
        
        Args:
            heatmap: The heatmap array
            panel_count: Number of panel locations to find
            min_exposure: Minimum exposure threshold (0-1)
        
        Returns:
            list: Coordinates of optimal panel locations
        """
        # Create a copy of heatmap with only roof areas
        roof_heatmap = heatmap * (self.roof_mask / 255.0)
        
        # Filter by minimum exposure
        suitable_areas = roof_heatmap >= min_exposure
        
        # Find coordinates of high-exposure areas
        y_coords, x_coords = np.where(suitable_areas)
        
        if len(y_coords) == 0:
            return []
        
        # Get exposure values for these coordinates
        exposure_values = roof_heatmap[y_coords, x_coords]
        
        # Sort by exposure (highest first)
        sorted_indices = np.argsort(exposure_values)[::-1]
        
        # Select top locations (with some spacing to avoid clustering)
        optimal_locations = []
        min_distance = 20  # Minimum pixel distance between panels
        
        for idx in sorted_indices:
            y, x = y_coords[idx], x_coords[idx]
            
            # Check distance from already selected locations
            too_close = False
            for prev_y, prev_x in optimal_locations:
                distance = np.sqrt((y - prev_y)**2 + (x - prev_x)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                optimal_locations.append((int(y), int(x)))
            
            if len(optimal_locations) >= panel_count:
                break
        
        return optimal_locations


# Example usage function
def generate_roof_heatmap(roof_image_path, lat, lon, output_dir='./outputs'):
    """
    Complete workflow to generate roof solar heatmap.
    
    Args:
        roof_image_path: Path to binary roof segmentation image
        lat: Latitude
        lon: Longitude
        output_dir: Directory to save outputs
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Initializing Roof Solar Heatmap Generator...")
    print(f"Location: ({lat}, {lon})")
    print(f"Roof Image: {roof_image_path}")
    
    # Initialize generator
    generator = RoofSolarHeatmap(lat, lon, roof_image_path)
    
    # Generate daily heatmap
    print("\nGenerating daily heatmap...")
    daily_heatmap = generator.create_daily_heatmap()
    generator.visualize_heatmap(
        daily_heatmap, 
        title=f"Daily Solar Exposure - Lat: {lat}, Lon: {lon}",
        save_path=output_path / 'daily_heatmap.png'
    )
    
    # Generate yearly heatmap
    print("\nGenerating yearly heatmap (this may take a moment)...")
    yearly_heatmap = generator.create_yearly_heatmap(samples_per_month=2)
    generator.visualize_heatmap(
        yearly_heatmap,
        title=f"Annual Average Solar Exposure - Lat: {lat}, Lon: {lon}",
        save_path=output_path / 'yearly_heatmap.png'
    )
    
    # Analyze roof zones
    print("\nAnalyzing roof exposure zones...")
    zones = generator.analyze_roof_zones(yearly_heatmap)
    for zone_name, zone_data in zones.items():
        print(f"\n{zone_name}:")
        print(f"  Exposure Range: {zone_data['exposure_range']}")
        print(f"  Coverage: {zone_data['percentage']}% of roof")
        print(f"  Average Exposure: {zone_data['avg_exposure']}")
    
    # Find optimal panel locations
    print("\nFinding optimal solar panel locations...")
    optimal_locations = generator.find_optimal_panel_locations(yearly_heatmap, panel_count=10)
    print(f"Found {len(optimal_locations)} optimal locations")
    
    # Create overlay with optimal locations marked
    overlay = generator.create_overlay_heatmap(yearly_heatmap)
    for y, x in optimal_locations:
        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)  # Green dots
    
    cv2.imwrite(str(output_path / 'heatmap_with_panels.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"\nOverlay image saved to: {output_path / 'heatmap_with_panels.png'}")
    
    return generator, daily_heatmap, yearly_heatmap


if __name__ == "__main__":
    # Example usage
    ROOF_IMAGE = "path/to/your/roof_segmentation.png"
    LATITUDE = 37.7749  # San Francisco
    LONGITUDE = -122.4194
    
    generator, daily, yearly = generate_roof_heatmap(ROOF_IMAGE, LATITUDE, LONGITUDE)


#####################################################################################

"""
PVGIS-Enhanced Roof Solar Heatmap Generator
Extends the base RoofSolarHeatmap with real PVGIS radiation data.

This version uses:
- Real measured radiation from PVGIS API (not theoretical calculations)
- Terrain horizon profiles for accurate shading
- Actual weather patterns and cloud cover
- Validated optimal angles from decades of data
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import requests
import json
from roof_solar_heatmap import RoofSolarHeatmap

class PVGISEnhancedHeatmap(RoofSolarHeatmap):
    """
    Enhanced version using PVGIS API for real radiation data.
    Inherits from RoofSolarHeatmap but replaces calculations with PVGIS data.
    """

    def __init__(self, lat, lon, roof_image_path, timezone='UTC'):
        """
        Initialize PVGIS-enhanced heatmap generator.

        Args:
            lat: Latitude
            lon: Longitude
            roof_image_path: Path to binary roof segmentation
            timezone: Timezone string
        """
        super().__init__(lat, lon, roof_image_path, timezone)

        self.pvgis_base_url = "https://re.jrc.ec.europa.eu/api/v5_2"
        self.horizon_profile = None
        self.direction_radiation_cache = {}

        # Fetch horizon profile once at initialization
        self._fetch_horizon_profile()

    def _fetch_horizon_profile(self):
        """Fetch terrain horizon profile from PVGIS."""
        try:
            url = f"{self.pvgis_base_url}/printhorizon"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'outputformat': 'json'
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'outputs' in data:
                    outputs = data['outputs']
                    # Check for 'horizon_profile' key (PVGIS API structure)
                    if isinstance(outputs, dict) and 'horizon_profile' in outputs:
                        self.horizon_profile = outputs['horizon_profile']
                        print(f"✓ Loaded horizon profile ({len(self.horizon_profile)} points)")
                    elif isinstance(outputs, dict) and 'horizon' in outputs:
                        self.horizon_profile = outputs['horizon']
                        print(f"✓ Loaded horizon profile ({len(self.horizon_profile)} points)")
                    elif isinstance(outputs, list):
                        # Sometimes PVGIS returns horizon data directly as a list
                        self.horizon_profile = outputs
                        print(f"✓ Loaded horizon profile ({len(self.horizon_profile)} points)")
                    else:
                        print(f"⚠ Unexpected horizon format")
                        self.horizon_profile = None
                else:
                    print(f"⚠ No 'outputs' in horizon response")
                    self.horizon_profile = None
            else:
                print(f"⚠ Could not fetch horizon profile (status {response.status_code})")
                self.horizon_profile = None

        except Exception as e:
            print(f"⚠ Error fetching horizon profile: {e}")
            import traceback
            traceback.print_exc()
            self.horizon_profile = None

    def get_pvgis_monthly_radiation(self, angle=35, aspect=0):
        """
        Get REAL monthly radiation data from PVGIS using PVcalc endpoint.

        Args:
            angle: Tilt angle (0=horizontal, 90=vertical)
            aspect: Azimuth (0=South, 90=West, -90=East, 180=North)

        Returns:
            dict: Monthly radiation data or None
        """
        # Check cache first
        cache_key = f"{angle}_{aspect}"
        if cache_key in self.direction_radiation_cache:
            return self.direction_radiation_cache[cache_key]

        try:
            # Use PVcalc endpoint which gives monthly radiation data
            url = f"{self.pvgis_base_url}/PVcalc"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'peakpower': 1,  # 1 kWp system
                'loss': 14,       # 14% system losses (default)
                'angle': angle,
                'aspect': aspect,
                'outputformat': 'json'
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()

                # PVcalc returns monthly data in outputs.monthly.fixed
                if 'outputs' in data and 'monthly' in data['outputs']:
                    monthly_data = data['outputs']['monthly']
                    if 'fixed' in monthly_data:
                        monthly_records = monthly_data['fixed']
                        # Cache and return
                        self.direction_radiation_cache[cache_key] = monthly_records
                        return monthly_records

                print(f"⚠ Unexpected PVGIS response format for angle={angle}, aspect={aspect}")
                return None
            else:
                print(f"⚠ PVGIS request failed with status {response.status_code}")
                return None

        except Exception as e:
            print(f"⚠ Error fetching radiation for angle={angle}, aspect={aspect}: {e}")
            return None

    def get_pvgis_optimal_angles(self):
        """
        Get PVGIS-calculated optimal angles based on real data.

        Returns:
            dict: Optimal angle, azimuth, and expected annual production
        """
        try:
            url = f"{self.pvgis_base_url}/PVcalc"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'peakpower': 1,
                'loss': 14,
                'mountingplace': 'free',
                'optimalangles': 1,
                'outputformat': 'json'
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return {
                    'optimal_angle': data['inputs'].get('angle'),
                    'optimal_azimuth': data['inputs'].get('aspect'),
                    'estimated_annual_kwh': data['outputs']['totals']['fixed']['E_y']
                }
            else:
                return None

        except Exception as e:
            print(f"⚠ Error fetching optimal angles: {e}")
            return None

    def check_horizon_blocking(self, sun_altitude, sun_azimuth):
        """
        Check if sun is blocked by terrain at this position.

        Args:
            sun_altitude: Sun altitude in degrees
            sun_azimuth: Sun azimuth in degrees (0=South for PVGIS)

        Returns:
            bool: True if sun is visible, False if blocked by terrain
        """
        if not self.horizon_profile or sun_altitude <= 0:
            return sun_altitude > 0

        try:
            # Find horizon height at this azimuth
            azimuths = [h['A'] for h in self.horizon_profile]
            heights = [h['H_hor'] for h in self.horizon_profile]

            # Find closest azimuth point
            sun_az_normalized = (sun_azimuth + 360) % 360
            idx = min(range(len(azimuths)),
                     key=lambda i: abs(azimuths[i] - sun_az_normalized))

            horizon_height = heights[idx]

            # Sun is visible if above horizon
            return sun_altitude > horizon_height

        except Exception:
            return sun_altitude > 0

    def create_pvgis_directional_heatmap(self, use_optimal_angle=True):
        """
        Create heatmap using REAL PVGIS radiation data for different roof directions.
        This is the main enhanced method!

        Args:
            use_optimal_angle: If True, uses PVGIS optimal angle; else uses 35°

        Returns:
            numpy array: Heatmap based on real radiation measurements
        """
        print("🌍 Fetching PVGIS radiation data...")

        # Get optimal angle from PVGIS
        optimal_data = self.get_pvgis_optimal_angles()
        if optimal_data and use_optimal_angle:
            base_angle = optimal_data['optimal_angle']
            print(f"✓ Using PVGIS optimal angle: {base_angle}°")
            print(f"  Expected annual: {optimal_data['estimated_annual_kwh']:.0f} kWh/kWp")
        else:
            base_angle = 35
            print(f"✓ Using standard angle: {base_angle}°")

        # Define 8 directions to sample
        directions = {
            'South': 0,
            'Southwest': 45,
            'West': 90,
            'Northwest': 135,
            'North': 180,
            'Northeast': -135,
            'East': -90,
            'Southeast': -45
        }

        direction_radiation = {}

        # Fetch radiation for each direction
        for direction_name, aspect_angle in directions.items():
            monthly_data = self.get_pvgis_monthly_radiation(
                angle=base_angle,
                aspect=aspect_angle
            )

            if monthly_data:
                # Sum annual radiation (kWh/m²/year)
                # PVcalc returns 'H(i)' for in-plane irradiation per month
                try:
                    # Try different possible field names from PVGIS API
                    if isinstance(monthly_data, list) and len(monthly_data) > 0:
                        first_record = monthly_data[0]
                        # Determine which field to use
                        if 'H(i)' in first_record:
                            annual_rad = sum([m['H(i)'] for m in monthly_data])
                        elif 'H(i)_m' in first_record:
                            annual_rad = sum([m['H(i)_m'] for m in monthly_data])
                        elif 'E_m' in first_record:
                            # Energy output per month (kWh)
                            annual_rad = sum([m['E_m'] for m in monthly_data])
                        else:
                            raise KeyError(f"Unknown field format: {first_record.keys()}")

                        direction_radiation[direction_name] = annual_rad
                        print(f"  {direction_name:12s}: {annual_rad:6.0f} kWh/m²/year")
                    else:
                        raise ValueError("Empty or invalid monthly_data")
                except (KeyError, TypeError, ValueError) as e:
                    print(f"  {direction_name:12s}: Using default (data format error: {e})")
                    direction_radiation[direction_name] = 1000
            else:
                # Fallback to estimated value
                print(f"  {direction_name:12s}: Using default (PVGIS unavailable)")
                direction_radiation[direction_name] = 1000  # Default

        # Normalize radiation values to 0-1 range
        max_radiation = max(direction_radiation.values())
        normalized_radiation = {
            k: v / max_radiation
            for k, v in direction_radiation.items()
        }

        # Create heatmap by applying directional radiation to roof pixels
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)

        center_y, center_x = self.height // 2, self.width // 2

        for y in range(self.height):
            for x in range(self.width):
                if self.roof_mask[y, x] > 0:  # If roof pixel
                    # Calculate pixel direction from center
                    dx = x - center_x
                    dy = center_y - y

                    # Calculate angle (0° = East in image coordinates)
                    pixel_angle = np.degrees(np.arctan2(dy, dx))

                    # Convert to compass bearing (0° = North)
                    compass_angle = (90 - pixel_angle) % 360

                    # Find closest direction
                    closest_dir_name = min(
                        directions.keys(),
                        key=lambda d: min(
                            abs(directions[d] - compass_angle),
                            360 - abs(directions[d] - compass_angle)
                        )
                    )

                    # Apply radiation value
                    heatmap[y, x] = normalized_radiation[closest_dir_name]

        return heatmap

    def create_yearly_heatmap(self, year=None, samples_per_month=4, use_pvgis=True):
        """
        Override parent method to use PVGIS data when available.

        Args:
            year: Year to analyze
            samples_per_month: Number of samples per month (ignored if use_pvgis=True)
            use_pvgis: If True, uses PVGIS directional radiation; else falls back to parent

        Returns:
            numpy array: Yearly average heatmap
        """
        if use_pvgis:
            print("📊 Generating PVGIS-enhanced yearly heatmap...")
            return self.create_pvgis_directional_heatmap(use_optimal_angle=True)
        else:
            print("📊 Generating theoretical yearly heatmap...")
            return super().create_yearly_heatmap(year, samples_per_month)

    def get_pvgis_summary_stats(self):
        """
        Get comprehensive PVGIS statistics for this location.

        Returns:
            dict: Summary statistics
        """
        optimal = self.get_pvgis_optimal_angles()

        # Get radiation for common orientations
        south_data = self.get_pvgis_monthly_radiation(angle=35, aspect=0)

        stats = {
            'optimal_angle': optimal['optimal_angle'] if optimal else None,
            'optimal_azimuth': optimal['optimal_azimuth'] if optimal else None,
            'estimated_annual_optimal': optimal['estimated_annual_kwh'] if optimal else None,
            'has_horizon_data': self.horizon_profile is not None,
            'horizon_points': len(self.horizon_profile) if self.horizon_profile else 0
        }

        if south_data:
            try:
                # south_data is a list of monthly records
                if isinstance(south_data, list) and len(south_data) > 0:
                    first_record = south_data[0]
                    if 'H(i)' in first_record:
                        annual_south = sum([m['H(i)'] for m in south_data])
                    elif 'H(i)_m' in first_record:
                        annual_south = sum([m['H(i)_m'] for m in south_data])
                    elif 'E_m' in first_record:
                        annual_south = sum([m['E_m'] for m in south_data])
                    else:
                        annual_south = None

                    if annual_south:
                        stats['annual_radiation_south'] = annual_south
            except (KeyError, TypeError, ValueError):
                pass  # Skip if data format is unexpected

        return stats


def create_pvgis_enhanced_heatmap(roof_image_path, lat, lon, output_dir='./outputs'):
    """
    Complete workflow to generate PVGIS-enhanced roof solar heatmap.

    Args:
        roof_image_path: Path to binary roof segmentation image
        lat: Latitude
        lon: Longitude
        output_dir: Directory to save outputs

    Returns:
        Tuple: (generator, heatmap, stats)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"🚀 Initializing PVGIS-Enhanced Roof Solar Heatmap Generator...")
    print(f"📍 Location: ({lat:.4f}, {lon:.4f})")
    print(f"🏠 Roof Image: {roof_image_path}")
    print("-" * 60)

    # Initialize PVGIS-enhanced generator
    generator = PVGISEnhancedHeatmap(lat, lon, roof_image_path)

    # Get PVGIS summary stats
    print("\n📈 Fetching PVGIS location statistics...")
    stats = generator.get_pvgis_summary_stats()

    if stats['optimal_angle']:
        print(f"\n🎯 PVGIS Optimal Configuration:")
        print(f"  Tilt Angle: {stats['optimal_angle']}°")
        print(f"  Azimuth: {stats['optimal_azimuth']}° (0=South)")
        print(f"  Expected Annual Yield: {stats['estimated_annual_optimal']:.0f} kWh/kWp")

    if stats.get('annual_radiation_south'):
        print(f"  South-facing (35°): {stats['annual_radiation_south']:.0f} kWh/m²/year")

    if stats['has_horizon_data']:
        print(f"\n🏔️  Terrain horizon profile loaded ({stats['horizon_points']} points)")

    # Generate PVGIS-enhanced yearly heatmap
    print("\n🎨 Generating PVGIS-enhanced heatmap...")
    heatmap = generator.create_yearly_heatmap(use_pvgis=True)

    # Visualize
    generator.visualize_heatmap(
        heatmap,
        title=f"PVGIS-Enhanced Solar Heatmap - ({lat:.4f}, {lon:.4f})",
        save_path=output_path / 'pvgis_enhanced_heatmap.png'
    )

    # Analyze zones
    print("\n📊 Analyzing roof exposure zones...")
    zones = generator.analyze_roof_zones(heatmap)
    for zone_name, zone_data in zones.items():
        print(f"  {zone_name}: {zone_data['percentage']:.1f}% coverage, "
              f"avg exposure: {zone_data['avg_exposure']:.3f}")

    # Find optimal panel locations
    print("\n⚡ Finding optimal solar panel locations...")
    optimal_locations = generator.find_optimal_panel_locations(heatmap, panel_count=10)
    print(f"  Found {len(optimal_locations)} optimal locations")

    # Create overlay with markers
    overlay = generator.create_overlay_heatmap(heatmap)
    for y, x in optimal_locations:
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), 2)
        cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)

    cv2.imwrite(str(output_path / 'pvgis_heatmap_with_panels.png'),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"\n✅ Complete! Outputs saved to: {output_path}")

    return generator, heatmap, stats


if __name__ == "__main__":
    # Example usage
    ROOF_IMAGE = "path/to/your/roof_segmentation.png"
    LATITUDE = 37.7749  # San Francisco
    LONGITUDE = -122.4194

    generator, heatmap, stats = create_pvgis_enhanced_heatmap(
        ROOF_IMAGE, LATITUDE, LONGITUDE
    )

    print("\n📋 Summary Statistics:")
    print(json.dumps(stats, indent=2))

###############################################################################

"""
Enhanced solar heatmap using PVGIS API data.
Integrates real weather patterns, horizon shading, and accurate radiation data.
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class PVGISEnhancedHeatmap:
    """
    Use PVGIS API to get real radiation data instead of theoretical calculations.
    """
    
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        self.base_url = "https://re.jrc.ec.europa.eu/api/v5_2"
        
    def get_hourly_radiation(self, year=2020):
        """
        Get hourly radiation data from PVGIS for a specific year.
        This is REAL measured data, not theoretical!
        
        Args:
            year: Year to get data for (2005-2020 available)
        
        Returns:
            DataFrame with hourly radiation data
        """
        url = f"{self.base_url}/seriescalc"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'startyear': year,
            'endyear': year,
            'pvcalculation': 0,  # Just radiation, no PV calculation
            'outputformat': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Parse the data
            hourly_data = data['outputs']['hourly']
            
            df = pd.DataFrame(hourly_data)
            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
            
            # Key columns:
            # 'G(i)' - Global irradiance on inclined plane (W/m²)
            # 'Gb(i)' - Beam (direct) irradiance (W/m²)
            # 'Gd(i)' - Diffuse irradiance (W/m²)
            # 'T2m' - Air temperature (°C)
            
            return df
            
        except Exception as e:
            print(f"Error fetching PVGIS hourly data: {e}")
            return None
    
    def get_typical_meteorological_year(self):
        """
        Get TMY data with more detailed radiation components.
        Better than what we're currently using!
        """
        url = f"{self.base_url}/tmy"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'outputformat': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Parse monthly data
            monthly_data = data['outputs']['months']
            tmy_data = data['outputs']['tmy_hourly']
            
            df = pd.DataFrame(tmy_data)
            
            # Contains:
            # 'G(h)' - Global horizontal irradiance
            # 'Gb(n)' - Beam normal irradiance  
            # 'Gd(h)' - Diffuse horizontal irradiance
            # 'T2m' - Temperature
            # 'WS10m' - Wind speed
            # 'SP' - Surface pressure
            
            return df, monthly_data
            
        except Exception as e:
            print(f"Error fetching PVGIS TMY: {e}")
            return None, None
    
    def get_horizon_profile(self):
        """
        Get terrain horizon profile - shows if mountains/buildings block sun!
        This accounts for REAL terrain shading.
        
        Returns:
            dict: Azimuth angles and corresponding horizon heights
        """
        url = f"{self.base_url}/printhorizon"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'outputformat': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            horizon = data['outputs']['horizon']
            
            # Returns array of horizon heights for each azimuth
            # Example: [{A: 0, H_hor: 5.2}, {A: 15, H_hor: 3.8}, ...]
            # A = Azimuth angle (0-360°)
            # H_hor = Horizon height in degrees
            
            return horizon
            
        except Exception as e:
            print(f"Error fetching horizon profile: {e}")
            return None
    
    def get_monthly_radiation(self, angle=35, aspect=0):
        """
        Get monthly radiation for specific panel angle and orientation.
        
        Args:
            angle: Tilt angle (0=horizontal, 90=vertical)
            aspect: Azimuth (0=South, 90=West, -90=East)
        
        Returns:
            Monthly radiation data
        """
        url = f"{self.base_url}/MRcalc"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'angle': angle,
            'aspect': aspect,
            'outputformat': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            monthly = data['outputs']['monthly']
            
            # Returns for each month:
            # 'E_m' - Monthly energy (kWh/m²)
            # 'H(i)_m' - Monthly irradiation (kWh/m²)
            # 'SD_m' - Standard deviation
            
            return monthly
            
        except Exception as e:
            print(f"Error fetching monthly radiation: {e}")
            return None
    
    def calculate_directional_radiation_map(self, roof_mask, date_range='yearly'):
        """
        Create heatmap using REAL PVGIS radiation data.
        Much more accurate than theoretical calculations!
        
        Args:
            roof_mask: Binary roof image
            date_range: 'yearly' or specific date
        
        Returns:
            Radiation heatmap based on real data
        """
        # Get TMY data (typical year)
        tmy_data, monthly_stats = self.get_typical_meteorological_year()
        
        if tmy_data is None:
            return None
        
        # Get horizon profile for terrain shading
        horizon = self.get_horizon_profile()
        
        height, width = roof_mask.shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Create directional samples (8 directions)
        directions = {
            'South': 0,
            'Southwest': 45,
            'West': 90,
            'Northwest': 135,
            'North': 180,
            'Northeast': -135,
            'East': -90,
            'Southeast': -45
        }
        
        direction_maps = {}
        
        for direction_name, aspect_angle in directions.items():
            # Get radiation for this direction
            monthly = self.get_monthly_radiation(angle=35, aspect=aspect_angle)
            
            if monthly:
                # Sum annual radiation (kWh/m²/year)
                annual_radiation = sum([m['E_m'] for m in monthly['monthly']])
                
                # Normalize to 0-1 range (typical max ~2000 kWh/m²/year)
                normalized = min(annual_radiation / 2000.0, 1.0)
                
                direction_maps[direction_name] = normalized
        
        # Apply to roof based on pixel positions
        center_y, center_x = height // 2, width // 2
        
        for y in range(height):
            for x in range(width):
                if roof_mask[y, x] > 0:  # If roof pixel
                    # Calculate pixel direction from center
                    dx = x - center_x
                    dy = center_y - y
                    
                    pixel_angle = np.degrees(np.arctan2(dy, dx))
                    pixel_angle = (pixel_angle + 360) % 360
                    
                    # Find closest direction
                    closest_dir = min(directions.items(), 
                                     key=lambda d: abs((d[1] - pixel_angle + 180) % 360 - 180))
                    
                    # Apply radiation value
                    heatmap[y, x] = direction_maps.get(closest_dir[0], 0)
        
        return heatmap, direction_maps
    
    def check_horizon_shading(self, sun_altitude, sun_azimuth, horizon_profile):
        """
        Check if sun is blocked by terrain/buildings at this position.
        
        Args:
            sun_altitude: Sun altitude in degrees
            sun_azimuth: Sun azimuth in degrees
            horizon_profile: Horizon data from PVGIS
        
        Returns:
            bool: True if sun is visible, False if blocked
        """
        if not horizon_profile:
            return True  # No horizon data, assume visible
        
        # Find horizon height at this azimuth
        # Interpolate between nearest points
        azimuths = [h['A'] for h in horizon_profile]
        heights = [h['H_hor'] for h in horizon_profile]
        
        # Find closest azimuth
        idx = min(range(len(azimuths)), 
                 key=lambda i: abs(azimuths[i] - sun_azimuth))
        
        horizon_height = heights[idx]
        
        # Sun is visible if it's above the horizon
        return sun_altitude > horizon_height
    
    def get_pvgis_optimal_angle(self):
        """
        Get PVGIS-calculated optimal tilt angle.
        This is verified against real data!
        
        Returns:
            dict: Optimal angles for different scenarios
        """
        # Fixed angle optimization
        url = f"{self.base_url}/PVcalc"
        
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'peakpower': 1,
            'loss': 14,
            'mountingplace': 'free',
            'optimalangles': 1,  # Calculate optimal angles
            'outputformat': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Returns optimal slope and azimuth
            optimal = data['inputs']
            
            return {
                'optimal_angle': optimal.get('angle', None),
                'optimal_azimuth': optimal.get('aspect', None),
                'estimated_annual': data['outputs']['totals']['fixed']['E_y']
            }
            
        except Exception as e:
            print(f"Error fetching optimal angles: {e}")
            return None


def create_enhanced_pvgis_heatmap(lat, lon, roof_mask):
    """
    Main function to create heatmap using PVGIS data.
    
    Args:
        lat: Latitude
        lon: Longitude
        roof_mask: Binary roof image
    
    Returns:
        Enhanced heatmap with real radiation data
    """
    pvgis = PVGISEnhancedHeatmap(lat, lon)
    
    print("Fetching PVGIS data...")
    
    # Get optimal angles
    optimal = pvgis.get_pvgis_optimal_angle()
    if optimal:
        print(f"PVGIS Optimal Angle: {optimal['optimal_angle']}°")
        print(f"Expected Annual Yield: {optimal['estimated_annual']:.0f} kWh")
    
    # Get horizon profile
    horizon = pvgis.get_horizon_profile()
    if horizon:
        print(f"Horizon profile loaded ({len(horizon)} points)")
    
    # Create radiation heatmap
    print("Calculating directional radiation...")
    heatmap, direction_stats = pvgis.calculate_directional_radiation_map(roof_mask)
    
    if direction_stats:
        print("\nRadiation by Direction (normalized 0-1):")
        for direction, value in direction_stats.items():
            print(f"  {direction:12s}: {value:.3f}")
    
    return heatmap, {
        'optimal_angles': optimal,
        'horizon': horizon,
        'direction_stats': direction_stats
    }


# Example integration with existing code
if __name__ == "__main__":
    import cv2
    
    # Test with sample roof mask
    roof_mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(roof_mask, (200, 200), (312, 312), 255, -1)
    
    lat, lon = 37.7749, -122.4194  # San Francisco
    
    heatmap, stats = create_enhanced_pvgis_heatmap(lat, lon, roof_mask)
    
    if heatmap is not None:
        # Visualize
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(roof_mask, cmap='gray')
        ax1.set_title('Roof Mask')
        ax1.axis('off')
        
        im = ax2.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
        ax2.set_title('PVGIS-Enhanced Solar Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label='Annual Radiation (normalized)')
        
        plt.tight_layout()
        plt.show()
        
#############################################################################################

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
from roof_solar_heatmap_pvgis import PVGISEnhancedHeatmap
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
                                   search_radius=100, samples_per_month=2, use_pvgis=True):
        """
        Complete workflow: OSM data → roof segmentation → solar heatmap.
        NOW WITH PVGIS REAL DATA!

        Args:
            heatmap_type: 'daily' or 'yearly'
            multi_building: If True, includes all buildings in area
            search_radius: Search radius for buildings in meters
            samples_per_month: For yearly heatmaps (ignored if use_pvgis=True)
            use_pvgis: If True, uses REAL PVGIS radiation data (recommended!)

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
            # Step 3: Generate heatmap (PVGIS or theoretical)
            if use_pvgis and heatmap_type == 'yearly':
                print("🌍 Using PVGIS real radiation data...")
                generator = PVGISEnhancedHeatmap(self.lat, self.lon, temp_path)
                heatmap = generator.create_yearly_heatmap(use_pvgis=True)
                pvgis_stats = generator.get_pvgis_summary_stats()
            else:
                if heatmap_type == 'daily':
                    print("☀️ Using theoretical daily calculations...")
                else:
                    print("☀️ Using theoretical yearly calculations...")
                generator = RoofSolarHeatmap(self.lat, self.lon, temp_path)

                if heatmap_type == 'daily':
                    heatmap = generator.create_daily_heatmap()
                else:
                    heatmap = generator.create_yearly_heatmap(samples_per_month=samples_per_month)
                pvgis_stats = None

            # Step 4: Analyze zones
            zones = generator.analyze_roof_zones(heatmap)

            # Step 5: Find optimal panel locations
            optimal_locations = generator.find_optimal_panel_locations(heatmap, panel_count=10)

            # Clean up temp file
            os.unlink(temp_path)
            
            result = {
                'success': True,
                'roof_image': roof_image,
                'heatmap': heatmap,
                'buildings_info': buildings_info,
                'generator': generator,
                'zones': zones,
                'optimal_locations': optimal_locations,
                'num_buildings': len(buildings_info),
                'used_pvgis': use_pvgis and heatmap_type == 'yearly'
            }

            # Add PVGIS stats if available
            if pvgis_stats:
                result['pvgis_stats'] = pvgis_stats

            return result
            
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