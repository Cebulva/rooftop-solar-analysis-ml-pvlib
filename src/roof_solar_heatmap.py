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