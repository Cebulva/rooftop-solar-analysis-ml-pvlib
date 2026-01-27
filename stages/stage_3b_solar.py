import streamlit as st
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
from shapely import affinity
import ui_components as ui

# Updated Imports
from src.solar_engine import (
    get_masked_roof_array, 
    analyze_roof_geometry,  # NEW: Enhanced detection
    analyze_roof_texture,   # DEPRECATED: Kept for fallback
    draw_azimuth_arrow,
    get_sunny_polygon_mask,
    calculate_solar_potential,
    calculate_global_gsd,
    # Import configuration constants from solar_engine (single source of truth)
    MIN_FLAT_ROOF_COVERAGE,
    TEXTURE_VARIANCE_THRESHOLD,
    MIN_BRIGHTNESS_DIFF
)
from src.geometry_utils import calculate_azimuth, mask_to_polygon

# ==========================================
# ⚙️ UI CONFIGURATION (Tweak these)
# ==========================================
# Detection thresholds are configured in solar_engine.py
# Only UI-specific settings here
SHOW_DEBUG_INFO = True          # Display detection reasoning in UI
DEFAULT_PITCHED_TILT = 38.0     # Default tilt angle for pitched roofs (degrees)
DEFAULT_FLAT_TILT = 10.0        # Default tilt angle for flat roofs (degrees, optimal mounting)

# Solar Panel String Configuration
MIN_PANELS_PER_STRING = 2       # Minimum panels in series (inverter requirement)
MAX_PANELS_PER_STRING = 20      # Maximum panels in series (voltage limit)
TYPICAL_MIN_STRING = 8          # Typical minimum for efficiency
TYPICAL_MAX_STRING = 15         # Typical maximum for efficiency

# Panel Installation Margins
ROOF_EDGE_MARGIN = 0.30         # Minimum distance from roof edge (meters) - safety margin
PANEL_SPACING = 0.05            # Gap between adjacent panels (meters) - maintenance access

# Panel Orientation Options
# Portrait: 1.76m wide × 1.13m tall (vertical strings, taller than wide)
# Landscape: 1.13m wide × 1.76m tall (horizontal strings, wider than tall)
DEFAULT_PANEL_ORIENTATION = "Portrait"  # "Portrait" or "Landscape"

# Wiring Visualization Parameters
WIRING_LINE_THICKNESS = 1       # Thickness of connection lines (pixels) - try 1-3
WIRING_ARROW_SIZE = 6           # Size of direction arrows (pixels) - try 4-10
WIRING_START_MARKER_SIZE = 6    # Size of start marker circles (pixels) - try 4-10
WIRING_SHOW_ARROWS = True       # Show direction arrows on connections
# ==========================================

def update_azimuth():
    st.session_state.data["user_azimuth"] = float(st.session_state.az_slider_widget)
    # Recalculate irradiance when azimuth changes
    recalculate_irradiance()

def update_threshold():
    st.session_state.data["sun_threshold"] = int(st.session_state.sun_slider_widget)

def recalculate_irradiance():
    """Real-time irradiance calculation when azimuth or tilt changes"""
    if "final_poly" in st.session_state.data:
        lat = st.session_state.data.get("confirmed_lat")
        lon = st.session_state.data.get("confirmed_lon")
        user_azimuth = st.session_state.data.get("user_azimuth", 180)
        user_tilt = st.session_state.data.get("user_tilt", 38)
        
        irrad_val = calculate_solar_potential(lat, lon, user_tilt, user_azimuth)
        st.session_state.data["current_irradiance"] = irrad_val

def create_solar_panel_sprite(width_px, height_px, azimuth):
    """
    Creates a realistic solar panel sprite with orientation
    
    Args:
        width_px: Width in pixels
        height_px: Height in pixels (already projected based on tilt)
        azimuth: Rotation angle in degrees
    
    Returns:
        Rotated RGBA image of solar panel
    """
    # Create base panel with realistic solar cell appearance
    panel = np.ones((int(height_px), int(width_px), 4), dtype=np.uint8) * 255
    
    # Dark blue/black base for solar cells
    panel[:, :, 0] = 25   # Blue
    panel[:, :, 1] = 35   # Green
    panel[:, :, 2] = 60   # Red
    panel[:, :, 3] = 255  # Alpha
    
    # Add cell grid pattern (6x10 cells typical for modern panels)
    cells_h = 6
    cells_w = 10
    cell_h = height_px / cells_h
    cell_w = width_px / cells_w
    
    # Draw cell borders (silver/gray lines)
    for i in range(1, cells_h):
        y = int(i * cell_h)
        cv2.line(panel, (0, y), (int(width_px), y), (180, 180, 180, 255), 1)
    
    for j in range(1, cells_w):
        x = int(j * cell_w)
        cv2.line(panel, (x, 0), (x, int(height_px)), (180, 180, 180, 255), 1)
    
    # Add frame border
    cv2.rectangle(panel, (0, 0), (int(width_px)-1, int(height_px)-1), 
                  (60, 60, 60, 255), 2)
    
    # Add slight gradient to simulate light reflection
    gradient = np.linspace(0.8, 1.0, int(height_px))
    for i in range(3):
        panel[:, :, i] = (panel[:, :, i] * gradient[:, np.newaxis]).astype(np.uint8)
    
    # Rotate panel to match azimuth
    # OpenCV rotation: positive = counter-clockwise
    # Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
    # We need to rotate the sprite so it "faces" the azimuth direction
    center = (width_px / 2, height_px / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -azimuth, 1.0)
    
    # Calculate new bounding box to prevent clipping
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((height_px * sin) + (width_px * cos))
    new_h = int((height_px * cos) + (width_px * sin))
    
    # Adjust rotation matrix for new size
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(panel, rotation_matrix, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    
    return rotated

def overlay_panel_sprite(base_img, panel_polygon, sprite):
    """
    Overlays a solar panel sprite onto the base image at the polygon location
    
    Args:
        base_img: Background image (BGR)
        panel_polygon: Shapely Polygon defining panel location
        sprite: RGBA sprite image
    
    Returns:
        Updated base image with panel overlaid
    """
    # Get polygon bounds
    coords = np.array(panel_polygon.exterior.coords[:-1], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(coords.astype(np.int32))
    
    # Resize sprite to match polygon size
    resized_sprite = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Ensure bounds are within image
    if y < 0 or x < 0 or y+h > base_img.shape[0] or x+w > base_img.shape[1]:
        return base_img
    
    # Extract the region of interest
    roi = base_img[y:y+h, x:x+w]
    
    # Separate alpha channel
    sprite_bgr = resized_sprite[:, :, :3]
    alpha = resized_sprite[:, :, 3] / 255.0
    
    # Blend the sprite with the background
    for c in range(3):
        roi[:, :, c] = (alpha * sprite_bgr[:, :, c] + 
                        (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    
    base_img[y:y+h, x:x+w] = roi
    
    return base_img

def find_contiguous_panel_groups(panels, min_group_size=2):
    """
    Groups panels into contiguous arrays based on spatial proximity.
    Filters out isolated panels that can't be wired practically.
    
    Args:
        panels: List of panel polygons
        min_group_size: Minimum panels required in a contiguous group (default 2)
    
    Returns:
        List of valid panels that belong to groups >= min_group_size
    """
    if not panels:
        return []
    
    # Build adjacency graph - which panels are neighbors?
    def are_adjacent(panel1, panel2, tolerance=5.0):
        """Check if two panels are adjacent (within tolerance pixels)"""
        # Get centroids
        c1 = panel1.centroid
        c2 = panel2.centroid
        
        # Calculate distance
        distance = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
        
        # Adjacent if distance is less than ~1.5 times panel diagonal
        # (accounts for panel dimensions + small gap)
        return distance < tolerance * 2
    
    # Build adjacency list
    adjacency = {i: [] for i in range(len(panels))}
    
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            if are_adjacent(panels[i], panels[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Find connected components (groups of adjacent panels)
    visited = set()
    groups = []
    
    def dfs(node, group):
        """Depth-first search to find connected panels"""
        visited.add(node)
        group.append(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, group)
    
    for i in range(len(panels)):
        if i not in visited:
            group = []
            dfs(i, group)
            groups.append(group)
    
    # Filter groups by minimum size and collect valid panel indices
    valid_indices = set()
    valid_groups = []
    
    for group in groups:
        if len(group) >= min_group_size:
            valid_indices.update(group)
            valid_groups.append(group)
    
    # Return only panels that belong to valid groups, in original order
    valid_panels = [panels[i] for i in range(len(panels)) if i in valid_indices]
    
    # Debug logging
    print(f"\n🔗 PANEL GROUPING:")
    print(f"   Total panels: {len(panels)}")
    print(f"   Groups found: {len(groups)}")
    for idx, group in enumerate(groups):
        status = "✅ Valid" if len(group) >= min_group_size else "❌ Too small"
        print(f"      Group {idx+1}: {len(group)} panels {status}")
    print(f"   Valid panels after filtering: {len(valid_panels)}")
    
    return valid_panels, valid_groups


def optimize_panel_selection(all_panels, target_count, min_group_size=2):
    """
    Selects panels to reach target count while maintaining contiguous groups.
    
    Strategy:
    1. Group panels into contiguous arrays
    2. Sort groups by size (largest first)
    3. Select groups until reaching target count
    4. Prefer complete groups over partial groups
    
    Args:
        all_panels: All possible panel positions
        target_count: Desired number of panels
        min_group_size: Minimum panels per group
    
    Returns:
        List of selected panels (maintaining contiguity)
    """
    if not all_panels:
        return []
    
    # Find contiguous groups
    valid_panels, groups = find_contiguous_panel_groups(all_panels, min_group_size)
    
    if not valid_panels:
        print("⚠️  No valid contiguous groups found!")
        return []
    
    # If we already have fewer valid panels than target, return all
    if len(valid_panels) <= target_count:
        return valid_panels
    
    # Build group info with panel indices
    group_info = []
    panel_to_group = {}
    
    for group_id, group_indices in enumerate(groups):
        if len(group_indices) >= min_group_size:
            group_info.append({
                'id': group_id,
                'size': len(group_indices),
                'indices': sorted(group_indices)  # Panel indices in original list
            })
            for idx in group_indices:
                panel_to_group[idx] = group_id
    
    # Sort groups by size (largest first) for better utilization
    group_info.sort(key=lambda g: g['size'], reverse=True)
    
    # Select groups to reach target count
    selected_indices = set()
    remaining = target_count
    
    for group in group_info:
        if remaining <= 0:
            break
        
        if group['size'] <= remaining:
            # Take entire group
            selected_indices.update(group['indices'])
            remaining -= group['size']
            print(f"   Selected group {group['id']+1}: {group['size']} panels (complete)")
        else:
            # Take partial group (from beginning to maintain contiguity)
            partial_indices = group['indices'][:remaining]
            selected_indices.update(partial_indices)
            print(f"   Selected group {group['id']+1}: {remaining}/{group['size']} panels (partial)")
            remaining = 0
    
    # Return panels in original order
    selected_panels = [all_panels[i] for i in sorted(selected_indices)]
    
    print(f"   Final selection: {len(selected_panels)} panels")
    
    return selected_panels


def generate_panel_grid(sunny_mask, gsd, azimuth, tilt, panel_w=1.76, panel_h=1.13, 
                        edge_margin=0.30, panel_spacing=0.05, orientation="Portrait"):
    """
    Creates a grid of panels organized into electrical strings.
    CRITICAL: Prioritizes COMPLETE ROWS for efficient wiring.
    Adjusts ONLY the VISUAL HEIGHT based on tilt angle (Cosine Projection).
    Width remains constant as it's unaffected by tilt in top-down view.
    
    Args:
        sunny_mask: Binary mask of sunny area
        gsd: Ground Sample Distance (meters per pixel)
        azimuth: Panel orientation in degrees
        tilt: Panel tilt angle in degrees (0° = flat, 38° = typical pitched)
        panel_w: Physical panel width in meters (default 1.76m)
        panel_h: Physical panel height in meters (default 1.13m)
        edge_margin: Minimum distance from roof edge in meters (default 0.30m)
        panel_spacing: Gap between panels in meters (default 0.05m)
        orientation: "Portrait" (vertical) or "Landscape" (horizontal)
    
    Returns:
        List of panel polygons organized into complete rows
    """
    sunny_pts = mask_to_polygon(sunny_mask)
    if not sunny_pts:
        return []
    
    # Handle orientation (swap dimensions for landscape)
    if orientation == "Landscape":
        actual_w = panel_h  # 1.13m wide
        actual_h = panel_w  # 1.76m tall
    else:  # Portrait
        actual_w = panel_w  # 1.76m wide
        actual_h = panel_h  # 1.13m tall
    
    # Width is NOT affected by tilt in top-down view
    projected_w = actual_w
    projected_h = actual_h * math.cos(math.radians(tilt))
    
    # Convert to pixels
    pw_px = projected_w / gsd
    ph_px = projected_h / gsd
    edge_margin_px = edge_margin / gsd
    spacing_px = panel_spacing / gsd
    
    # Create sunny polygon with edge margin buffer
    sunny_poly = Polygon(sunny_pts).buffer(-edge_margin_px)
    
    if sunny_poly.is_empty or sunny_poly.area < (pw_px * ph_px):
        return []
    
    center = sunny_poly.centroid
    
    # Rotate polygon to align with azimuth
    aligned_poly = affinity.rotate(sunny_poly, -azimuth, origin=center)
    
    minx, miny, maxx, maxy = aligned_poly.bounds
    
    # Step sizes include spacing
    step_x = pw_px + spacing_px
    step_y = ph_px + spacing_px
    
    # IMPROVED: Generate panels ROW BY ROW and only keep COMPLETE rows
    # This ensures proper wiring efficiency
    all_rows = []
    
    y = miny
    while y + ph_px <= maxy:
        row_panels = []
        x = minx
        
        while x + pw_px <= maxx:
            # Create panel rectangle
            p = Polygon([
                (x, y), 
                (x + pw_px, y), 
                (x + pw_px, y + ph_px), 
                (x, y + ph_px)
            ])
            
            # Check if panel fits
            if aligned_poly.contains(p.buffer(-0.5)):
                row_panels.append(p)
            
            x += step_x
        
        # Only add rows with at least 2 panels (minimum for wiring)
        if len(row_panels) >= 2:
            all_rows.append(row_panels)
        
        y += step_y
    
    if not all_rows:
        return []
    
    # Flatten to single list of panels
    aligned_panels = []
    for row in all_rows:
        aligned_panels.extend(row)
    
    # Rotate all panels back to real-world orientation
    rotated_panels = [affinity.rotate(p, azimuth, origin=center) for p in aligned_panels]
    
    # Final validation: verify panels are within sunny mask
    valid_panels = []
    
    for panel in rotated_panels:
        panel_coords = np.array(panel.exterior.coords[:-1], dtype=np.int32)
        
        is_valid = True
        
        # Check all corners
        for corner in panel_coords:
            x, y = int(corner[0]), int(corner[1])
            
            if x < 0 or y < 0 or y >= sunny_mask.shape[0] or x >= sunny_mask.shape[1]:
                is_valid = False
                break
            
            if sunny_mask[y, x] == 0:
                is_valid = False
                break
        
        # Check center point
        if is_valid:
            center_x = int(np.mean(panel_coords[:, 0]))
            center_y = int(np.mean(panel_coords[:, 1]))
            
            if (center_x < 0 or center_y < 0 or 
                center_y >= sunny_mask.shape[0] or center_x >= sunny_mask.shape[1] or
                sunny_mask[center_y, center_x] == 0):
                is_valid = False
        
        if is_valid:
            valid_panels.append(panel)
    
    print(f"\n📐 GRID GENERATION:")
    print(f"   Rows created: {len(all_rows)}")
    for i, row in enumerate(all_rows):
        print(f"      Row {i+1}: {len(row)} panels")
    print(f"   Total panels: {len(valid_panels)}")
    
    return valid_panels


def organize_panels_into_strings(panels, min_string=8, max_string=15):
    """
    DEPRECATED: Simple string organization (kept for backward compatibility).
    Use optimize_string_wiring() for row-aware organization.
    
    Organizes panels into optimal electrical string configurations.
    
    Strategy:
    1. Try to create strings of equal length within typical range (8-15)
    2. Pair strings when possible (two strings of same length)
    3. Minimize number of different string lengths
    
    Args:
        panels: List of panel polygons
        min_string: Minimum panels per string
        max_string: Maximum panels per string
    
    Returns:
        List of string configurations: [{'string_id': 1, 'panel_indices': [...], 'panel_count': N}, ...]
    """
    total_panels = len(panels)
    
    if total_panels < MIN_PANELS_PER_STRING:
        return []
    
    # Simple equal division
    string_len = min(total_panels, max_string)
    if total_panels <= max_string:
        return [{
            'string_id': 1,
            'panel_indices': list(range(total_panels)),
            'panel_count': total_panels
        }]
    
    # Divide into equal strings
    num_strings = (total_panels + max_string - 1) // max_string
    panels_per_string = total_panels // num_strings
    
    strings = []
    start_idx = 0
    
    for i in range(num_strings):
        end_idx = min(start_idx + panels_per_string, total_panels)
        if i == num_strings - 1:
            end_idx = total_panels
        
        strings.append({
            'string_id': i + 1,
            'panel_indices': list(range(start_idx, end_idx)),
            'panel_count': end_idx - start_idx
        })
        start_idx = end_idx
    
    return strings


def organize_panels_into_rows(panels, gsd, orientation="Portrait"):
    """
    Organizes panels into physical rows based on their Y-coordinates.
    Uses smart clustering to find natural row breaks.
    
    Args:
        panels: List of panel polygons
        gsd: Ground Sample Distance (for tolerance calculation)
        orientation: Panel orientation
    
    Returns:
        List of rows, where each row is a list of (panel_index, panel, x_coord)
    """
    if not panels:
        return []
    
    # Get panel height for reference
    if orientation == "Landscape":
        panel_h = 1.76 * 0.788  # Approximate projected height
    else:
        panel_h = 1.13 * 0.788
    
    panel_h_px = panel_h / gsd
    
    # Get Y-coordinates of panel centroids
    panel_data = []
    for idx, panel in enumerate(panels):
        centroid = panel.centroid
        panel_data.append((idx, panel, centroid.x, centroid.y))
    
    # Sort by Y-coordinate (top to bottom)
    panel_data.sort(key=lambda p: p[3])
    
    if len(panel_data) == 1:
        return [[(panel_data[0][0], panel_data[0][1], panel_data[0][2])]]
    
    # Find gaps between consecutive panels
    y_positions = [p[3] for p in panel_data]
    gaps = []
    for i in range(len(y_positions) - 1):
        gap = y_positions[i + 1] - y_positions[i]
        gaps.append((i, gap))
    
    # Determine threshold for row breaks
    # A real row break should be significantly larger than within-row spacing
    # Within-row spacing: small (rotation artifacts, ~few pixels)
    # Between-row spacing: large (actual panel spacing, ~panel height)
    
    if gaps:
        gap_sizes = [g[1] for g in gaps]
        avg_gap = sum(gap_sizes) / len(gap_sizes)
        
        # Threshold: gaps larger than 0.5x panel height are row breaks
        # (Within-row variations should be < 0.3x panel height)
        row_break_threshold = panel_h_px * 0.5
        
        # Find row break indices
        row_breaks = [0]  # Start of first row
        for idx, gap in gaps:
            if gap > row_break_threshold:
                row_breaks.append(idx + 1)  # Start of next row
        row_breaks.append(len(panel_data))  # End marker
        
        print(f"\n📏 ROW DETECTION ({orientation}):")
        print(f"   Panel height: {panel_h_px:.1f} pixels")
        print(f"   Row break threshold: {row_break_threshold:.1f} pixels")
        print(f"   Gaps found: {len(gaps)}, Avg: {avg_gap:.1f}px")
        print(f"   Large gaps (row breaks): {sum(1 for g in gap_sizes if g > row_break_threshold)}")
    else:
        row_breaks = [0, len(panel_data)]
    
    # Group panels into rows based on breaks
    rows = []
    for i in range(len(row_breaks) - 1):
        start_idx = row_breaks[i]
        end_idx = row_breaks[i + 1]
        
        row_panels = []
        for j in range(start_idx, end_idx):
            idx, panel, x, y = panel_data[j]
            row_panels.append((idx, panel, x))
        
        # Sort row by X-coordinate (left to right)
        row_panels.sort(key=lambda p: p[2])
        rows.append(row_panels)
    
    # Debug output
    print(f"   Rows detected: {len(rows)}")
    for i, row in enumerate(rows):
        # Get Y positions for this row
        row_indices = [p[0] for p in row]
        row_y_vals = [panel_data[j][3] for j in range(len(panel_data)) 
                      if panel_data[j][0] in row_indices]
        avg_y = sum(row_y_vals) / len(row_y_vals) if row_y_vals else 0
        print(f"      Row {i+1}: {len(row)} panels (avg Y: {avg_y:.1f})")
    
    return rows


def optimize_string_wiring(panels, rows, min_string=8, max_string=15):
    """
    Creates optimal string configurations that follow actual physical layout.
    Prioritizes complete rows and logical wiring paths (serpentine).
    
    Args:
        panels: List of all panels
        rows: Organized rows from organize_panels_into_rows()
        min_string: Minimum panels per string
        max_string: Maximum panels per string
    
    Returns:
        List of string configurations with wiring paths
    """
    if not panels or not rows:
        return []
    
    total_panels = len(panels)
    
    print(f"\n🔌 STRING WIRING OPTIMIZATION:")
    print(f"   Total panels: {total_panels}")
    print(f"   Rows detected: {len(rows)}")
    for i, row in enumerate(rows):
        print(f"      Row {i+1}: {len(row)} panels")
    
    # Strategy: Try to create strings from complete rows
    # If rows don't align with string sizes, combine/split rows intelligently
    
    strings = []
    string_id = 1
    current_string_panels = []
    
    for row_idx, row in enumerate(rows):
        row_panel_indices = [p[0] for p in row]
        
        # Can we fit this entire row in current string?
        if len(current_string_panels) + len(row_panel_indices) <= max_string:
            # Add entire row to current string
            current_string_panels.extend(row_panel_indices)
            print(f"   Added row {row_idx+1} to string {string_id}")
            
        else:
            # Current string would exceed max, finalize it
            if len(current_string_panels) >= min_string:
                strings.append({
                    'string_id': string_id,
                    'panel_indices': current_string_panels,
                    'panel_count': len(current_string_panels)
                })
                print(f"   ✅ Finalized String {string_id}: {len(current_string_panels)} panels")
                string_id += 1
                current_string_panels = []
            
            # Start new string with current row
            current_string_panels.extend(row_panel_indices)
            print(f"   Started new string {string_id} with row {row_idx+1}")
    
    # Finalize last string
    if len(current_string_panels) >= min_string:
        strings.append({
            'string_id': string_id,
            'panel_indices': current_string_panels,
            'panel_count': len(current_string_panels)
        })
        print(f"   ✅ Finalized String {string_id}: {len(current_string_panels)} panels")
    elif current_string_panels:
        # Try to merge with previous string if it exists
        if strings and len(strings[-1]['panel_indices']) + len(current_string_panels) <= max_string:
            strings[-1]['panel_indices'].extend(current_string_panels)
            strings[-1]['panel_count'] = len(strings[-1]['panel_indices'])
            print(f"   ⚠️ Merged remaining {len(current_string_panels)} panels into String {strings[-1]['string_id']}")
        else:
            print(f"   ⚠️ Dropped {len(current_string_panels)} panels (below minimum)")
    
    # Generate serpentine wiring paths for each string
    for string in strings:
        string['wiring_path'] = generate_serpentine_path(
            string['panel_indices'], 
            panels, 
            rows
        )
    
    return strings


def generate_serpentine_path(panel_indices, panels, rows):
    """
    Generates a serpentine (back-and-forth) wiring path through panels.
    This minimizes cable runs and is standard for solar installations.
    
    CRITICAL: Properly handles row direction to ensure adjacent panels connect.
    
    Args:
        panel_indices: Indices of panels in this string
        panels: All panels
        rows: Row organization
    
    Returns:
        List of panel indices in wiring order
    """
    # Group panels by their row
    panels_by_row = {}
    for idx in panel_indices:
        # Find which row this panel belongs to
        for row_idx, row in enumerate(rows):
            if any(p[0] == idx for p in row):
                if row_idx not in panels_by_row:
                    panels_by_row[row_idx] = []
                panels_by_row[row_idx].append(idx)
                break
    
    # Sort rows
    sorted_rows = sorted(panels_by_row.keys())
    
    # Create serpentine path
    wiring_path = []
    for i, row_idx in enumerate(sorted_rows):
        row_panels = panels_by_row[row_idx]
        
        # Get actual X-coordinates for proper spatial sorting
        panel_positions = []
        for p_idx in row_panels:
            panel = panels[p_idx]
            panel_positions.append((p_idx, panel.centroid.x))
        
        # Sort by X-coordinate
        panel_positions.sort(key=lambda p: p[1])
        
        # CRITICAL FIX: Alternate direction for serpentine
        # This ensures the last panel of row N connects to the first panel of row N+1
        if i % 2 == 0:
            # Even rows: Left to right
            wiring_path.extend([p[0] for p in panel_positions])
        else:
            # Odd rows: Right to left (REVERSE) - this creates the serpentine!
            wiring_path.extend([p[0] for p in reversed(panel_positions)])
    
    # IMPORTANT: Log the wiring path for verification
    print(f"      Serpentine path: {wiring_path}")
    
    return wiring_path


def create_wiring_schematic(panels, strings, panel_rows, orientation="Portrait"):
    """
    Creates a schematic that EXACTLY mirrors how panels are placed on the roof.
    Uses the same row-by-row logic as the roof visualization.
    
    Args:
        panels: List of panel polygons IN THE ORDER THEY WERE GENERATED
        strings: String configurations with wiring paths
        panel_rows: Row organization from organize_panels_into_rows()
        orientation: Panel orientation
    
    Returns:
        Schematic image
    """
    if not panels or not strings or not panel_rows:
        return None
    
    # Use the ACTUAL rows as detected
    num_rows = len(panel_rows)
    max_cols = max(len(row) for row in panel_rows)
    
    # Schematic sizing
    panel_w = 90
    panel_h = 60
    spacing_x = 25
    spacing_y = 40
    margin = 100
    
    canvas_w = max_cols * (panel_w + spacing_x) + 2 * margin
    canvas_h = 200 + len(strings) * (num_rows * (panel_h + spacing_y) + 150)
    
    # Create canvas
    schematic = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # String colors
    string_colors = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255)
    ]
    
    # Title
    cv2.putText(schematic, "String Wiring Schematic", 
               (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(schematic, f"Layout: {num_rows} rows, max {max_cols} panels per row", 
               (margin, 85), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (100, 100, 100), 1, cv2.LINE_AA)
    
    current_y = 150
    
    # Draw each string
    for string_idx, string in enumerate(strings):
        color = string_colors[string_idx % len(string_colors)]
        wiring_path = string.get('wiring_path', string['panel_indices'])
        
        # Header
        cv2.putText(schematic, 
                   f"String {string['string_id']} ({string['panel_count']} panels)", 
                   (margin, current_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        current_y += 50
        
        # Draw panels row by row, EXACTLY as they appear in panel_rows
        panel_screen_positions = {}
        
        for row_idx, row in enumerate(panel_rows):
            # Get panel indices in this row (in left-to-right order)
            row_panel_indices = [p[0] for p in row]
            
            # Filter to only panels in current string
            row_panels_in_string = [idx for idx in row_panel_indices if idx in wiring_path]
            
            if not row_panels_in_string:
                continue  # Skip this row for this string
            
            # Determine if this row is reversed in wiring (serpentine)
            # Check if the first panel in this row comes AFTER the last panel visually
            if len(row_panels_in_string) > 1:
                first_panel_idx = row_panels_in_string[0]
                last_panel_idx = row_panels_in_string[-1]
                
                first_in_wiring = wiring_path.index(first_panel_idx)
                last_in_wiring = wiring_path.index(last_panel_idx)
                
                is_reversed = first_in_wiring > last_in_wiring
            else:
                is_reversed = False
            
            # Draw panels in this row
            for col_idx, panel_idx in enumerate(row_panels_in_string):
                # Calculate screen position
                x = margin + col_idx * (panel_w + spacing_x)
                y = current_y + row_idx * (panel_h + spacing_y)
                
                panel_screen_positions[panel_idx] = (x + panel_w//2, y + panel_h//2)
                
                # Draw panel box
                cv2.rectangle(schematic, (x, y), 
                             (x + panel_w, y + panel_h), 
                             color, 3)
                
                # Fill
                overlay = schematic.copy()
                cv2.rectangle(overlay, (x+3, y+3), 
                             (x + panel_w-3, y + panel_h-3), 
                             tuple([int(c * 0.2 + 255 * 0.8) for c in color]), -1)
                schematic = cv2.addWeighted(schematic, 0.6, overlay, 0.4, 0)
                
                # Panel number in wiring sequence
                wiring_num = wiring_path.index(panel_idx) + 1
                text = f"{wiring_num}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_x = x + (panel_w - text_size[0]) // 2
                text_y = y + (panel_h + text_size[1]) // 2
                cv2.putText(schematic, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Draw wiring connections following the actual path
        for i in range(len(wiring_path) - 1):
            p1 = wiring_path[i]
            p2 = wiring_path[i + 1]
            
            if p1 in panel_screen_positions and p2 in panel_screen_positions:
                pt1 = panel_screen_positions[p1]
                pt2 = panel_screen_positions[p2]
                
                # Connection line
                cv2.line(schematic, pt1, pt2, color, 3, cv2.LINE_AA)
                
                # Arrow
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 15:
                    dx /= length
                    dy /= length
                    arrow_tip = (int(mid_x + dx * 18), int(mid_y + dy * 18))
                    cv2.arrowedLine(schematic, (mid_x, mid_y), arrow_tip, 
                                   color, 3, tipLength=0.3)
        
        # Start marker
        if wiring_path and wiring_path[0] in panel_screen_positions:
            start_pos = panel_screen_positions[wiring_path[0]]
            cv2.circle(schematic, start_pos, 14, (0, 180, 0), -1)
            cv2.circle(schematic, start_pos, 14, (0, 0, 0), 2)
            cv2.putText(schematic, "START", 
                       (start_pos[0] - 28, start_pos[1] - 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 0), 1, cv2.LINE_AA)
        
        # End marker
        if wiring_path and wiring_path[-1] in panel_screen_positions:
            end_pos = panel_screen_positions[wiring_path[-1]]
            inv_y = end_pos[1] + 45
            cv2.arrowedLine(schematic, end_pos, (end_pos[0], inv_y), 
                           (50, 50, 50), 3, tipLength=0.3)
            cv2.putText(schematic, "To Inverter", 
                       (end_pos[0] - 45, inv_y + 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
        
        current_y += num_rows * (panel_h + spacing_y) + 80
    
    return schematic
    """
    Creates a clean schematic diagram showing string wiring based on ACTUAL roof layout.
    Uses the real row data from organize_panels_into_rows() to match roof reality.
    
    Args:
        panels: List of panel polygons (for reference)
        strings: String configurations with wiring paths
        panel_rows: ACTUAL row organization from organize_panels_into_rows()
        orientation: Panel orientation
    
    Returns:
        Schematic image showing wiring layout
    """
    # Panel representation size in schematic (larger for clarity)
    schematic_panel_w = 80
    schematic_panel_h = 50
    spacing_x = 20
    spacing_y = 40
    
    # Analyze actual row structure to determine layout
    # Count panels per row from the actual roof data
    rows_info = []
    for row_idx, row in enumerate(panel_rows):
        rows_info.append({
            'row_idx': row_idx,
            'panel_count': len(row),
            'panels': [p[0] for p in row]  # Panel indices
        })
    
    # Find maximum panels per row for canvas width
    max_panels_per_row = max([r['panel_count'] for r in rows_info]) if rows_info else 4
    total_rows = len(rows_info)
    
    # Calculate canvas size
    margin = 80
    canvas_w = max_panels_per_row * (schematic_panel_w + spacing_x) + 2 * margin
    
    # Calculate height: each string gets its own section
    canvas_h = 150  # Header
    for string in strings:
        # Count how many roof rows this string spans
        string_panels = set(string.get('wiring_path', string['panel_indices']))
        rows_in_string = sum(1 for r in rows_info if any(p in string_panels for p in r['panels']))
        canvas_h += rows_in_string * (schematic_panel_h + spacing_y) + 80
    
    # Create white canvas
    schematic = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # String colors
    string_colors = [
        (255, 80, 80),    # Light red
        (80, 255, 80),    # Light green  
        (80, 80, 255),    # Light blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    
    # Draw title
    cv2.putText(schematic, "String Wiring Schematic", 
               (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(schematic, "Actual roof layout - Follow numbered sequence", 
               (margin, 80), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Draw each string based on actual roof rows
    panel_positions = {}
    current_y = 120
    
    for string_idx, string in enumerate(strings):
        wiring_path = string.get('wiring_path', string['panel_indices'])
        color = string_colors[string_idx % len(string_colors)]
        
        # Draw string header
        cv2.putText(schematic, 
                   f"String {string['string_id']} ({string['panel_count']} panels)", 
                   (margin, current_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        current_y += 50
        
        # Group this string's panels by which roof row they're in
        string_panel_set = set(wiring_path)
        string_rows = []
        
        for row_info in rows_info:
            # Which panels from this row are in this string?
            panels_in_this_row = [p for p in row_info['panels'] if p in string_panel_set]
            if panels_in_this_row:
                string_rows.append({
                    'roof_row_idx': row_info['row_idx'],
                    'panels': panels_in_this_row,
                    'total_in_roof_row': row_info['panel_count']
                })
        
        # Now draw each row as it appears on the roof
        for local_row_idx, string_row in enumerate(string_rows):
            panels_in_row = string_row['panels']
            
            # Determine if this row should be reversed (serpentine)
            # Check the actual wiring path to see if these panels go left-to-right or right-to-left
            first_panel_idx_in_path = min(wiring_path.index(p) for p in panels_in_row)
            last_panel_idx_in_path = max(wiring_path.index(p) for p in panels_in_row)
            
            # If the wiring goes backwards (higher index first), reverse
            is_reversed = wiring_path[first_panel_idx_in_path] > wiring_path[last_panel_idx_in_path]
            
            # Sort panels by their position in wiring path
            panels_ordered = sorted(panels_in_row, key=lambda p: wiring_path.index(p))
            
            # Draw panels in this row
            for i, panel_idx in enumerate(panels_ordered):
                # Position based on actual roof layout
                if is_reversed:
                    # Right to left
                    col_pos = len(panels_in_row) - 1 - i
                else:
                    # Left to right
                    col_pos = i
                
                x = margin + col_pos * (schematic_panel_w + spacing_x)
                y = current_y
                
                panel_positions[panel_idx] = (x + schematic_panel_w//2, y + schematic_panel_h//2)
                
                # Draw panel rectangle
                cv2.rectangle(schematic, (x, y), 
                             (x + schematic_panel_w, y + schematic_panel_h), 
                             color, 3)
                
                # Fill with lighter version
                overlay = schematic.copy()
                cv2.rectangle(overlay, (x+3, y+3), 
                             (x + schematic_panel_w-3, y + schematic_panel_h-3), 
                             tuple([int(c * 0.2 + 255 * 0.8) for c in color]), -1)
                schematic = cv2.addWeighted(schematic, 0.6, overlay, 0.4, 0)
                
                # Panel number in string (position in wiring path)
                panel_num = wiring_path.index(panel_idx) + 1
                text = f"{panel_num}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = x + (schematic_panel_w - text_size[0]) // 2
                text_y = y + (schematic_panel_h + text_size[1]) // 2
                cv2.putText(schematic, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            
            # Add row label
            cv2.putText(schematic, 
                       f"Row {string_row['roof_row_idx']+1} ({len(panels_in_row)} panels)", 
                       (margin + max_panels_per_row * (schematic_panel_w + spacing_x) - 150, 
                        current_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
            
            current_y += schematic_panel_h + spacing_y
        
        # Draw connections for this string (following wiring path)
        for i in range(len(wiring_path) - 1):
            p1_idx = wiring_path[i]
            p2_idx = wiring_path[i + 1]
            
            if p1_idx in panel_positions and p2_idx in panel_positions:
                pt1 = panel_positions[p1_idx]
                pt2 = panel_positions[p2_idx]
                
                # Draw thick connection line
                cv2.line(schematic, pt1, pt2, color, 3, cv2.LINE_AA)
                
                # Draw arrow showing direction
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 10:
                    dx /= length
                    dy /= length
                    arrow_tip = (int(mid_x + dx * 15), int(mid_y + dy * 15))
                    cv2.arrowedLine(schematic, (mid_x, mid_y), arrow_tip, 
                                   color, 3, tipLength=0.3)
        
        # Draw start marker
        if wiring_path and wiring_path[0] in panel_positions:
            start_pos = panel_positions[wiring_path[0]]
            cv2.circle(schematic, start_pos, 12, (0, 180, 0), -1)
            cv2.circle(schematic, start_pos, 12, (0, 0, 0), 2)
            cv2.putText(schematic, "START", 
                       (start_pos[0] - 25, start_pos[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1, cv2.LINE_AA)
        
        # Draw end marker
        if wiring_path and wiring_path[-1] in panel_positions:
            end_pos = panel_positions[wiring_path[-1]]
            inv_y = end_pos[1] + 40
            cv2.arrowedLine(schematic, end_pos, (end_pos[0], inv_y), 
                           (50, 50, 50), 3, tipLength=0.3)
            cv2.putText(schematic, "To Inverter", 
                       (end_pos[0] - 40, inv_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        
        current_y += 50  # Space between strings
    
    return schematic
    """
    Creates a clean schematic diagram showing string wiring without the roof context.
    Panels are arranged to show the ACTUAL WIRING PATH clearly.
    
    Args:
        panels: List of panel polygons (for reference)
        strings: String configurations with wiring paths
        panel_w_px: Panel width in pixels (for sizing)
        panel_h_px: Panel height in pixels (for sizing)
        orientation: Panel orientation
    
    Returns:
        Schematic image showing wiring layout
    """
    # Determine schematic dimensions
    total_panels = sum(s['panel_count'] for s in strings)
    
    # Panel representation size in schematic (larger for clarity)
    schematic_panel_w = 80
    schematic_panel_h = 50
    spacing_x = 20
    spacing_y = 30
    
    # Arrange panels to match physical rows
    # Find maximum panels per row across all strings
    max_panels_per_row = 0
    for string in strings:
        wiring_path = string.get('wiring_path', string['panel_indices'])
        # Estimate panels per row (assume roughly square layout)
        panels_per_row = max(4, int(math.sqrt(len(wiring_path)) * 1.2))
        max_panels_per_row = max(max_panels_per_row, panels_per_row)
    
    # Calculate canvas size
    margin = 80
    canvas_w = max_panels_per_row * (schematic_panel_w + spacing_x) + 2 * margin
    
    # Estimate total rows needed
    total_rows = 0
    for string in strings:
        path_len = string['panel_count']
        string_rows = math.ceil(path_len / max_panels_per_row)
        total_rows += string_rows + 1  # +1 for spacing between strings
    
    canvas_h = total_rows * (schematic_panel_h + spacing_y) + 2 * margin + 150
    
    # Create white canvas
    schematic = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # String colors (same as main view)
    string_colors = [
        (255, 80, 80),    # Light red
        (80, 255, 80),    # Light green  
        (80, 80, 255),    # Light blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    
    # Draw title
    cv2.putText(schematic, "String Wiring Schematic", 
               (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(schematic, "Follow numbered sequence for installation", 
               (margin, 75), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (100, 100, 100), 1, cv2.LINE_AA)
    
    # Draw each string
    panel_positions = {}  # Store positions for connection drawing
    current_row = 0
    
    for string_idx, string in enumerate(strings):
        wiring_path = string.get('wiring_path', string['panel_indices'])
        color = string_colors[string_idx % len(string_colors)]
        
        # Draw string header
        header_y = margin + 100 + current_row * (schematic_panel_h + spacing_y)
        cv2.putText(schematic, 
                   f"String {string['string_id']} ({string['panel_count']} panels)", 
                   (margin, header_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        
        # Draw panels in WIRING ORDER arranged in rows
        # This creates the serpentine visual pattern
        row_in_string = 0
        col_in_row = 0
        
        for path_idx, panel_global_idx in enumerate(wiring_path):
            # Determine which row we're in
            row_in_string = path_idx // max_panels_per_row
            col_in_row = path_idx % max_panels_per_row
            
            # For odd rows (serpentine), reverse the column position
            if row_in_string % 2 == 1:
                col_in_row = max_panels_per_row - 1 - col_in_row
            
            # Calculate pixel position
            x = margin + col_in_row * (schematic_panel_w + spacing_x)
            y = header_y + row_in_string * (schematic_panel_h + spacing_y)
            
            panel_positions[panel_global_idx] = (x + schematic_panel_w//2, y + schematic_panel_h//2)
            
            # Draw panel rectangle
            cv2.rectangle(schematic, (x, y), 
                         (x + schematic_panel_w, y + schematic_panel_h), 
                         color, 3)
            
            # Fill with lighter version
            overlay = schematic.copy()
            cv2.rectangle(overlay, (x+3, y+3), 
                         (x + schematic_panel_w-3, y + schematic_panel_h-3), 
                         tuple([int(c * 0.2 + 255 * 0.8) for c in color]), -1)
            schematic = cv2.addWeighted(schematic, 0.6, overlay, 0.4, 0)
            
            # Panel number in string (1, 2, 3...)
            text = f"{path_idx + 1}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x + (schematic_panel_w - text_size[0]) // 2
            text_y = y + (schematic_panel_h + text_size[1]) // 2
            cv2.putText(schematic, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Draw connections for this string (following wiring path)
        for i in range(len(wiring_path) - 1):
            p1_idx = wiring_path[i]
            p2_idx = wiring_path[i + 1]
            
            if p1_idx in panel_positions and p2_idx in panel_positions:
                pt1 = panel_positions[p1_idx]
                pt2 = panel_positions[p2_idx]
                
                # Draw thick connection line
                cv2.line(schematic, pt1, pt2, color, 3, cv2.LINE_AA)
                
                # Draw arrow showing direction
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 10:  # Only draw arrow if connection is long enough
                    dx /= length
                    dy /= length
                    arrow_tip = (int(mid_x + dx * 15), int(mid_y + dy * 15))
                    cv2.arrowedLine(schematic, (mid_x, mid_y), arrow_tip, 
                                   color, 3, tipLength=0.3)
        
        # Draw start marker (green circle)
        if wiring_path and wiring_path[0] in panel_positions:
            start_pos = panel_positions[wiring_path[0]]
            cv2.circle(schematic, start_pos, 12, (0, 180, 0), -1)
            cv2.circle(schematic, start_pos, 12, (0, 0, 0), 2)
            cv2.putText(schematic, "START", 
                       (start_pos[0] - 25, start_pos[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1, cv2.LINE_AA)
        
        # Draw end marker (to inverter)
        if wiring_path and wiring_path[-1] in panel_positions:
            end_pos = panel_positions[wiring_path[-1]]
            # Arrow pointing down to inverter
            inv_y = end_pos[1] + 40
            cv2.arrowedLine(schematic, end_pos, (end_pos[0], inv_y), 
                           (50, 50, 50), 3, tipLength=0.3)
            cv2.putText(schematic, "To Inverter", 
                       (end_pos[0] - 40, inv_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        
        # Move to next string position
        rows_used = math.ceil(len(wiring_path) / max_panels_per_row)
        current_row += rows_used + 2  # +2 for spacing between strings
    
    return schematic


def draw_wiring_paths(image, panels, strings, 
                      line_thickness=1, 
                      arrow_size=6, 
                      marker_size=6,
                      show_arrows=True,
                      color_per_string=True):
    """
    Draws wiring connection lines on the image showing how panels are connected.
    
    Args:
        image: Base image to draw on
        panels: List of panel polygons
        strings: String configurations with wiring paths
        line_thickness: Thickness of connection lines in pixels (1-3 recommended)
        arrow_size: Size of direction arrows in pixels (4-10 recommended)
        marker_size: Size of start marker circles in pixels (4-10 recommended)
        show_arrows: Whether to show direction arrows
        color_per_string: Use different color for each string
    
    Returns:
        Image with wiring paths drawn
    """
    canvas = image.copy()
    
    # Different colors for each string (lighter, more visible colors)
    string_colors = [
        (255, 80, 80),    # Light red
        (80, 255, 80),    # Light green  
        (80, 80, 255),    # Light blue
        (255, 255, 80),   # Yellow
        (255, 80, 255),   # Magenta
        (80, 255, 255),   # Cyan
    ]
    
    for string_idx, string in enumerate(strings):
        wiring_path = string.get('wiring_path', string['panel_indices'])
        color = string_colors[string_idx % len(string_colors)] if color_per_string else (0, 255, 255)
        
        # Draw connections between consecutive panels
        for i in range(len(wiring_path) - 1):
            panel_idx_1 = wiring_path[i]
            panel_idx_2 = wiring_path[i + 1]
            
            # Get centroids
            c1 = panels[panel_idx_1].centroid
            c2 = panels[panel_idx_2].centroid
            
            pt1 = (int(c1.x), int(c1.y))
            pt2 = (int(c2.x), int(c2.y))
            
            # Draw connection line with configurable thickness
            cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)
            
            # Draw arrow at midpoint to show direction (if enabled)
            if show_arrows:
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                
                # Calculate arrow direction
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # Normalize
                    dx /= length
                    dy /= length
                    
                    # Arrow tip
                    arrow_tip = (int(mid_x + dx * arrow_size), int(mid_y + dy * arrow_size))
                    
                    # Draw arrow with configurable size
                    cv2.arrowedLine(canvas, (mid_x, mid_y), arrow_tip, color, 
                                   max(1, line_thickness), tipLength=0.5)
        
        # Draw start marker (first panel in string) with configurable size
        if wiring_path:
            first_panel = panels[wiring_path[0]]
            c = first_panel.centroid
            cv2.circle(canvas, (int(c.x), int(c.y)), marker_size, (0, 255, 0), -1)  # Green start
            
            # Add string label
            label_offset_x = -marker_size - 5
            label_offset_y = marker_size // 2
            cv2.putText(canvas, f"S{string['string_id']}", 
                       (int(c.x) + label_offset_x, int(c.y) + label_offset_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return canvas
    """
    Organizes panels into optimal electrical string configurations.
    
    Strategy:
    1. Try to create strings of equal length within typical range (8-15)
    2. Pair strings when possible (two strings of same length)
    3. Minimize number of different string lengths
    
    Args:
        panels: List of panel polygons
        min_string: Minimum panels per string
        max_string: Maximum panels per string
    
    Returns:
        List of string configurations: [(string_id, start_idx, end_idx, panel_count), ...]
    """
    total_panels = len(panels)
    
    if total_panels < MIN_PANELS_PER_STRING:
        return []
    
    # Cap at maximum installable
    total_panels = min(total_panels, MAX_PANELS_PER_STRING * 10)  # Max 10 strings
    
    string_configs = []
    
    # Try to find optimal string configuration
    # Priority: Equal-length strings in pairs
    
    best_config = None
    best_score = float('inf')
    
    # Try different string lengths in typical range
    for string_len in range(min_string, max_string + 1):
        num_strings = total_panels // string_len
        remainder = total_panels % string_len
        
        # Calculate score (prefer fewer strings, balanced lengths, pairs)
        # Score = variance penalty + remainder penalty - pairing bonus
        variance = 0
        pairing_bonus = 0
        
        if remainder >= MIN_PANELS_PER_STRING:
            # Can make an additional shorter string
            variance = abs(string_len - remainder)
            num_strings += 1
        elif remainder > 0:
            # Remainder too small, distribute across strings
            variance = remainder
        
        # Bonus for even number of strings (paired configuration)
        if num_strings % 2 == 0:
            pairing_bonus = -5
        
        score = variance * 2 + remainder + pairing_bonus
        
        if score < best_score and num_strings <= 10:  # Max 10 strings practical limit
            best_score = score
            best_config = (string_len, num_strings, remainder)
    
    if not best_config:
        # Fallback: single string or equal distribution
        string_len = min(total_panels, max_string)
        num_strings = max(1, total_panels // string_len)
        remainder = total_panels % string_len
        best_config = (string_len, num_strings, remainder)
    
    # Build string configuration
    string_len, num_strings, remainder = best_config
    
    current_idx = 0
    
    # Create main strings
    for string_id in range(num_strings):
        # Distribute remainder panels across first few strings
        panels_in_string = string_len
        if remainder > 0 and string_id < remainder:
            panels_in_string += 1
        
        end_idx = min(current_idx + panels_in_string, total_panels)
        
        if end_idx > current_idx:  # Valid string
            string_configs.append({
                'string_id': string_id + 1,
                'start_idx': current_idx,
                'end_idx': end_idx,
                'panel_count': end_idx - current_idx
            })
            current_idx = end_idx
    
    return string_configs

def show():
    st.header("Step 3b: Solar And Irradiance Analysis")

    if "final_poly" not in st.session_state.data:
        st.warning("Please complete the roof refinement in Step 2 first.")
        return

    # Back button
    if st.button("⬅️ Back to Questionnaire", key="back_to_step3a"):
        st.session_state.step = 3
        st.rerun()

    res = st.session_state.data["res"]
    lat = st.session_state.data["confirmed_lat"]
    lon = st.session_state.data["confirmed_lon"]

    # 1. INITIALIZATION & DATA RETRIEVAL
    # Get the recommendation from the questionnaire (Stage 3a)
    recommended_limit = st.session_state.data.get("recommended_count", 20)
    consumption_inputs = st.session_state.data.get("consumption_inputs", {})
    annual_kwh = consumption_inputs.get("annual_kwh", 3500)
    breakdown = consumption_inputs.get("breakdown", {})

    # Show consumption summary and recommendation from Stage 3a
    with st.container(border=True):
        st.subheader("📊 Your Estimated Annual Consumption")
        col_cons1, col_cons2 = st.columns([2, 1])

        with col_cons1:
            for item, kwh in breakdown.items():
                if item != 'Total':
                    st.write(f"- {item}: {kwh:,} kWh")

        with col_cons2:
            st.metric("Total", f"{annual_kwh:,} kWh/year")
            recommended_kwp = (recommended_limit * 440) / 1000
            st.metric("Recommended", f"{recommended_limit} panels ({recommended_kwp:.1f} kWp)")

    # Initialize the target count in session state if not present
    if "target_panel_count" not in st.session_state.data:
        st.session_state.data["target_panel_count"] = recommended_limit

    # 2. Geometry and Masking
    mask, roof_only = get_masked_roof_array(res["zoom_img"], st.session_state.data["final_poly"])
    gsd = calculate_global_gsd(lat, zoom=19) 
    pixel_area_m2 = gsd ** 2
    
    total_area_m2 = np.sum(mask > 0) * pixel_area_m2
    
    current_threshold = st.session_state.data.get("sun_threshold", 25)
    sun_mask = get_sunny_polygon_mask(roof_only, mask, threshold_offset=current_threshold)
    usable_area_m2 = np.sum(sun_mask > 0) * pixel_area_m2

    # 3. Analysis - IMPROVED ROOF TYPE DETECTION
    if "auto_roof_type" not in st.session_state.data:
        # Use enhanced detection with multiple heuristics
        detected_type, confidence, debug_info = analyze_roof_geometry(roof_only, mask, sun_mask)
        auto_azimuth = calculate_azimuth(st.session_state.data["final_poly"], img=roof_only)
        
        # Set default tilt based on roof type
        default_tilt = DEFAULT_PITCHED_TILT if detected_type == "Pitched" else DEFAULT_FLAT_TILT
        
        st.session_state.data.update({
            "auto_roof_type": detected_type,
            "user_azimuth": float(auto_azimuth),
            "user_tilt": default_tilt,
            "panel_orientation": DEFAULT_PANEL_ORIENTATION,  # Initialize orientation
            "detection_confidence": confidence,
            "detection_debug": debug_info
        })
        
        # Display detection reasoning in console/logs
        print(f"\n🏠 ROOF TYPE DETECTION:")
        print(f"   Result: {detected_type} (Confidence: {confidence:.1%})")
        print(f"   Reason: {debug_info.get('reason', 'N/A')}")
        print(f"   Coverage Ratio: {debug_info.get('coverage_ratio', 0):.1%}")
        print(f"   Brightness Range: {debug_info.get('brightness_range', 0):.0f}")
        print(f"   Texture StdDev: {debug_info.get('std_dev', 0):.1f}")
        print(f"   Default Tilt: {default_tilt}°")
    
    # Get current panel orientation
    current_orientation = st.session_state.data.get("panel_orientation", DEFAULT_PANEL_ORIENTATION)

    current_azimuth = st.session_state.data["user_azimuth"]
    user_tilt = st.session_state.data["user_tilt"]

    # Calculate real-time irradiance if not already calculated
    if "current_irradiance" not in st.session_state.data:
        recalculate_irradiance()

    current_irradiance = st.session_state.data.get("current_irradiance", 0)

    # 4. Generate Panel Grid with Contiguous Grouping
    all_possible_panels = generate_panel_grid(
        sun_mask, gsd, current_azimuth, user_tilt,
        edge_margin=ROOF_EDGE_MARGIN,
        panel_spacing=PANEL_SPACING,
        orientation=current_orientation
    )
    
    # Apply contiguous grouping filter (min 2 panels per group)
    valid_contiguous_panels, panel_groups = find_contiguous_panel_groups(
        all_possible_panels, 
        min_group_size=MIN_PANELS_PER_STRING
    )
    
    # Calculate actual panel dimensions based on orientation
    if current_orientation == "Landscape":
        display_w = 1.13  # Width when in landscape
        display_h = 1.76 * math.cos(math.radians(user_tilt))  # Projected height
    else:  # Portrait
        display_w = 1.76  # Width when in portrait
        display_h = 1.13 * math.cos(math.radians(user_tilt))  # Projected height
    
    # Debug logging
    print(f"\n🔧 PANEL GENERATION DEBUG:")
    print(f"   GSD: {gsd:.4f} m/pixel")
    print(f"   Orientation: {current_orientation}")
    print(f"   Panel dimensions: {display_w}m × {display_h:.2f}m (projected)")
    print(f"   Panel size in pixels: {display_w/gsd:.1f} × {display_h/gsd:.1f}")
    print(f"   Edge margin: {ROOF_EDGE_MARGIN}m ({ROOF_EDGE_MARGIN/gsd:.1f} px)")
    print(f"   Azimuth: {current_azimuth}°")
    print(f"   Tilt: {user_tilt}°")
    print(f"   Raw panels generated: {len(all_possible_panels)}")
    print(f"   After contiguous filtering: {len(valid_contiguous_panels)}")
    
    # Get target count and optimize selection
    limit = st.session_state.data["target_panel_count"]
    
    # Use optimized selection that maintains contiguity
    panels = optimize_panel_selection(
        all_possible_panels,
        target_count=limit,
        min_group_size=MIN_PANELS_PER_STRING
    )
    
    selected_count = len(panels)
    
    # Organize panels into physical rows for optimal wiring
    panel_rows = organize_panels_into_rows(panels, gsd, current_orientation)
    
    # Create optimized string configuration based on actual rows
    actual_string_configs = optimize_string_wiring(
        panels,
        panel_rows,
        min_string=TYPICAL_MIN_STRING,
        max_string=TYPICAL_MAX_STRING
    )
    
    print(f"   Selected for installation: {selected_count} panels")
    print(f"   String configuration: {len(actual_string_configs)} strings")
    for config in actual_string_configs:
        print(f"      String {config['string_id']}: {config['panel_count']} panels")

    # 5. UI Layout
    col_main, col_R = st.columns([4, 1.5])
    
    with col_main:
        # Create two visualizations: roof context and wiring schematic
        viz_tabs = st.tabs(["📍 Roof View", "🔌 Wiring Schematic"])
        
        with viz_tabs[0]:
            # Roof view with panels
            display_img = roof_only.copy()
            
            # Yellow Mask Overlay
            mask_overlay = np.zeros_like(display_img)
            mask_overlay[sun_mask > 0] = (0, 255, 255)
            display_img = cv2.addWeighted(display_img, 1.0, mask_overlay, 0.3, 0)
            
            display_img = draw_azimuth_arrow(display_img, current_azimuth)
            
            # Create solar panel sprite
            if current_orientation == "Landscape":
                panel_w_px = 1.13 / gsd
                panel_h_px = (1.76 * math.cos(math.radians(user_tilt))) / gsd
            else:  # Portrait
                panel_w_px = 1.76 / gsd
                panel_h_px = (1.13 * math.cos(math.radians(user_tilt))) / gsd
            
            panel_sprite = create_solar_panel_sprite(panel_w_px, panel_h_px, current_azimuth)
            
            # Overlay each panel as a sprite
            for p in panels:
                display_img = overlay_panel_sprite(display_img, p, panel_sprite)
            
            # DON'T draw wiring on roof view - it obscures panels
            # Wiring is shown clearly in the separate schematic tab
            
            st.image(display_img, use_container_width=True, caption="Panel placement on roof")
        
        with viz_tabs[1]:
            # Clean wiring schematic based on ACTUAL roof layout
            if actual_string_configs and panel_rows:
                schematic = create_wiring_schematic(
                    panels, 
                    actual_string_configs,
                    panel_rows,  # Pass actual roof rows!
                    current_orientation
                )
                st.image(schematic, use_container_width=True, caption="Electrical wiring diagram (matches roof layout)")
                
                # Add detailed wiring information
                st.markdown("### 📋 Wiring Instructions")
                for string in actual_string_configs:
                    with st.expander(f"String {string['string_id']} ({string['panel_count']} panels)"):
                        path = string.get('wiring_path', string['panel_indices'])
                        st.write(f"**Connection order:** {' → '.join([str(i+1) for i in range(len(path))])}")
                        st.write(f"**Pattern:** Serpentine (back-and-forth between rows)")
                        st.write(f"**Start panel:** Panel 1 (marked with green circle)")
                        st.write(f"**End panel:** Panel {len(path)} (connects to inverter)")
            else:
                st.info("Configure panels to see wiring schematic")
        
        with st.expander("🛠️ Analysis And Adjustments", expanded=True):
            # Panel Count Slider with String Validation
            max_capacity = len(valid_contiguous_panels)  # Use filtered count
            
            # Enforce minimum panel count for series string
            min_installable = max(MIN_PANELS_PER_STRING, 1)
            
            # Show warning if roof capacity is below minimum string requirement
            if max_capacity < MIN_PANELS_PER_STRING:
                st.error(f"⚠️ Roof capacity ({max_capacity} panels) is below the minimum string requirement "
                        f"({MIN_PANELS_PER_STRING} panels in series). Installation not viable with current settings.")
            elif max_capacity < TYPICAL_MIN_STRING:
                st.warning(f"⚠️ Roof capacity ({max_capacity} contiguous panels) is below typical minimum "
                          f"({TYPICAL_MIN_STRING} panels). Consider adjusting settings or using micro-inverters.")
            
            # Show info about filtered panels
            if len(all_possible_panels) > max_capacity:
                filtered_count = len(all_possible_panels) - max_capacity
                st.info(f"ℹ️ Filtered out {filtered_count} isolated panels (below minimum group size of {MIN_PANELS_PER_STRING}). "
                       f"Panels must be installed in contiguous arrays.")
            
            # Determine default target count (capped by string limits)
            default_target = min(
                st.session_state.data.get("target_panel_count", recommended_limit),
                max_capacity,
                MAX_PANELS_PER_STRING
            )
            
            # Ensure default meets minimum requirement
            default_target = max(default_target, min_installable)
            
            current_count = st.slider(
                "Number of Panels to Install", 
                min_value=min_installable, 
                max_value=min(max_capacity, MAX_PANELS_PER_STRING), 
                value=int(default_target), 
                key="panel_slider_widget",
                help=(f"String limits: Min {MIN_PANELS_PER_STRING}, Max {MAX_PANELS_PER_STRING}. "
                      f"Typical: {TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING} panels per string. "
                      f"Questionnaire recommended: {recommended_limit}. "
                      f"Only contiguous panel groups shown."),
                on_change=lambda: st.session_state.data.update({
                    "target_panel_count": st.session_state.panel_slider_widget
                })
            )
            
            # Visual feedback on string configuration
            if current_count < TYPICAL_MIN_STRING:
                st.caption(f"⚠️ Below typical minimum ({TYPICAL_MIN_STRING}). May require special inverter configuration.")
            elif current_count > TYPICAL_MAX_STRING:
                st.caption(f"ℹ️ Above typical maximum ({TYPICAL_MAX_STRING}). May require multiple strings.")
            else:
                st.caption(f"✅ Within typical range ({TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING} panels).")

            c1, c2 = st.columns(2)
            selected_type = c1.selectbox("Roof Form", ["Pitched", "Flat"], 
                                       index=0 if st.session_state.data["auto_roof_type"] == "Pitched" else 1)
            
            # Panel Orientation Toggle
            selected_orientation = c2.selectbox(
                "Panel Orientation",
                ["Portrait", "Landscape"],
                index=0 if current_orientation == "Portrait" else 1,
                help="Portrait: Vertical strings (1.76m wide). Landscape: Horizontal strings (1.13m wide)."
            )
            
            # Update orientation if changed
            if selected_orientation != st.session_state.data.get("panel_orientation"):
                st.session_state.data["panel_orientation"] = selected_orientation
                st.rerun()
            
            # Show detection confidence and reasoning if enabled
            if SHOW_DEBUG_INFO and "detection_debug" in st.session_state.data:
                debug = st.session_state.data["detection_debug"]
                confidence = st.session_state.data.get("detection_confidence", 0)
                
                # Show detection info in expander instead of column
                with st.expander("🔍 Detection Details", expanded=False):
                    st.metric("Detection Confidence", f"{confidence:.0%}", 
                             help=debug.get("reason", "Auto-detected roof type"))
                    st.write(f"**Reasoning:** {debug.get('reason', 'N/A')}")
                    st.write(f"**Coverage Ratio:** {debug.get('coverage_ratio', 0):.1%} "
                            f"(Flat if ≥ {MIN_FLAT_ROOF_COVERAGE:.0%})")
                    st.write(f"**Brightness Range:** {debug.get('brightness_range', 0):.0f}/255 "
                            f"(Pitched if ≥ 30)")
                    st.write(f"**Texture Variance:** {debug.get('std_dev', 0):.1f} "
                            f"(Pitched if > 15)")
            
            # Re-run if type changes to update tilt
            if selected_type != st.session_state.data["auto_roof_type"]:
                st.session_state.data["auto_roof_type"] = selected_type
                # Set appropriate default tilt for the roof type
                st.session_state.data["user_tilt"] = DEFAULT_PITCHED_TILT if selected_type == "Pitched" else DEFAULT_FLAT_TILT
                recalculate_irradiance()
                st.rerun()

            # Tilt Angle Slider - visible for PITCHED roofs
            if selected_type == "Pitched":
                st.slider(
                    "Panel Tilt Angle (°)", 
                    min_value=10, 
                    max_value=60, 
                    value=int(st.session_state.data["user_tilt"]),
                    key="tilt_slider_widget",
                    help=f"Typical range: 25-45°. Default: {DEFAULT_PITCHED_TILT}°",
                    on_change=lambda: st.session_state.data.update({
                        "user_tilt": float(st.session_state.tilt_slider_widget)
                    }) or recalculate_irradiance()
                )
            else:
                # For flat roofs, show info but don't allow adjustment (optimal mounting angle)
                st.info(f"ℹ️ Flat roof panels use {DEFAULT_FLAT_TILT}° mounting angle for optimal drainage and performance.")
            
            st.slider("Solar Orientation (Azimuth °)", 0, 359, int(current_azimuth), 
                      key="az_slider_widget", on_change=update_azimuth,
                      help="Watch the Irradiance Potential change as you rotate!")
            
            st.slider("Shadow Tolerance (Threshold)", 0, 100, int(current_threshold), 
                      key="sun_slider_widget", on_change=update_threshold)

    with col_R:
        st.markdown("### 📊 Global Metrics")
        st.metric("Total Roof Area", f"{total_area_m2:.1f} m²")
        st.metric("Usable Space", f"{usable_area_m2:.1f} m²")
        
        # String Configuration Display
        string_info = f"{selected_count}"
        if selected_count < MIN_PANELS_PER_STRING:
            string_info += " ⚠️"
        elif selected_count > MAX_PANELS_PER_STRING:
            string_info += " ⚠️"
        
        st.metric("Selected Panels", string_info, 
                 help=f"String limits: {MIN_PANELS_PER_STRING}-{MAX_PANELS_PER_STRING} panels. "
                      f"Typical: {TYPICAL_MIN_STRING}-{TYPICAL_MAX_STRING}")
        
        system_kwp = (selected_count * 440) / 1000
        st.metric("System Size", f"{system_kwp:.2f} kWp")
        
        # Accurate string configuration based on actual panel layout
        if selected_count >= MIN_PANELS_PER_STRING and actual_string_configs:
            num_strings = len(actual_string_configs)
            
            if num_strings == 1:
                config = actual_string_configs[0]
                st.caption(f"✅ Single string: {config['panel_count']} panels")
            else:
                # Show detailed string breakdown
                string_details = ", ".join([f"{cfg['panel_count']}" for cfg in actual_string_configs])
                st.caption(f"ℹ️ {num_strings} strings: [{string_details}] panels each")
                st.caption(f"📍 Wiring: Serpentine pattern (optimized for installation)")
        elif selected_count >= MIN_PANELS_PER_STRING:
            # Fallback if string organization failed
            st.caption(f"ℹ️ {selected_count} panels")
        
        # NEW: Real-time Irradiance Potential
        st.metric("☀️ Irradiance Potential", 
                  f"{current_irradiance:.0f} W/m²",
                  help="Real-time solar irradiance at current azimuth. Rotate to see changes!")
        
        # Calculate annual energy production estimate
        # Simplified calculation: kWp × irradiance × hours × efficiency
        peak_sun_hours = 4.5  # Average for Central Europe
        system_efficiency = 0.85  # Account for losses
        annual_production = system_kwp * peak_sun_hours * 365 * (current_irradiance / 1000) * system_efficiency
        
        st.metric("Est. Annual Production", 
                  f"{annual_production:,.0f} kWh/year",
                  help="Estimated yearly energy production")
        
        # Coverage percentage
        coverage_pct = (annual_production / annual_kwh * 100) if annual_kwh > 0 else 0
        st.metric("Coverage", 
                  f"{coverage_pct:.0f}%",
                  help="Percentage of your consumption covered by solar")
        
        if st.button("Run Simulation And Generate Report ☀️", type="primary", use_container_width=True):
            # Validate minimum panel count before generating report
            if selected_count < MIN_PANELS_PER_STRING:
                st.error(f"Cannot generate report: Minimum {MIN_PANELS_PER_STRING} panels required for series connection.")
            else:
                # Store accurate string configuration
                string_config_data = {
                    'num_strings': len(actual_string_configs),
                    'strings': [
                        {
                            'string_id': cfg['string_id'],
                            'panel_count': cfg['panel_count']
                        }
                        for cfg in actual_string_configs
                    ]
                }
                
                st.session_state.data["solar_results"] = {
                    "total_roof_area_m2": total_area_m2,
                    "usable_roof_area_m2": usable_area_m2,
                    "panel_count": selected_count,
                    "system_kwp": system_kwp,
                    "azimuth": current_azimuth,
                    "tilt_angle": user_tilt,
                    "panel_orientation": current_orientation,  # NEW: Save orientation
                    "roof_form": st.session_state.data["auto_roof_type"],
                    "irradiance_potential": current_irradiance,
                    "annual_production_kwh": annual_production,
                    "coverage_percentage": coverage_pct,
                    "string_configuration": string_config_data
                }
                st.session_state.step = 5
                st.rerun()