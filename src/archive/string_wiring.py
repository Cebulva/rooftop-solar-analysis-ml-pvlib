"""
String Wiring Module (Legacy)

This module now contains only physical panel row detection.
Electrical configuration has been moved to src/electrical_config.py which implements
proper series/parallel wiring based on electrical engineering standards.
"""

import math


def organize_panels_into_rows(panels, gsd, orientation="Portrait"):
    """
    Organizes panels into physical rows based on their Y-coordinates.
    Uses smart clustering to find natural row breaks.

    This function is still used for physical layout reference, but electrical
    configuration is now handled by src/electrical_config.py

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
    if gaps:
        gap_sizes = [g[1] for g in gaps]
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
        print(f"   Rows detected: {len(row_breaks) - 1}")
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
    for i, row in enumerate(rows):
        row_indices = [p[0] for p in row]
        row_y_vals = [panel_data[j][3] for j in range(len(panel_data))
                      if panel_data[j][0] in row_indices]
        avg_y = sum(row_y_vals) / len(row_y_vals) if row_y_vals else 0
        print(f"      Row {i+1}: {len(row)} panels (avg Y: {avg_y:.1f})")

    return rows
