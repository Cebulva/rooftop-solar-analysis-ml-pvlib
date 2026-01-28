"""
String Wiring Module

Handles electrical string configuration, row detection, serpentine wiring paths,
and wiring schematic generation for solar panel installations.
"""

import math
import numpy as np
import cv2


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
