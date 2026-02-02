"""
Electrical Configuration Module

Implements simple serpentine wiring for solar panels:
- Odd rows (1st, 3rd, 5th...): wired left-to-right
- Even rows (2nd, 4th, 6th...): wired right-to-left

This creates optimal wiring where adjacent panels in the sequence
are physically adjacent on the roof.
"""

import math
import cv2
import numpy as np

# Standard residential solar panel specifications
PANEL_POWER = 400  # Watts
PANEL_VOLTAGE = 31  # Volts (Vmp - Maximum Power Voltage)
PANEL_CURRENT = 13  # Amps (Imp - Maximum Power Current)

# Color palette for visualization
STRING_COLORS = [
    (255, 100, 100),  # Light blue
    (100, 255, 100),  # Light green
    (255, 180, 100),  # Light orange
]


def create_serpentine_wiring(panels, rows_structure):
    """
    Creates a serpentine wiring path through the panels.

    Odd rows: left-to-right (panels 1, 2, 3)
    Even rows: right-to-left (panels 6, 5, 4)

    Result: Panel 3 (end of row 1) connects directly to Panel 4 (end of row 2)

    Args:
        panels: Flat list of panel polygons
        rows_structure: List of lists (each inner list is a row of panels)

    Returns:
        List of panel indices in wiring order
    """
    if not rows_structure:
        return []

    # Build a mapping: panel polygon -> index in flat list
    panel_to_idx = {id(panel): idx for idx, panel in enumerate(panels)}

    wiring_path = []

    for row_num, row in enumerate(rows_structure):
        # Odd rows (0, 2, 4...): left-to-right
        # Even rows (1, 3, 5...): right-to-left
        if row_num % 2 == 0:
            # Odd row - keep order
            for panel in row:
                idx = panel_to_idx.get(id(panel))
                if idx is not None:
                    wiring_path.append(idx)
        else:
            # Even row - reverse order
            for panel in reversed(row):
                idx = panel_to_idx.get(id(panel))
                if idx is not None:
                    wiring_path.append(idx)

    print(f"\n🔌 SERPENTINE WIRING:")
    print(f"   Total panels: {len(wiring_path)}")
    print(f"   Rows: {len(rows_structure)}")

    # Show wiring order for debugging
    wiring_display = []
    panel_counter = 1
    for i, row in enumerate(rows_structure):
        direction = "→" if i % 2 == 0 else "←"
        row_numbers = list(range(panel_counter, panel_counter + len(row)))
        if i % 2 == 1:  # Odd rows wire right-to-left
            row_numbers.reverse()
        wiring_display.append((i+1, direction, row_numbers))
        panel_counter += len(row)

    for row_num, direction, numbers in wiring_display:
        print(f"      Row {row_num} {direction}: P{numbers[0]}→P{numbers[-1]}")

    return wiring_path


def calculate_electrical_specs(num_panels):
    """
    Calculates electrical specifications for a series string.

    Args:
        num_panels: Number of panels in series

    Returns:
        Dict with voltage, current, power
    """
    return {
        'voltage': num_panels * PANEL_VOLTAGE,
        'current': PANEL_CURRENT,
        'power': num_panels * PANEL_POWER
    }


def create_wiring_schematic(panels, rows_structure, wiring_path):
    """
    Creates a visual schematic showing the serpentine wiring pattern.

    Args:
        panels: Flat list of panel polygons
        rows_structure: List of lists (rows of panels)
        wiring_path: List of panel indices in wiring order

    Returns:
        Schematic image (numpy array)
    """
    if not panels or not rows_structure or not wiring_path:
        return None

    # Panel display settings
    panel_w = 100
    panel_h = 60
    spacing_x = 30
    spacing_y = 50
    margin = 120

    # Calculate canvas size
    max_panels_per_row = max(len(row) for row in rows_structure)
    num_rows = len(rows_structure)

    canvas_w = max_panels_per_row * (panel_w + spacing_x) + 2 * margin + 200
    canvas_h = num_rows * (panel_h + spacing_y) + 2 * margin + 200

    # Create white canvas
    schematic = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(schematic, "Solar Panel Wiring - Serpentine Pattern",
               (margin, 50), cv2.FONT_HERSHEY_SIMPLEX,
               1.2, (0, 0, 0), 3, cv2.LINE_AA)

    # System specs
    num_panels = len(wiring_path)
    specs = calculate_electrical_specs(num_panels)
    cv2.putText(schematic,
               f"System: {specs['voltage']:.0f}V @ {specs['current']:.0f}A = {specs['power']:.0f}W",
               (margin, 85), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (60, 60, 60), 2, cv2.LINE_AA)

    # Build panel position mapping
    panel_to_idx = {id(panel): idx for idx, panel in enumerate(panels)}
    panel_positions = {}  # panel_idx -> (x, y)
    color = STRING_COLORS[0]

    # Draw panels row by row
    current_y = 150

    for row_num, row in enumerate(rows_structure):
        direction = "→" if row_num % 2 == 0 else "←"

        # Row label
        cv2.putText(schematic, f"Row {row_num + 1} {direction}",
                   (margin - 100, current_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2, cv2.LINE_AA)

        # Draw each panel in this row
        for col_idx, panel in enumerate(row):
            panel_idx = panel_to_idx.get(id(panel))
            if panel_idx is None:
                continue

            x = margin + col_idx * (panel_w + spacing_x)
            y = current_y

            # Find this panel's position in wiring sequence
            wiring_num = wiring_path.index(panel_idx) + 1 if panel_idx in wiring_path else 0

            # Panel rectangle
            cv2.rectangle(schematic, (x, y), (x + panel_w, y + panel_h),
                         color, 3)

            # Light fill
            cv2.rectangle(schematic, (x + 3, y + 3), (x + panel_w - 3, y + panel_h - 3),
                         tuple([int(c * 0.3 + 255 * 0.7) for c in color]), -1)

            # Panel number in wiring sequence
            text = f"P{wiring_num}"
            cv2.putText(schematic, text, (x + 28, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

            # Voltage label
            cv2.putText(schematic, f"{PANEL_VOLTAGE:.0f}V", (x + 20, y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

            # Polarity markers
            # Negative (left side)
            cv2.circle(schematic, (x + 10, y + panel_h // 2), 8, (0, 0, 200), -1)
            cv2.putText(schematic, "-", (x + 6, y + panel_h // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Positive (right side)
            cv2.circle(schematic, (x + panel_w - 10, y + panel_h // 2), 8, (200, 0, 0), -1)
            cv2.putText(schematic, "+", (x + panel_w - 15, y + panel_h // 2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Store position for wiring
            panel_positions[panel_idx] = (x, y)

        current_y += panel_h + spacing_y

    # Draw wiring connections following serpentine path
    for i in range(len(wiring_path) - 1):
        p1_idx = wiring_path[i]
        p2_idx = wiring_path[i + 1]

        if p1_idx in panel_positions and p2_idx in panel_positions:
            x1, y1 = panel_positions[p1_idx]
            x2, y2 = panel_positions[p2_idx]

            # Connection from + of panel i to - of panel i+1
            start_pt = (x1 + panel_w - 10, y1 + panel_h // 2)
            end_pt = (x2 + 10, y2 + panel_h // 2)

            # Draw connection line
            cv2.line(schematic, start_pt, end_pt, (0, 0, 180), 4, cv2.LINE_AA)

            # Arrow at midpoint
            mid_x = (start_pt[0] + end_pt[0]) // 2
            mid_y = (start_pt[1] + end_pt[1]) // 2
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            length = math.sqrt(dx*dx + dy*dy)

            if length > 15:
                dx /= length
                dy /= length
                arrow_tip = (int(mid_x + dx * 15), int(mid_y + dy * 15))
                cv2.arrowedLine(schematic, (mid_x, mid_y), arrow_tip,
                               (0, 0, 150), 3, tipLength=0.4)

    # Mark start
    if wiring_path and wiring_path[0] in panel_positions:
        start_x, start_y = panel_positions[wiring_path[0]]
        cv2.circle(schematic, (start_x + panel_w // 2, start_y - 15), 12, (0, 180, 0), -1)
        cv2.putText(schematic, "START",
                   (start_x + panel_w // 2 - 25, start_y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2, cv2.LINE_AA)

    # Output to inverter
    if wiring_path and wiring_path[-1] in panel_positions:
        last_x, last_y = panel_positions[wiring_path[-1]]
        inverter_y = last_y + panel_h + 80

        cv2.arrowedLine(schematic,
                       (last_x + panel_w - 10, last_y + panel_h),
                       (last_x + panel_w - 10, inverter_y),
                       (0, 150, 0), 5, tipLength=0.2)

        cv2.putText(schematic, "To Inverter",
                   (last_x + panel_w - 60, inverter_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2, cv2.LINE_AA)

    # Add legend
    legend_y = canvas_h - 100
    cv2.putText(schematic, "Wiring Rules:",
               (margin, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(schematic, "• Series connection: + of Panel N connects to - of Panel N+1",
               (margin, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(schematic, "• Serpentine pattern minimizes wire length",
               (margin, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (60, 60, 60), 1, cv2.LINE_AA)

    return schematic
