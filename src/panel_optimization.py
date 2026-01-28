"""
Panel Selection Module

Simple row-by-row panel selection that naturally maintains serpentine wiring.
"""

import math


def score_contiguity(panels, rows_structure, target_count):
    """
    Scores panel layout based on contiguity and adjacency.

    Scoring system:
    - Adjacent panels: +10 per pair
    - Single row layout: +5 per panel
    - Isolated panels: -20 per panel
    - Full rows preferred: +2 per full row

    Args:
        panels: Flat list of panel polygons
        rows_structure: List of lists (rows of panels)
        target_count: Target number of panels

    Returns:
        tuple: (score, is_acceptable, warning_message)
    """
    if not panels or target_count <= 0:
        return 0, True, None

    # Single panel is always acceptable
    if target_count == 1:
        return 10, True, None

    # Build adjacency map using panel centroids
    adjacency_count = {}
    for i, panel in enumerate(panels):
        adjacency_count[i] = 0

    # Distance threshold: 2.5m (accounts for 1.76m panel width + 0.5m max gap + margin)
    distance_threshold = 2.5

    # Check all pairs for adjacency
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            c1 = panels[i].centroid
            c2 = panels[j].centroid
            distance = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)

            # Convert to meters (assuming centroids are in pixels, need GSD)
            # For now, use a heuristic: if distance < 100 pixels, they're adjacent
            # This works for typical roof scales (GSD ~0.03m/pixel → 100px = 3m)
            if distance < 100:  # pixels
                adjacency_count[i] += 1
                adjacency_count[j] += 1

    # Calculate score
    score = 0
    isolated_panels = []

    # Count adjacent pairs (each connection counted once)
    total_connections = sum(adjacency_count.values()) // 2
    score += total_connections * 10

    # Check for isolated panels (no neighbors)
    for i, count in adjacency_count.items():
        if count == 0:
            isolated_panels.append(i)
            score -= 20

    # Massive bonus for single-row layout (highly preferred for serpentine wiring)
    if len(rows_structure) == 1:
        score += 100 * len(panels)  # 100 points per panel in single row

    # Bonus for full rows
    for row in rows_structure:
        if len(row) >= 4:  # Consider row "full" if it has 4+ panels
            score += 2 * len(row)

    # Determine if acceptable
    is_acceptable = len(isolated_panels) == 0

    warning_message = None
    if isolated_panels:
        warning_message = (
            f"⚠️ Cannot place {target_count} panels contiguously with current azimuth/spacing. "
            f"{len(isolated_panels)} panel(s) will be isolated. "
            f"Consider adjusting panel orientation or roof area."
        )

    return score, is_acceptable, warning_message


def select_panels_from_grid(all_panels, rows_structure, target_count):
    """
    Selects N panels from the grid, distributing them evenly across rows.

    Distribution logic:
    - Calculates minimum rows needed for target count
    - Distributes panels evenly (e.g., 8 panels → 4+4, not 6+2)
    - Always selects from LEFT of each row (no gaps)
    - Scores contiguity after selection

    Args:
        all_panels: Flat list of all available panels
        rows_structure: List of lists (each inner list is a row)
        target_count: Number of panels to select

    Returns:
        Tuple: (selected_panels_flat, selected_rows_structure, contiguity_score, warning_message)
            - selected_panels_flat: List of selected panels
            - selected_rows_structure: Row structure for selected panels
            - contiguity_score: int, higher is better
            - warning_message: str or None if panels are contiguous
    """
    if not rows_structure or target_count <= 0:
        return [], []

    total_available = len(all_panels)
    actual_count = min(target_count, total_available)

    # Determine how many rows we need and distribute evenly
    # ALWAYS fill from LEFT physically to avoid gaps

    # Calculate row capacities
    row_capacities = [len(row) for row in rows_structure]

    if not row_capacities:
        return [], [], 0, "No panels available"

    # Determine minimum rows needed
    max_per_row = max(row_capacities)
    rows_needed = min(len(rows_structure), (actual_count + max_per_row - 1) // max_per_row)

    # Distribute panels evenly across the needed rows
    panels_per_row = actual_count // rows_needed
    remainder = actual_count % rows_needed

    selected_rows = []

    for row_idx in range(rows_needed):
        row = rows_structure[row_idx]

        # Calculate how many panels this row gets
        panels_for_this_row = panels_per_row
        if row_idx < remainder:
            panels_for_this_row += 1

        # Don't exceed row capacity
        panels_for_this_row = min(panels_for_this_row, len(row))

        # Always take from LEFT (start of list) to avoid visual gaps
        selected_row = row[:panels_for_this_row]
        selected_rows.append(selected_row)

    # Flatten to single list
    selected_flat = []
    for row in selected_rows:
        selected_flat.extend(row)

    print(f"\n📋 PANEL SELECTION:")
    print(f"   Target: {target_count} panels")
    print(f"   Selected: {len(selected_flat)} panels")
    print(f"   Rows used: {len(selected_rows)}")
    for i, row in enumerate(selected_rows):
        print(f"      Row {i+1}: {len(row)} panels (from LEFT side)")

    # Score contiguity
    score, is_acceptable, warning = score_contiguity(selected_flat, selected_rows, target_count)

    print(f"   Contiguity score: {score}")
    if not is_acceptable:
        print(f"   ⚠️  {warning}")

    return selected_flat, selected_rows, score, warning
