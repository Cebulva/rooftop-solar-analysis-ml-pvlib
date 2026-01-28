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
    Selects N panels from the grid using optimal serpentine distribution.

    Distribution logic:
    - Tries to minimize number of rows (prefer 2x4 over 3+3+2)
    - Distributes panels evenly when possible (e.g., 8 panels → 4+4)
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
        return [], [], 0, "No panels available"

    total_available = len(all_panels)
    actual_count = min(target_count, total_available)

    # Get row capacities (how many panels each row can hold)
    row_capacities = [len(row) for row in rows_structure]

    if not row_capacities:
        return [], [], 0, "No panels available"

    # Sort capacities (largest first) for checking
    sorted_capacities = sorted(row_capacities, reverse=True)

    # Try to fit panels in fewest rows possible
    # Priority: 1 row > 2 rows > 3 rows > ...
    best_distribution = None

    for num_rows in range(1, len(rows_structure) + 1):
        # Calculate how many panels per row we'd need
        # For even distribution, all rows get same count
        if actual_count % num_rows == 0:
            panels_per_row = actual_count // num_rows

            # Check if we have enough rows with this capacity
            rows_with_capacity = sum(1 for cap in sorted_capacities if cap >= panels_per_row)

            if rows_with_capacity >= num_rows:
                # Perfect even distribution found!
                best_distribution = [panels_per_row] * num_rows
                break

        # For uneven distribution
        panels_per_row_base = actual_count // num_rows
        remainder = actual_count % num_rows

        # First rows get more panels (e.g., 7 panels in 3 rows → 3,2,2)
        distribution = []
        for i in range(num_rows):
            if i < remainder:
                distribution.append(panels_per_row_base + 1)
            else:
                distribution.append(panels_per_row_base)

        # Check if we can support this distribution
        # We need rows with sufficient capacity for each position
        distribution_sorted = sorted(distribution, reverse=True)
        can_fit = True
        for i, needed in enumerate(distribution_sorted):
            if i >= len(sorted_capacities) or sorted_capacities[i] < needed:
                can_fit = False
                break

        if can_fit:
            best_distribution = distribution
            break

    if best_distribution is None:
        return [], [], 0, f"Cannot fit {actual_count} panels with available rows"

    # Now select panels according to best distribution
    # We need to match distribution to actual rows (keep physical order for serpentine)

    # Sort rows with their indices to maintain order
    rows_with_indices = [(i, row) for i, row in enumerate(rows_structure)]

    # Sort by capacity (largest first) to match with distribution
    rows_with_indices.sort(key=lambda x: len(x[1]), reverse=True)

    # Create selection mapping
    selected_rows_mapping = []
    for i, panels_needed in enumerate(best_distribution):
        if i < len(rows_with_indices):
            row_idx, row = rows_with_indices[i]
            selected_row = row[:panels_needed]
            selected_rows_mapping.append((row_idx, selected_row))

    # Sort back by original row index to maintain physical order for serpentine
    selected_rows_mapping.sort(key=lambda x: x[0])

    # Extract just the selected rows
    selected_rows = [row for _, row in selected_rows_mapping]

    # Flatten to single list
    selected_flat = []
    for row in selected_rows:
        selected_flat.extend(row)

    print(f"\n📋 PANEL SELECTION:")
    print(f"   Target: {target_count} panels")
    print(f"   Selected: {len(selected_flat)} panels")
    print(f"   Distribution: {' + '.join(map(str, [len(r) for r in selected_rows]))}")
    print(f"   Rows used: {len(selected_rows)}")
    for i, row in enumerate(selected_rows):
        print(f"      Row {i+1}: {len(row)} panels (from LEFT side)")

    # Score contiguity
    score, is_acceptable, warning = score_contiguity(selected_flat, selected_rows, target_count)

    print(f"   Contiguity score: {score}")
    if not is_acceptable:
        print(f"   ⚠️  {warning}")

    return selected_flat, selected_rows, score, warning
