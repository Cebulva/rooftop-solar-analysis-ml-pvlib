"""
Panel Selection Module

Simple row-by-row panel selection that naturally maintains serpentine wiring.
"""


def select_panels_from_grid(all_panels, rows_structure, target_count):
    """
    Selects N panels from the grid, taking them row by row.

    This automatically creates balanced distribution:
    - If 8 panels fit in one row, take 8 from row 1
    - If only 6 fit per row, take 6 from row 1, then 2 from row 2

    Args:
        all_panels: Flat list of all available panels
        rows_structure: List of lists (each inner list is a row)
        target_count: Number of panels to select

    Returns:
        Tuple: (selected_panels_flat, selected_rows_structure)
            - selected_panels_flat: List of selected panels
            - selected_rows_structure: Row structure for selected panels
    """
    if not rows_structure or target_count <= 0:
        return [], []

    total_available = len(all_panels)
    actual_count = min(target_count, total_available)

    # Select panels row by row
    # CRITICAL: For serpentine wiring, odd rows must be selected from the RIGHT
    # so they're adjacent to where the previous row ends
    selected_rows = []
    panels_taken = 0

    for row_idx, row in enumerate(rows_structure):
        if panels_taken >= actual_count:
            break

        # How many panels do we still need?
        remaining = actual_count - panels_taken

        # Take up to the remaining count from this row
        panels_from_row = min(len(row), remaining)

        # For odd rows (which wire right-to-left), select from RIGHT side
        # This ensures P6 (rightmost row 1) connects to P7 (rightmost row 2)
        if row_idx % 2 == 1:
            # Odd row - take from RIGHT (end of list)
            selected_row = row[-panels_from_row:] if panels_from_row < len(row) else row
        else:
            # Even row - take from LEFT (start of list)
            selected_row = row[:panels_from_row]

        selected_rows.append(selected_row)
        panels_taken += panels_from_row

    # Flatten to single list
    selected_flat = []
    for row in selected_rows:
        selected_flat.extend(row)

    print(f"\n📋 PANEL SELECTION:")
    print(f"   Target: {target_count} panels")
    print(f"   Selected: {len(selected_flat)} panels")
    print(f"   Rows used: {len(selected_rows)}")
    for i, row in enumerate(selected_rows):
        side = "LEFT" if i % 2 == 0 else "RIGHT"
        print(f"      Row {i+1}: {len(row)} panels (from {side} side for serpentine)")

    return selected_flat, selected_rows
