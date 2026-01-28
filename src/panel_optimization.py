"""
Panel Optimization Module

Handles panel grouping, contiguity analysis, and intelligent selection
for solar panel installations.
"""

import math
from shapely.geometry import Polygon


def find_contiguous_panel_groups(panels, min_group_size=2):
    """
    Groups panels into contiguous arrays based on spatial proximity.
    Filters out isolated panels that can't be wired practically.

    Args:
        panels: List of panel polygons
        min_group_size: Minimum panels required in a contiguous group (default 2)

    Returns:
        Tuple: (valid_panels, valid_groups)
            - valid_panels: List of panels belonging to groups >= min_group_size
            - valid_groups: List of groups (each group is a list of panel indices)
    """
    if not panels:
        return [], []

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
