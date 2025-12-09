"""
Swipe curve generation utilities for human-like gestures.
"""

import random
from typing import List, Tuple


def generate_curved_path(
    start_x: int, start_y: int, end_x: int, end_y: int, num_points: int = 15
) -> List[Tuple[int, int]]:
    """
    Generate a curved path using a quadratic Bezier curve with randomized control point.

    Args:
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        end_x: Ending X coordinate
        end_y: Ending Y coordinate
        num_points: Number of intermediate points to generate

    Returns:
        List of (x, y) coordinate tuples along the curve
    """
    # Calculate distance to determine curve intensity
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    # Only add curve for distances > 100 pixels
    if distance <= 100:
        # For short swipes, return straight line
        return [(start_x, start_y), (end_x, end_y)]

    # Calculate midpoint
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2

    # Calculate perpendicular offset for control point
    # Random curve intensity between 10-25% of distance
    curve_intensity = random.uniform(0.1, 0.25)
    max_offset = distance * curve_intensity
    offset = random.uniform(-max_offset, max_offset)

    # Calculate perpendicular direction
    dx = end_x - start_x
    dy = end_y - start_y

    # Perpendicular vector is (-dy, dx) normalized
    if distance > 0:
        perp_x = -dy / distance
        perp_y = dx / distance

        # Control point with perpendicular offset
        control_x = mid_x + perp_x * offset
        control_y = mid_y + perp_y * offset
    else:
        control_x = mid_x
        control_y = mid_y

    # Generate points along quadratic Bezier curve
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)

        # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        x = (1 - t) ** 2 * start_x + 2 * (1 - t) * t * control_x + t**2 * end_x
        y = (1 - t) ** 2 * start_y + 2 * (1 - t) * t * control_y + t**2 * end_y

        points.append((int(x), int(y)))

    return points
