"""
Stealth ADB Tools - Human-like interaction patterns for Android devices.

This module provides stealthy UI interactions with randomization and curved gestures
to mimic natural human behavior on Android devices.
"""

import asyncio
import random
from typing import List, Tuple

from droidrun.tools.adb import AdbTools


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


class StealthAdbTools(AdbTools):
    """
    Stealth Android device tools with human-like interaction patterns.

    Extends AdbTools with randomization features:
    - Randomized tap positions within element bounds
    - Word-by-word typing with random delays
    - Curved swipe paths that mimic natural hand movements
    """

    def _extract_element_coordinates_by_index(self, index: int) -> Tuple[int, int]:
        """
        Extract center coordinates from an element by its index with randomization.

        Args:
            index: Index of the element to find and extract coordinates from

        Returns:
            Tuple of (x, y) randomized coordinates within element bounds

        Raises:
            ValueError: If element not found, bounds format is invalid, or missing bounds
        """

        def collect_all_indices(elements):
            """Collect all indices from elements and their children."""
            indices = []
            for element in elements:
                if "index" in element:
                    indices.append(element["index"])
                if "children" in element:
                    indices.extend(collect_all_indices(element["children"]))
            return indices

        def find_element_by_index(elements, target_index):
            """Find an element with the given index (including in children)."""
            for item in elements:
                if item.get("index") == target_index:
                    return item
                if "children" in item:
                    result = find_element_by_index(item["children"], target_index)
                    if result:
                        return result
            return None

        # Check if we have cached elements
        if not self.clickable_elements_cache:
            raise ValueError("No clickable elements cached. Call get_state() first.")

        # Find the element with the given index (including in children)
        element = find_element_by_index(self.clickable_elements_cache, index)

        if not element:
            available_indices = collect_all_indices(self.clickable_elements_cache)
            raise ValueError(
                f"Element with index {index} not found in cached clickable elements.\n"
                f"Available indices: {sorted(available_indices)}\n"
                f"Total elements: {len(available_indices)}"
            )

        # Get the bounds of the element
        bounds_str = element.get("bounds")
        if not bounds_str:
            raise ValueError(
                f"Element with index {index} does not have bounds attribute.\n"
                f"Element: {element}\n"
                f"This may indicate a non-clickable element. "
                f"Please verify the element has valid coordinates."
            )

        # Parse the bounds (format: "left,top,right,bottom")
        try:
            left, top, right, bottom = map(int, bounds_str.split(","))
        except ValueError as e:
            raise ValueError(f"Invalid bounds format '{bounds_str}': {e}") from e

        # Calculate the center of the element
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # Add randomization with safe zone (avoid edges)
        width = right - left
        height = bottom - top

        # Use 40% of the width/height as safe zone (20% from each side)
        safe_zone_factor = 0.4
        x_range = int(width * safe_zone_factor)
        y_range = int(height * safe_zone_factor)

        # Ensure we have at least some range
        x_range = max(x_range, 5)
        y_range = max(y_range, 5)

        # Add random offset within safe zone
        x_offset = random.randint(-x_range // 2, x_range // 2)
        y_offset = random.randint(-y_range // 2, y_range // 2)

        tap_x = center_x + x_offset
        tap_y = center_y + y_offset

        # Ensure coordinates are still within bounds (safety check)
        tap_x = max(left + 2, min(tap_x, right - 2))
        tap_y = max(top + 2, min(tap_y, bottom - 2))

        return tap_x, tap_y

    async def input_text(self, text: str, index: int = -1, clear: bool = False) -> str:
        """
        Type text with randomization - splits by spaces and types word by word with delays.

        Args:
            text: The text to type
            index: The index of the element to type into (-1 for already focused)
            clear: Whether to clear existing text before typing (default: False)

        Returns:
            Result message from the input operation
        """
        # Split by spaces and type each word separately with delay
        words = text.split(" ")
        results = []

        for i, word in enumerate(words):
            # Type the word using parent class method
            result = await super().input_text(
                word, index=index, clear=(clear and i == 0)
            )
            results.append(result)

            # Add space after word (except for last word)
            if i < len(words) - 1:
                await super().input_text(" ", index=-1, clear=False)
                # Random delay between words (100-300ms)
                await asyncio.sleep(random.uniform(0.1, 0.3))

        return f"Stealth typing completed: {len(words)} words typed"

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: float = 1000,
    ) -> bool:
        """
        Perform a curved swipe gesture that mimics natural hand movement.

        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds (default: 1000)

        Returns:
            True if successful, False otherwise
        """
        await self._ensure_connected()

        try:
            # Generate curved path
            path_points = generate_curved_path(start_x, start_y, end_x, end_y)

            # Start touch at first point
            x0, y0 = path_points[0]
            await self.device.shell(f"input motionevent DOWN {x0} {y0}")

            # Calculate delay between points
            delay_between_points = duration_ms / 1000 / len(path_points)

            # Move through intermediate points
            for x, y in path_points[1:]:
                await asyncio.sleep(delay_between_points)
                await self.device.shell(f"input motionevent MOVE {x} {y}")

            # End touch at last point
            x_end, y_end = path_points[-1]
            await self.device.shell(f"input motionevent UP {x_end} {y_end}")

            await asyncio.sleep(duration_ms / 1000)
            return True
        except Exception:
            return False
