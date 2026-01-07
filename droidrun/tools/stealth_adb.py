"""
Stealth ADB Tools - Human-like interaction patterns for Android devices.

This module provides stealthy UI interactions with randomization and curved gestures
to mimic natural human behavior on Android devices.
"""

import asyncio
import math
import random
from typing import List, Tuple

from droidrun.tools.adb import AdbTools
from droidrun.tools.geometry import find_clear_point, rects_overlap


def ease_in_out_cubic(t: float) -> float:
    """
    Smooth acceleration/deceleration curve for human-like velocity profiling.

    Humans naturally accelerate at the start and decelerate at the end of swipes.
    This cubic easing function creates that natural motion pattern.

    Args:
        t: Progress value between 0.0 and 1.0

    Returns:
        Eased value between 0.0 and 1.0
    """
    if t < 0.5:
        return 4 * t**3
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def perlin_noise_1d(x: float, seed: int = 0) -> float:
    """
    Simple 1D Perlin-like noise for micro-jitter simulation.

    Creates smooth, natural-looking random variations that simulate hand tremor.
    Unlike pure random noise, this creates coherent patterns that look more organic.

    Args:
        x: Input value (position along the swipe path)
        seed: Random seed for reproducibility

    Returns:
        Noise value between -1.0 and 1.0
    """
    # Simple implementation using sine waves with pseudo-random frequencies
    # This avoids external dependencies while providing smooth noise
    random.seed(seed + int(x * 1000))
    freq1 = random.uniform(0.5, 1.5)
    freq2 = random.uniform(2.0, 3.0)
    freq3 = random.uniform(4.0, 6.0)

    noise = (
        math.sin(x * freq1) * 0.5
        + math.sin(x * freq2) * 0.3
        + math.sin(x * freq3) * 0.2
    )

    return noise


def generate_curved_path(
    start_x: int, start_y: int, end_x: int, end_y: int, num_points: int = 15
) -> List[Tuple[int, int]]:
    """
    Generate a curved path using a quadratic Bezier curve with human-like characteristics.

    Enhancements over basic Bezier curves:
    - Velocity profiling: Ease-in/ease-out using cubic easing for natural acceleration
    - Micro-jitter: Perlin noise to simulate hand tremor
    - Randomized control points for organic curves

    Args:
        start_x: Starting X coordinate
        start_y: Starting Y coordinate
        end_x: Ending X coordinate
        end_y: Ending Y coordinate
        num_points: Number of intermediate points to generate

    Returns:
        List of (x, y) coordinate tuples along the curve with human-like motion
    """
    # Calculate distance to determine curve intensity
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    # For short swipes, use fewer points but still maintain curve
    if distance <= 100:
        num_points = max(5, int(num_points / 3))

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

    # Generate random seed for consistent noise across this swipe
    noise_seed = random.randint(0, 10000)

    # Jitter intensity scales with distance (subtle for all swipes)
    jitter_intensity = min(2.0, distance * 0.01)

    # Generate points along quadratic Bezier curve with velocity profiling
    points = []
    for i in range(num_points):
        # Linear progress
        linear_t = i / (num_points - 1)

        # Apply ease-in-out cubic for velocity profiling
        # This creates natural acceleration at start and deceleration at end
        eased_t = ease_in_out_cubic(linear_t)

        # Quadratic Bezier formula: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        x = (
            (1 - eased_t) ** 2 * start_x
            + 2 * (1 - eased_t) * eased_t * control_x
            + eased_t**2 * end_x
        )
        y = (
            (1 - eased_t) ** 2 * start_y
            + 2 * (1 - eased_t) * eased_t * control_y
            + eased_t**2 * end_y
        )

        # Add micro-jitter using Perlin-like noise to simulate hand tremor
        jitter_x = perlin_noise_1d(linear_t * 10, noise_seed) * jitter_intensity
        jitter_y = perlin_noise_1d(linear_t * 10, noise_seed + 1000) * jitter_intensity

        # Apply jitter (subtle hand shake)
        final_x = int(x + jitter_x)
        final_y = int(y + jitter_y)

        points.append((final_x, final_y))

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

            # Check if the word input failed
            if "Error" in result or "failed" in result:
                return f"Stealth typing failed at word {i + 1}/{len(words)}: {result}"

            # Add space after word (except for last word)
            if i < len(words) - 1:
                space_result = await super().input_text(" ", index=-1, clear=False)
                # Check if space input failed
                if "Error" in space_result or "failed" in space_result:
                    return f"Stealth typing failed adding space after word {i + 1}: {space_result}"
                # Random delay between words (100-300ms)
                await asyncio.sleep(random.uniform(0.1, 0.3))

        return f"Stealth typing completed: {len(words)} words typed successfully"

    async def tap_on_index(self, index: int) -> str:
        """Tap on element by index, avoiding overlapping elements."""
        await self._ensure_connected()
        try:
            def find_element_by_index(elements, target_index):
                for item in elements:
                    if item.get("index") == target_index:
                        return item
                    result = find_element_by_index(item.get("children", []), target_index)
                    if result:
                        return result
                return None

            def collect_all_elements(elements):
                result = []
                for item in elements:
                    result.append(item)
                    result.extend(collect_all_elements(item.get("children", [])))
                return result

            if not self.clickable_elements_cache:
                raise ValueError("No UI elements cached. Call get_state first.")

            element = find_element_by_index(self.clickable_elements_cache, index)
            if not element:
                raise ValueError(f"No element found with index {index}")

            bounds_str = element.get("bounds")
            if not bounds_str:
                raise ValueError(f"Element {index} has no bounds")

            target_bounds = tuple(map(int, bounds_str.split(",")))
            left, top, right, bottom = target_bounds

            all_elements = collect_all_elements(self.clickable_elements_cache)
            blockers = []
            for el in all_elements:
                el_idx = el.get("index")
                el_bounds_str = el.get("bounds")
                if el_idx is not None and el_idx > index and el_bounds_str:
                    el_bounds = tuple(map(int, el_bounds_str.split(",")))
                    if rects_overlap(target_bounds, el_bounds):
                        blockers.append(el_bounds)

            point = find_clear_point(target_bounds, blockers)
            if not point:
                raise ValueError(f"Element {index} is fully obscured by overlapping elements")

            cx, cy = point

            # Add randomization around the clear point
            width, height = right - left, bottom - top
            x_range = max(5, int(width * 0.2))
            y_range = max(5, int(height * 0.2))

            x = cx + random.randint(-x_range // 2, x_range // 2)
            y = cy + random.randint(-y_range // 2, y_range // 2)

            x = max(left + 2, min(x, right - 2))
            y = max(top + 2, min(y, bottom - 2))

            await self.device.click(x, y)
            print(f"Tapped element with index {index} at coordinates ({x}, {y})")

            response_parts = []
            response_parts.append(f"Tapped element with index {index}")
            response_parts.append(f"Text: '{element.get('text', 'No text')}'")
            response_parts.append(f"Class: {element.get('className', 'Unknown class')}")
            response_parts.append(f"Coordinates: ({x}, {y})")

            return " | ".join(response_parts)
        except ValueError as e:
            return f"Error: {str(e)}"

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

            return True
        except Exception:
            return False
