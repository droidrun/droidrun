"""
Coordinate conversion utilities.

Provides conversion between normalized coordinates and absolute pixel coordinates.
Normalized coordinate range: [0, 1000]
"""

from dataclasses import dataclass
from typing import Tuple

# Normalized coordinate range constant
NORMALIZED_RANGE = 1000


@dataclass
class ScreenSize:
    """
    Screen size data class.
    
    Attributes:
        width: Screen width in pixels
        height: Screen height in pixels
    """
    width: int
    height: int

    @classmethod
    def from_device_context(cls, device_context: dict) -> "ScreenSize":
        """
        Create ScreenSize from device context.
        
        Args:
            device_context: Device context dict containing screen_bounds
            
        Returns:
            ScreenSize instance
        """
        screen_bounds = device_context.get("screen_bounds", {})
        return cls(
            width=screen_bounds.get("width", 1080),
            height=screen_bounds.get("height", 2400)
        )


def normalized_to_absolute(
    norm_x: int,
    norm_y: int,
    screen_size: ScreenSize
) -> Tuple[int, int]:
    """
    Convert normalized coordinates to absolute pixel coordinates.
    
    Args:
        norm_x: Normalized X coordinate [0-1000]
        norm_y: Normalized Y coordinate [0-1000]
        screen_size: Screen size
        
    Returns:
        (abs_x, abs_y) tuple of absolute pixel coordinates
        
    Raises:
        ValueError: If normalized coordinates are out of valid range
    """
    if not (0 <= norm_x <= NORMALIZED_RANGE):
        raise ValueError(f"norm_x must be in [0, {NORMALIZED_RANGE}], got {norm_x}")
    if not (0 <= norm_y <= NORMALIZED_RANGE):
        raise ValueError(f"norm_y must be in [0, {NORMALIZED_RANGE}], got {norm_y}")
    
    abs_x = int(norm_x * screen_size.width / NORMALIZED_RANGE)
    abs_y = int(norm_y * screen_size.height / NORMALIZED_RANGE)
    
    return abs_x, abs_y


def absolute_to_normalized(
    abs_x: int,
    abs_y: int,
    screen_size: ScreenSize
) -> Tuple[int, int]:
    """
    Convert absolute pixel coordinates to normalized coordinates.
    
    Args:
        abs_x: Absolute X coordinate in pixels
        abs_y: Absolute Y coordinate in pixels
        screen_size: Screen size
        
    Returns:
        (norm_x, norm_y) tuple of normalized coordinates [0-1000]
    """
    norm_x = int(abs_x * NORMALIZED_RANGE / screen_size.width)
    norm_y = int(abs_y * NORMALIZED_RANGE / screen_size.height)
    
    # Clamp to valid range
    norm_x = max(0, min(NORMALIZED_RANGE, norm_x))
    norm_y = max(0, min(NORMALIZED_RANGE, norm_y))
    
    return norm_x, norm_y


def normalized_area_to_center(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    screen_size: ScreenSize
) -> Tuple[int, int]:
    """
    Convert normalized area coordinates to center point absolute coordinates.
    
    Args:
        x1: Top-left normalized X coordinate
        y1: Top-left normalized Y coordinate
        x2: Bottom-right normalized X coordinate
        y2: Bottom-right normalized Y coordinate
        screen_size: Screen size
        
    Returns:
        (center_x, center_y) absolute pixel coordinates of area center
    """
    # Calculate normalized center point
    center_norm_x = (x1 + x2) // 2
    center_norm_y = (y1 + y2) // 2
    
    return normalized_to_absolute(center_norm_x, center_norm_y, screen_size)


def bounds_to_normalized(
    bounds_str: str,
    screen_size: ScreenSize
) -> Tuple[int, int, int, int]:
    """
    Convert element bounds string to normalized coordinates.
    
    Args:
        bounds_str: Bounds string in format "left,top,right,bottom"
        screen_size: Screen size
        
    Returns:
        (norm_x1, norm_y1, norm_x2, norm_y2) normalized bounds coordinates
    """
    left, top, right, bottom = map(int, bounds_str.split(","))
    
    norm_x1, norm_y1 = absolute_to_normalized(left, top, screen_size)
    norm_x2, norm_y2 = absolute_to_normalized(right, bottom, screen_size)
    
    return norm_x1, norm_y1, norm_x2, norm_y2
