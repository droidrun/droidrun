"""
DroidRun Tools - Core functionality for Android device control.
"""

from droidrun.tools.adb import AdbTools
from droidrun.tools.ios import IOSTools
from droidrun.tools.stealth_adb import StealthAdbTools
from droidrun.tools.tools import Tools, describe_tools
from droidrun.tools.coordinate import (
    ScreenSize,
    NORMALIZED_RANGE,
    normalized_to_absolute,
    absolute_to_normalized,
    normalized_area_to_center,
    bounds_to_normalized,
)

__all__ = [
    "Tools",
    "describe_tools",
    "AdbTools",
    "IOSTools",
    "StealthAdbTools",
    # Coordinate utilities
    "ScreenSize",
    "NORMALIZED_RANGE",
    "normalized_to_absolute",
    "absolute_to_normalized",
    "normalized_area_to_center",
    "bounds_to_normalized",
]

try:
    from droidrun.tools.cloud import MobileRunTools

    __all__.append("MobileRunTools")
except ImportError:
    pass
