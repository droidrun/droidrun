"""
Environment management utilities for AndroidWorld benchmarks.
"""

import os
import sys
import logging
from typing import Optional

logger = logging.getLogger("android_world_bench")


def initialize_environment(
    adb_path: str = "adb", console_port: int = 5554, perform_setup: bool = False
) -> Optional[object]:
    """Initialize the AndroidWorld environment.

    Args:
        adb_path: Path to ADB executable
        console_port: Emulator console port
        perform_setup: Whether to perform initial emulator setup

    Returns:
        Initialized environment or None if initialization failed
    """
    logger.info(f"Initializing Android environment with ADB at {adb_path}")

    try:
        # Import from AndroidWorld
        from android_world.env import env_launcher

        # Load and setup environment
        env = env_launcher.load_and_setup_env(
            console_port=console_port,
            emulator_setup=perform_setup,
            adb_path=adb_path,
        )

        # Reset environment to start from a clean state
        env.reset(go_home=True)

        logger.info("Android environment initialized successfully")
        return env
    except Exception as e:
        logger.exception(f"Failed to initialize Android environment: {e}")
        return None


async def get_device_serial() -> Optional[str]:
    """Get the serial number of the connected Android device.

    Returns:
        Device serial number or None if no device found
    """
    try:
        from droidrun.tools import DeviceManager

        device_manager = DeviceManager()
        devices = await device_manager.list_devices()

        if not devices:
            logger.error(
                "No devices found. Make sure an emulator or device is running."
            )
            return None

        device_serial = devices[0].serial
        logger.info(f"Using device with serial: {device_serial}")
        return device_serial
    except Exception as e:
        logger.exception(f"Error getting device serial: {e}")
        return None
