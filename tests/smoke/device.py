"""Device lifecycle management via mobilerun SDK."""

import asyncio
import logging
import time

from mobilerun import AsyncMobilerun
from mobilerun.types.device import Device
from mobilerun.types.devices.state_ui_response import StateUiResponse

logger = logging.getLogger("smoke")


async def provision_device(client: AsyncMobilerun) -> Device:
    """Provision a temporary emulated device."""
    logger.info("Provisioning device...")
    device = await client.devices.create(device_type="dedicated_emulated_device")
    logger.info(f"Device created: {device.id} (state={device.state})")
    return device


async def wait_for_ready(
    client: AsyncMobilerun, device_id: str, timeout: float = 180
) -> Device:
    """Wait for the device to reach 'ready' state, with retries on timeout."""
    logger.info(f"Waiting for device {device_id} to be ready (timeout={timeout}s)...")
    deadline = time.monotonic() + timeout

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Device {device_id} not ready after {timeout}s")

        try:
            # Long-poll with remaining time (capped at 60s per request)
            poll_timeout = min(remaining, 60)
            device = await client.devices.wait_ready(device_id, timeout=poll_timeout)
            if device.state == "ready":
                logger.info(f"Device {device_id} is ready")
                return device
            logger.info(f"Device {device_id} state={device.state}, retrying...")
        except Exception as e:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Device {device_id} not ready after {timeout}s") from e
            logger.info(f"Wait poll returned ({e.__class__.__name__}), retrying ({remaining:.0f}s left)...")

        await asyncio.sleep(2)


async def press_home(client: AsyncMobilerun, device_id: str) -> None:
    """Press the Home button to reset device state."""
    await client.devices.actions.global_(device_id, action=2)
    await asyncio.sleep(2)


async def get_ui_state(
    client: AsyncMobilerun, device_id: str
) -> StateUiResponse:
    """Get the current UI state of the device."""
    return await client.devices.state.ui(device_id)


async def terminate_device(client: AsyncMobilerun, device_id: str) -> None:
    """Terminate the device. Best-effort, never raises."""
    try:
        await client.devices.terminate(device_id, extra_body={})
        logger.info(f"Device {device_id} terminated")
    except Exception as e:
        logger.warning(f"Failed to terminate device {device_id}: {e}")
